// rexgpu-plume -- the GPU plugin for Daytona's native renderer.
//
// This plugin does NOT draw. That is deliberate, and it is the whole design.
//
// Daytona's native renderer (src/dt_plume.cpp, DT_PLUME=1) is driven by the
// game's intercepted D3D9 calls, not by the PM4 ring: by the time a packet
// reaches the ring the draw has already been recorded through the hooks. So
// replacing rexglue's Xenos emulation does not mean reimplementing it -- the
// drawing already has another route.
//
// What it DOES mean was measured, not guessed. Running with no GPU plugin at
// all (DT_PLUME_NOGPU=1, commit e5c1bdc) gets the guest through boot and shader
// creation and then hangs before the first draw, because a GPU is not only a
// thing that draws. It is also:
//
//   1. a vblank tick, and the interrupt callback the guest waits on;
//   2. the GPU register window at 0x7FC80000;
//   3. something that drains the ring, so the guest does not block writing
//      into one it believes is full.
//
// All three are BOOKKEEPING, and none of them needs to DRAW. This plugin
// supplies exactly those three and discards the draws, which makes
//   --gpu_plugin=plume   vs   --gpu_plugin=xenos
// a no-rebuild A/B between "the native renderer draws" and "Xenos emulation
// draws", with the guest equally happy either way.
//
// ⚠ Presentation stays with the app. dt_plume owns its own window and
// swapchain (commit 997456c), so has_presentation() is false and provider()
// and presenter() stay null. A plugin that claimed presentation would have to
// hand ReXApp a GraphicsProvider it cannot build without a backend.

#include <rex/system/gpu_plugin.h>

#include <rex/chrono/clock.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/logging.h>
#include <rex/memory/ring_buffer.h>
#include <rex/memory/utils.h>
#include <rex/system/function_dispatcher.h>
#include <rex/system/interfaces/graphics.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xmemory.h>
#include <rex/system/xthread.h>
#include <rex/system/xvideo.h>
#include <rex/thread.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstring>
#include <vector>

namespace {

// X_STATUS is a typedef in namespace rex, but X_STATUS_SUCCESS and friends are
// MACROS that cast to an unqualified X_STATUS -- writing the qualified form
// expands to `rex::((X_STATUS)0)`, which does not compile. Pull the name in
// once and use both unqualified.
using rex::X_STATUS;

// The register window the guest reaches through MMIO. Register index is
// (addr & 0xFFFF) / 4, so the file has to cover the whole 16-bit space.
constexpr uint32_t kRegisterCount = 0x5003;

// Register indices we have to answer for by name. Values copied from
// rexglue's own GraphicsSystem::ReadRegister/WriteRegister -- these are what
// the guest's D3D9 runtime actually reads, and inventing our own answers here
// would be inventing hardware.
constexpr uint32_t kReg_RB_EDRAM_TIMING = 0x0F00;
constexpr uint32_t kReg_RB_BC_CONTROL = 0x0F01;
constexpr uint32_t kReg_CP_RB_RPTR = 0x01C4;
constexpr uint32_t kReg_CP_RB_WPTR = 0x01C5;
constexpr uint32_t kReg_D1MODE_V_COUNTER = 0x194C;
constexpr uint32_t kReg_INTERRUPT_STATUS = 0x1951;
constexpr uint32_t kReg_D1MODE_VIEWPORT_SIZE = 0x1961;
constexpr uint32_t kReg_COHER_STATUS_HOST = 0x0A31;
constexpr uint32_t kReg_SCRATCH_UMSK = 0x01DC;
constexpr uint32_t kReg_SCRATCH_ADDR = 0x01DD;
constexpr uint32_t kReg_SCRATCH_REG0 = 0x0578;   // "interrupt sync" per the table
constexpr uint32_t kReg_VGT_EVENT_INITIATOR = 0x21F9;
constexpr uint32_t kReg_PA_SC_VIZ_QUERY_STATUS_0 = 0x0C44;
constexpr uint32_t kReg_PA_SC_VIZ_QUERY_STATUS_1 = 0x0C45;

// The guest's "nothing written yet" sentinel for the ring write pointer, as
// used by rexglue's command processor.
constexpr uint32_t kWritePointerUnset = 0xBAADF00D;

class PlumeGraphicsSystem final : public rex::system::IGraphicsSystem {
 public:
  PlumeGraphicsSystem() { std::memset(registers_, 0, sizeof(registers_)); }
  ~PlumeGraphicsSystem() override { Shutdown(); }

  // No host presentation: dt_plume owns its own window and swapchain.
  X_STATUS SetupPresentation(rex::ui::WindowedAppContext* app_context) override {
    (void)app_context;
    REXLOG_INFO("rexgpu-plume: no host presentation -- the native renderer owns its window");
    return X_STATUS_SUCCESS;
  }

  bool has_presentation() const override { return false; }

  X_STATUS SetupGuestGpu(rex::runtime::FunctionDispatcher* function_dispatcher,
                         rex::system::KernelState* kernel_state) override {
    if (function_dispatcher == nullptr || kernel_state == nullptr) {
      REXLOG_ERROR("rexgpu-plume: SetupGuestGpu called without a dispatcher or kernel state");
      return X_STATUS_INVALID_PARAMETER;
    }
    function_dispatcher_ = function_dispatcher;
    kernel_state_ = kernel_state;
    memory_ = kernel_state->memory();

    // The GPU register window, 0x7FC80000-0x7FC8FFFF. Without it every guest
    // register access faults instead of reading a value.
    if (!memory_->AddVirtualMappedRange(
            0x7FC80000, 0xFFFF0000, 0x0000FFFF, this,
            reinterpret_cast<rex::runtime::MMIOReadCallback>(ReadRegisterThunk),
            reinterpret_cast<rex::runtime::MMIOWriteCallback>(WriteRegisterThunk))) {
      REXLOG_ERROR("rexgpu-plume: could not map the GPU register window at 0x7FC80000");
      return X_STATUS_UNSUCCESSFUL;
    }

    workers_running_ = true;

    // The ring drainer. It walks the PM4 stream for the packets that write
    // guest memory and discards the rest -- see DrainLoop for why discarding
    // everything is not enough.
    drain_thread_ = rex::system::object_ref<rex::system::XHostThread>(
        new rex::system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
            DrainLoop();
            return 0;
        }));
    drain_thread_->set_name("plume ring drain");
    drain_thread_->Create();

    // The vblank tick. This is the one the guest was actually waiting on.
    vsync_thread_ = rex::system::object_ref<rex::system::XHostThread>(
        new rex::system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
            VsyncLoop();
            return 0;
        }));
    vsync_thread_->set_name("plume vsync");
    vsync_thread_->Create();

    REXLOG_INFO("rexgpu-plume: guest GPU up -- registers mapped, ring drainer and vblank running "
                "(no draws are executed; drawing comes from the D3D9 hooks)");
    return X_STATUS_SUCCESS;
  }

  void SetInterruptCallback(uint32_t callback, uint32_t user_data) override {
    interrupt_callback_ = callback;
    interrupt_callback_data_ = user_data;
    REXLOG_INFO("rexgpu-plume: SetInterruptCallback({:08X}, {:08X})", callback, user_data);
  }

  void InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) override {
    read_ptr_ = 0;
    ring_ptr_ = ptr;
    // Matches rexglue's CommandProcessor::InitializeRingBuffer: the guest
    // stores log2 of the size in QUADWORDS, hence the +3.
    ring_size_ = uint32_t(1) << (size_log2 + 3);
    REXLOG_INFO("rexgpu-plume: ring buffer at {:08X}, {} bytes", ptr, ring_size_);
  }

  void EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) override {
    (void)block_size_log2;   // we write back on every drain, so the frequency
                             // hint only ever makes us MORE prompt
    read_ptr_writeback_ = ptr;
    REXLOG_INFO("rexgpu-plume: read-pointer write-back to {:08X}", ptr);
  }

  void Shutdown() override {
    if (!workers_running_.exchange(false)) return;
    for (auto* t : {&drain_thread_, &vsync_thread_}) {
      if (*t) {
        (*t)->Wait(0, 0, 0, nullptr);
        t->reset();
      }
    }
    REXLOG_INFO("rexgpu-plume: shut down");
  }

 private:
  static uint32_t ReadRegisterThunk(void* ppc_context, PlumeGraphicsSystem* self, uint32_t addr) {
    (void)ppc_context;
    return self->ReadRegister(addr);
  }
  static void WriteRegisterThunk(void* ppc_context, PlumeGraphicsSystem* self, uint32_t addr,
                                 uint32_t value) {
    (void)ppc_context;
    self->WriteRegister(addr, value);
  }

  uint32_t ReadRegister(uint32_t addr) {
    const uint32_t r = (addr & 0xFFFF) / 4;
    // A guest that is stuck polling a register reads the SAME one thousands of
    // times. Nothing else in this plugin can see that, and it is the cheapest
    // way to tell "waiting on us" from "waiting on something else entirely".
    if (r < kRegisterCount) {
      if (++reg_reads_[r] == 2000) {
        REXLOG_WARN("rexgpu-plume: guest has read register {:04X} 2000 times (value {:08X}) "
                    "-- it is probably spinning on it", r, registers_[r]);
      }
    }
    switch (r) {
      case kReg_RB_EDRAM_TIMING:
        return 0x08100748;
      case kReg_RB_BC_CONTROL:
        return 0x0000200E;
      case kReg_D1MODE_V_COUNTER: {
        rex::system::X_VIDEO_MODE mode;
        rex::kernel::xboxkrnl::VdQueryVideoMode(&mode);
        return std::min(uint32_t(mode.display_height), uint32_t(0x0FFF));
      }
      case kReg_INTERRUPT_STATUS:
        return 1;   // always in vblank; the guest only tests the bit
      case kReg_D1MODE_VIEWPORT_SIZE: {
        rex::system::X_VIDEO_MODE mode;
        rex::kernel::xboxkrnl::VdQueryVideoMode(&mode);
        const uint32_t w = std::min(uint32_t(mode.display_width), uint32_t(0x0FFF));
        const uint32_t h = std::min(uint32_t(mode.display_height), uint32_t(0x0FFF));
        return (w << 16) | h;
      }
      default:
        break;
    }
    return r < kRegisterCount ? registers_[r] : 0;
  }

  void WriteRegister(uint32_t addr, uint32_t value) {
    const uint32_t r = (addr & 0xFFFF) / 4;
    if (r == kReg_CP_RB_WPTR) {
      if (wptr_writes_ < 4) {
        REXLOG_INFO("rexgpu-plume: guest set the ring write pointer to {:08X}", value);
      }
      ++wptr_writes_;
      write_ptr_.store(value);
    }
    // Same side-effectful path as packet writes (scratch mirror, COHER flag):
    // rexglue routes MMIO and packets through one WriteRegister for a reason.
    WritePacketRegister(r, value);
  }

  // Consume the ring. Moving the read pointer is NOT enough on its own: the
  // first attempt discarded every packet, and the guest then spun forever in
  // its GPU-wait path (sub_824140B8, 4000 calls with identical arguments).
  //
  // The reason is that a GPU does not only draw -- some packets WRITE MEMORY
  // the guest then polls. Fences, event completions, query results. Those
  // writes are the guest's only evidence that work finished, and discarding
  // them leaves it waiting on a value that will never arrive.
  //
  // So this walks the PM4 stream and services exactly the packets with a
  // memory side effect, skipping everything else by its declared length. The
  // draws are still discarded: they already arrived through the D3D9 hooks.
  void DrainLoop() {
    while (workers_running_) {
      const uint32_t write_ptr = write_ptr_.load();
      if (write_ptr == kWritePointerUnset || write_ptr == read_ptr_ || ring_ptr_ == 0 ||
          memory_ == nullptr) {
        rex::thread::Sleep(std::chrono::milliseconds(1));
        continue;
      }
      rex::memory::RingBuffer reader(memory_->TranslatePhysical(ring_ptr_), ring_size_);
      reader.set_read_offset(read_ptr_ * sizeof(uint32_t));
      reader.set_write_offset(write_ptr * sizeof(uint32_t));
      while (reader.read_count() >= sizeof(uint32_t)) {
        if (!ExecutePacket(&reader)) break;
      }
      read_ptr_ = write_ptr;
      // The guest can learn the read pointer two ways: the write-back address
      // it gave us, and the CP_RB_RPTR register. Answer both -- which one it
      // actually polls is not ours to assume.
      registers_[kReg_CP_RB_RPTR] = read_ptr_;
      if (packets_ < 8) {
        REXLOG_INFO("rexgpu-plume: drained to {:08X} ({} packets walked so far)", read_ptr_,
                    packets_);
      }
      if (read_ptr_writeback_ != 0) {
        // Guest-endian, like every other value the guest reads back.
        rex::memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(read_ptr_writeback_),
                                              read_ptr_);
      }
    }
  }

  // Undo the GPU-side endian swap a packet asks for, so a value lands in guest
  // memory the way the guest expects to read it. The mode is the low two bits
  // of the address, as it is everywhere else on this GPU.
  static uint32_t GpuSwap(uint32_t value, uint32_t endianness) {
    switch (endianness & 0x3) {
      case 1:   // 8in16
        return ((value & 0x00FF00FF) << 8) | ((value & 0xFF00FF00) >> 8);
      case 2:   // 8in32
        return __builtin_bswap32(value);
      case 3:   // 16in32
        return (value >> 16) | (value << 16);
      default:
        return value;
    }
  }

  // A register write from the COMMAND STREAM, with the side effects the guest
  // depends on. Two of them, both copied from rexglue's WriteRegister because
  // they are hardware behaviour, not emulator invention:
  //
  //   - SCRATCH WRITEBACK. A write to SCRATCH_REG0..7 whose bit is set in
  //     SCRATCH_UMSK is mirrored into guest memory at SCRATCH_ADDR + reg*4.
  //     ⚠ This is the swap release latch. D3D's swap sequence is
  //         type-0 write SCRATCH_REG -> WAIT_REG_MEM [mem]==value -> INTERRUPT
  //     with SCRATCH_ADDR aimed at the block the ISR reads its callback from:
  //     the scratch writeback stores the release value, the wait observes it,
  //     and only then does the interrupt fire -- by which time the guest has
  //     filled in the callback. Without the writeback the wait parks forever
  //     (measured: WAIT_REG_MEM mem 10090046 ==1, current=00000000, 1000+
  //     spins) and no swap ever retires.
  //
  //   - COHER_STATUS_HOST writes SET bit 31 (the "flush requested" flag the
  //     next WAIT_REG_MEM poll acknowledges by clearing).
  void WritePacketRegister(uint32_t r, uint32_t value) {
    if (r >= kRegisterCount) return;
    if (r == kReg_COHER_STATUS_HOST) value |= 0x80000000u;
    registers_[r] = value;
    if (r >= kReg_SCRATCH_REG0 && r < kReg_SCRATCH_REG0 + 8) {
      const uint32_t scratch = r - kReg_SCRATCH_REG0;
      if ((registers_[kReg_SCRATCH_UMSK] >> scratch) & 1) {
        const uint32_t addr = registers_[kReg_SCRATCH_ADDR] + scratch * 4;
        rex::memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(addr), value);
        if (scratch_writebacks_ < 8) {
          REXLOG_WARN("rexgpu-plume: scratch writeback -- reg {} value {:08X} -> phys {:08X}",
                      scratch, value, addr);
        }
        ++scratch_writebacks_;
      }
    }
  }

  void StoreGuest(uint32_t address, uint32_t value) {
    const uint32_t endianness = address & 0x3;
    address &= ~uint32_t(0x3);
    rex::memory::store<uint32_t>(memory_->TranslatePhysical(address),
                                 GpuSwap(value, endianness));
  }

  bool ExecutePacket(rex::memory::RingBuffer* reader) {
    ++packets_;
    const uint32_t packet = reader->ReadAndSwap<uint32_t>();
    if (packet == 0) return true;
    const uint32_t type = packet >> 30;
    const uint32_t count = ((packet >> 16) & 0x3FFF) + 1;

    switch (type) {
      case 0: {   // register writes
        if (reader->read_count() < count * sizeof(uint32_t)) return false;
        const uint32_t base = packet & 0x7FFF;
        const bool one_reg = ((packet >> 15) & 0x1) != 0;
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t value = reader->ReadAndSwap<uint32_t>();
          WritePacketRegister(one_reg ? base : base + i, value);
        }
        return true;
      }
      case 1: {   // two registers
        if (reader->read_count() < 2 * sizeof(uint32_t)) return false;
        const uint32_t r0 = packet & 0x7FF;
        const uint32_t r1 = (packet >> 11) & 0x7FF;
        const uint32_t v0 = reader->ReadAndSwap<uint32_t>();
        const uint32_t v1 = reader->ReadAndSwap<uint32_t>();
        WritePacketRegister(r0, v0);
        WritePacketRegister(r1, v1);
        return true;
      }
      case 2:     // nop, carries no data
        return true;
      default:
        break;
    }

    // Type 3.
    if (reader->read_count() < count * sizeof(uint32_t)) return false;
    const uint32_t opcode = (packet >> 8) & 0x7F;
    // DT_PLUME_NOSWEEP=1 reverts the one-sweep side-effect port (bd8f87b) to
    // skip-by-length, because boot progress regressed from FrameRender-and-
    // resolves to a park at KickOff #12 in the same change window. The gate
    // covers the whole blast radius so the A/B is one env var, not a rebuild.
    static const bool no_sweep = std::getenv("DT_PLUME_NOSWEEP") != nullptr;
    if (no_sweep) {
      switch (opcode) {
        case 0x21: case 0x3E: case 0x45: case 0x46: case 0x5A: case 0x4F:
        case 0x23: case 0x2D: case 0x55: case 0x56: case 0x2F: {
          const size_t skip_end = reader->read_offset() + count * sizeof(uint32_t);
          reader->set_read_offset(skip_end);
          return true;
        }
        default:
          break;
      }
    }
    // What the guest actually submits is the whole question when it waits for
    // work it believes is outstanding. Log the opcodes once each.
    if (opcode < 128 && !opcode_seen_[opcode]) {
      opcode_seen_[opcode] = true;
      REXLOG_WARN("rexgpu-plume: first PM4 type-3 opcode {:02X} ({} words)", opcode, count);
    }
    const size_t end = reader->read_offset() + count * sizeof(uint32_t);

    switch (opcode) {
      case 0x3F:      // INDIRECT_BUFFER
      case 0x37: {    // INDIRECT_BUFFER_PFD
        // Its contents can write memory too, so follow it rather than skip it.
        // ⚠ The pointer must be masked to the physical range before it is
        // translated: rexglue round-trips it through CpuToGpu (p & 0x1FFFFFFF)
        // for exactly this reason. Passing the raw pointer translates to the
        // wrong place, the sub-buffer decodes as garbage, and every packet
        // inside it -- which is ALL the real work, since the ring itself only
        // ever carries ME_INIT and INDIRECT_BUFFER -- is silently lost.
        const uint32_t list_ptr = reader->ReadAndSwap<uint32_t>() & 0x1FFFFFFF;
        const uint32_t list_words = reader->ReadAndSwap<uint32_t>() & 0xFFFFF;
        if (indirect_buffers_ < 4) {
          REXLOG_WARN("rexgpu-plume: indirect buffer at {:08X}, {} words", list_ptr, list_words);
        }
        ++indirect_buffers_;
        if (list_words != 0 && indirect_depth_ >= 4) {
          // A silent skip here silently drops every packet inside -- including
          // fences -- and a dropped fence is a guest that waits forever.
          REXLOG_WARN("rexgpu-plume: indirect buffer SKIPPED at depth {} ({} words) -- "
                      "fences inside it are lost", indirect_depth_, list_words);
        }
        if (list_words != 0 && indirect_depth_ < 4) {
          ++indirect_depth_;
          rex::memory::RingBuffer sub(memory_->TranslatePhysical(list_ptr),
                                      list_words * sizeof(uint32_t));
          sub.set_write_offset(list_words * sizeof(uint32_t));
          // ⚠ do/while, NOT while. set_write_offset takes the offset modulo
          // the capacity, so a buffer that is exactly full sets write_offset
          // to 0 -- and read_count() then reads 0 == 0 as EMPTY. A pre-test
          // loop therefore executes nothing at all, silently, which is how
          // every packet in every indirect buffer went missing while the walk
          // looked like it was running. rexglue's do/while is load-bearing.
          do {
            if (!ExecutePacket(&sub)) {
              // Breaking here abandons the REST of the buffer -- the fences
              // sit at the end, so this is exactly how they get dropped.
              REXLOG_WARN("rexgpu-plume: packet failed inside indirect buffer at {:08X} -- "
                          "{} of {} words abandoned",
                          list_ptr, sub.read_count() / 4, list_words);
              break;
            }
          } while (sub.read_count() >= sizeof(uint32_t));
          --indirect_depth_;
        }
        break;
      }
      case 0x3C: {    // WAIT_REG_MEM -- the CPU<->GPU handshake
        // ⚠ This packet is LOAD-BEARING for the swap protocol and skipping it
        // is what kept the boot swap from ever retiring. The guest emits:
        //     WAIT_REG_MEM (wait here) ... INTERRUPT (notify)
        // and only writes the ISR's callback mailbox ([[device+11028]+16])
        // while the CP is parked at the wait. Blow through the wait and the
        // interrupt is delivered to a NULL mailbox -- one-shot, lost, and the
        // present-wait spins forever. Measured, not theorised: under xenos
        // the ISR logs callback=82233568; under a skip-the-wait drain it
        // logged callback=00000000 and the mailbox never filled.
        const uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
        const uint32_t poll_addr = reader->ReadAndSwap<uint32_t>();
        const uint32_t ref = reader->ReadAndSwap<uint32_t>();
        const uint32_t mask = reader->ReadAndSwap<uint32_t>();
        const uint32_t wait = reader->ReadAndSwap<uint32_t>();
        const bool is_memory = (wait_info & 0x10) != 0;
        if (waits_ < 8) {
          REXLOG_WARN("rexgpu-plume: WAIT_REG_MEM {} {:08X} op={} ref={:08X} mask={:08X}",
                      is_memory ? "mem" : "reg", poll_addr, wait_info & 0x7, ref, mask);
        }
        ++waits_;
        uint32_t spins = 0;
        bool matched = false;
        uint32_t value = 0;
        do {
          if (is_memory) {
            value = rex::memory::load<uint32_t>(
                memory_->TranslatePhysical(poll_addr & ~uint32_t(0x3)));
            value = GpuSwap(value, poll_addr & 0x3);
          } else {
            const uint32_t r = poll_addr;
            // Coherency polls wait for the HOST to acknowledge a cache-flush
            // request by clearing the status. There is no cache here, so the
            // honest acknowledgement is immediate.
            if (r == kReg_COHER_STATUS_HOST) registers_[r] = 0;
            value = r < kRegisterCount ? registers_[r] : 0;
          }
          switch (wait_info & 0x7) {
            case 0x1: matched = (value & mask) < ref; break;
            case 0x2: matched = (value & mask) <= ref; break;
            case 0x3: matched = (value & mask) == ref; break;
            case 0x4: matched = (value & mask) != ref; break;
            case 0x5: matched = (value & mask) >= ref; break;
            case 0x6: matched = (value & mask) > ref; break;
            case 0x7: matched = true; break;
            default: matched = false; break;
          }
          if (!matched) {
            if (!workers_running_) return false;
            if (++spins == 1000) {
              REXLOG_WARN("rexgpu-plume: WAIT_REG_MEM parked 1000 spins -- {} {:08X} op={} "
                          "ref={:08X} mask={:08X} current={:08X}",
                          is_memory ? "mem" : "reg", poll_addr, wait_info & 0x7, ref, mask,
                          value);
            }
            rex::thread::Sleep(std::chrono::milliseconds(wait >= 0x100 ? wait / 0x100 : 1));
          }
        } while (!matched);
        break;
      }
      // -- The full guest-visible side-effect surface, ported in ONE sweep
      // from rexglue's generic command_processor.cpp rather than discovered
      // one hang at a time. That file is the complete specification of what a
      // Xenos owes the guest short of drawing; after this block, anything the
      // guest still waits on is provably a product of EXECUTION, not
      // bookkeeping. Every case follows the reference exactly.
      case 0x21: {    // REG_RMW -- register read/modify/write
        const uint32_t rmw_info = reader->ReadAndSwap<uint32_t>();
        const uint32_t and_mask = reader->ReadAndSwap<uint32_t>();
        const uint32_t or_mask = reader->ReadAndSwap<uint32_t>();
        const uint32_t r = rmw_info & 0x1FFF;
        uint32_t value = r < kRegisterCount ? registers_[r] : 0;
        value &= ((rmw_info >> 31) & 0x1) ? registers_[and_mask & 0x1FFF] : and_mask;
        value |= ((rmw_info >> 30) & 0x1) ? registers_[or_mask & 0x1FFF] : or_mask;
        WritePacketRegister(r, value);
        break;
      }
      case 0x3E: {    // REG_TO_MEM -- register value into guest memory
        const uint32_t reg_addr = reader->ReadAndSwap<uint32_t>();
        const uint32_t mem_addr = reader->ReadAndSwap<uint32_t>();
        const uint32_t r = reg_addr & 0x1FFF;
        StoreGuest(mem_addr, r < kRegisterCount ? registers_[r] : 0);
        break;
      }
      case 0x45: {    // COND_WRITE -- conditional write to memory or register
        const uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
        const uint32_t poll_addr = reader->ReadAndSwap<uint32_t>();
        const uint32_t ref = reader->ReadAndSwap<uint32_t>();
        const uint32_t mask = reader->ReadAndSwap<uint32_t>();
        const uint32_t write_addr = reader->ReadAndSwap<uint32_t>();
        const uint32_t write_data = reader->ReadAndSwap<uint32_t>();
        uint32_t value;
        if (wait_info & 0x10) {
          value = rex::memory::load<uint32_t>(
              memory_->TranslatePhysical(poll_addr & ~uint32_t(0x3)));
          value = GpuSwap(value, poll_addr & 0x3);
        } else {
          const uint32_t pr = poll_addr & 0x1FFF;
          value = pr < kRegisterCount ? registers_[pr] : 0;
        }
        bool matched;
        switch (wait_info & 0x7) {
          case 0x1: matched = (value & mask) < ref; break;
          case 0x2: matched = (value & mask) <= ref; break;
          case 0x3: matched = (value & mask) == ref; break;
          case 0x4: matched = (value & mask) != ref; break;
          case 0x5: matched = (value & mask) >= ref; break;
          case 0x6: matched = (value & mask) > ref; break;
          case 0x7: matched = true; break;
          default: matched = false; break;
        }
        if (matched) {
          if (wait_info & 0x100) {
            StoreGuest(write_addr, write_data);
          } else {
            WritePacketRegister(write_addr & 0x1FFF, write_data);
          }
        }
        break;
      }
      case 0x46: {    // EVENT_WRITE -- initiator into VGT_EVENT_INITIATOR
        const uint32_t initiator = reader->ReadAndSwap<uint32_t>();
        WritePacketRegister(kReg_VGT_EVENT_INITIATOR, initiator & 0x3F);
        break;   // remaining words (if any) skipped by declared length
      }
      case 0x5A: {    // EVENT_WRITE_EXT -- screen-extent event, writes 6 shorts
        const uint32_t initiator = reader->ReadAndSwap<uint32_t>();
        uint32_t address = reader->ReadAndSwap<uint32_t>();
        WritePacketRegister(kReg_VGT_EVENT_INITIATOR, initiator & 0x3F);
        address &= ~uint32_t(0x3);
        // Reference values (rexglue): full-surface extents, 8in16-swapped.
        const uint16_t extents[6] = {
            0, uint16_t(8192 >> 3), 0, uint16_t(8192 >> 3), 0, 1};
        uint8_t* dst = memory_->TranslatePhysical<uint8_t*>(address);
        if (dst != nullptr) {
          for (int i = 0; i < 6; ++i) {
            dst[i * 2 + 0] = uint8_t(extents[i] >> 8);
            dst[i * 2 + 1] = uint8_t(extents[i]);
          }
        }
        break;
      }
      case 0x4F: {    // MEM_WRITE_CNTR -- CP_PROG_COUNTER to memory
        const uint32_t mem_addr = reader->ReadAndSwap<uint32_t>();
        StoreGuest(mem_addr, 0);   // we have no microcode counter; 0, like idle
        break;
      }
      case 0x23: {    // VIZ_QUERY -- fake every query visible (reference does)
        const uint32_t dword0 = reader->ReadAndSwap<uint32_t>();
        const uint32_t id = dword0 & 0x3F;
        if (dword0 & 0x100) {
          WritePacketRegister(kReg_VGT_EVENT_INITIATOR, 8);   // VIZQUERY_END
          if (id < 32) {
            registers_[kReg_PA_SC_VIZ_QUERY_STATUS_0] |= uint32_t(1) << id;
          } else {
            registers_[kReg_PA_SC_VIZ_QUERY_STATUS_1] |= uint32_t(1) << (id - 32);
          }
        } else {
          WritePacketRegister(kReg_VGT_EVENT_INITIATOR, 7);   // VIZQUERY_START
        }
        break;
      }
      case 0x2D: {    // SET_CONSTANT -- register ranges by type
        const uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
        const uint32_t index = offset_type & 0x7FF;
        uint32_t rbase;
        switch ((offset_type >> 16) & 0xFF) {
          case 0: rbase = index + 0x4000; break;   // ALU
          case 1: rbase = index + 0x4800; break;   // FETCH
          case 2: rbase = index + 0x4900; break;   // BOOL
          case 3: rbase = index + 0x4908; break;   // LOOP
          case 4: rbase = index + 0x2000; break;   // REGISTERS
          default: rbase = UINT32_MAX; break;
        }
        for (uint32_t i = 0; i + 1 < count; ++i) {
          const uint32_t value = reader->ReadAndSwap<uint32_t>();
          if (rbase != UINT32_MAX) WritePacketRegister(rbase + i, value);
        }
        break;
      }
      case 0x55:      // SET_CONSTANT2 -- raw register index
      case 0x56: {    // SET_SHADER_CONSTANTS -- same shape
        const uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
        const uint32_t rbase = offset_type & 0xFFFF;
        for (uint32_t i = 0; i + 1 < count; ++i) {
          WritePacketRegister(rbase + i, reader->ReadAndSwap<uint32_t>());
        }
        break;
      }
      case 0x2F: {    // LOAD_ALU_CONSTANT -- register ranges from guest memory
        const uint32_t address = reader->ReadAndSwap<uint32_t>() & 0x3FFFFFFF;
        const uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
        const uint32_t index = offset_type & 0x7FF;
        const uint32_t size_dwords = reader->ReadAndSwap<uint32_t>() & 0xFFF;
        uint32_t rbase;
        switch ((offset_type >> 16) & 0xFF) {
          case 0: rbase = index + 0x4000; break;
          case 1: rbase = index + 0x4800; break;
          case 2: rbase = index + 0x4900; break;
          case 3: rbase = index + 0x4908; break;
          case 4: rbase = index + 0x2000; break;
          default: rbase = UINT32_MAX; break;
        }
        const uint8_t* src = memory_->TranslatePhysical<const uint8_t*>(address);
        if (rbase != UINT32_MAX && src != nullptr) {
          for (uint32_t i = 0; i < size_dwords; ++i) {
            WritePacketRegister(rbase + i, rex::memory::load_and_swap<uint32_t>(src + i * 4));
          }
        }
        break;
      }
      case 0x3D: {    // MEM_WRITE -- N words to consecutive addresses
        uint32_t address = reader->ReadAndSwap<uint32_t>();
        for (uint32_t i = 0; i + 1 < count; ++i) {
          StoreGuest(address, reader->ReadAndSwap<uint32_t>());
          address += 4;
        }
        break;
      }
      case 0x58: {    // EVENT_WRITE_SHD -- the VS|PS-done fence
        const uint32_t initiator = reader->ReadAndSwap<uint32_t>();
        const uint32_t address = reader->ReadAndSwap<uint32_t>();
        const uint32_t value = reader->ReadAndSwap<uint32_t>();
        // Bit 31 asks for the GPU's own counter instead of the given value.
        // rexglue writes its swap counter; ours is the count of XE_SWAP
        // packets seen, which advances at the same moments.
        const uint32_t written = ((initiator >> 31) & 0x1) ? uint32_t(swaps_) : value;
        StoreGuest(address, written);
        if (fences_ < 16) {
          REXLOG_WARN("rexgpu-plume: EVENT_WRITE_SHD -> phys {:08X} = {:08X}{}",
                      address & ~uint32_t(3), written,
                      ((initiator >> 31) & 0x1) ? " (counter)" : "");
        }
        ++fences_;
        break;
      }
      case 0x5B: {    // EVENT_WRITE_ZPD -- z-pass done, i.e. an occlusion query
        // ⚠ This packet HAS a memory side effect and skipping it caused an
        // interrupt STORM: 2.7 million re-arms in 40 seconds. The protocol
        // (from rexglue's handler, which is the reference): D3D pre-writes
        // 0xFFFFFEED markers into the sample-count struct at
        // [RB_SAMPLE_COUNT_ADDR] on D3DISSUE_END; the GPU's ZPD event
        // OVERWRITES the struct with real counts; the guest's XPS/tiling
        // callback polls the struct and treats a surviving marker as "query
        // still running" -- and re-queues its command buffer, forever. The
        // earlier version of this case read the packet and wrote nothing,
        // WITH a comment predicting a stalled query would hang. It did.
        //
        // No depth work happens here, so report a fixed non-zero sample count
        // (rexglue's own workaround, cvar default 1000) on the end-marker.
        reader->ReadAndSwap<uint32_t>();   // initiator
        const uint32_t counts_addr = registers_[0x2325];   // RB_SAMPLE_COUNT_ADDR
        if (zpds_ < 12) {
          const uint8_t* q = counts_addr ? memory_->TranslatePhysical<const uint8_t*>(counts_addr)
                                         : nullptr;
          uint32_t w[8] = {};
          if (q) std::memcpy(w, q, 32);
          REXLOG_WARN("rexgpu-plume: ZPD #{} counts_addr={:08X} struct: {:08X} {:08X} {:08X} "
                      "{:08X} {:08X} {:08X} {:08X} {:08X}",
                      zpds_, counts_addr, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        }
        ++zpds_;
        if (counts_addr != 0) {
          uint8_t* p = memory_->TranslatePhysical<uint8_t*>(counts_addr);
          if (p != nullptr) {
            // Struct is LITTLE-endian (le<> fields; D3D writes it swapped).
            auto ld = [&](uint32_t off) {
              uint32_t v;
              std::memcpy(&v, p + off, 4);
              return v;
            };
            const uint32_t kFinished = 0xEDFEFFFFu;   // 0xFFFFFEED as LE bytes
            const bool end_zpass = ld(16) == kFinished && ld(20) == kFinished;
            const bool end_zfail = ld(8) == kFinished && ld(12) == kFinished;
            std::memset(p, 0, 32);
            if (end_zpass || end_zfail) {
              const uint32_t fake = 1000;
              std::memcpy(p + 16, &fake, 4);   // ZPass_A
              std::memcpy(p + 0, &fake, 4);    // Total_A
            }
          }
        }
        break;
      }
      case 0x64:      // XE_SWAP -- rexglue's VdSwap token; count it, the
                      // EVENT_WRITE_SHD counter semantics depend on it
        ++swaps_;
        break;
      case 0x54: {    // INTERRUPT -- from the command stream
        // ⚠ NOT the same interrupt as the vblank one. Source 1 means "the
        // command stream reached here", and the guest's handler branches on
        // that: firing source 0 instead tells it a frame flipped, which is not
        // the thing it is waiting for. The CPU also comes from the packet's
        // mask rather than being assumed.
        const uint32_t cpu_mask = reader->ReadAndSwap<uint32_t>();
        if (cs_interrupts_ < 4) {
          REXLOG_WARN("rexgpu-plume: command-stream INTERRUPT, cpu mask {:02X}", cpu_mask);
        }
        ++cs_interrupts_;
        // Dispatched INLINE, in stream order -- deliberately, after trying it
        // the other way. The ISR reads its callback from a SINGLE mailbox slot
        // ([[device+11028]+16]) that each kick REWRITES via the scratch
        // writeback earlier in the same stream. A deferred dispatch (added
        // when the mailbox read as null -- which was really the missing
        // scratch writeback, not a race) batches notifications, and by
        // delivery time the mailbox holds only the LAST kick's callback:
        // every earlier notification is delivered to the wrong target or
        // lost, the D3D worker never sees its completion, it stops
        // submitting, and the fence parks two behind the cursor for good.
        //
        // The stream itself guarantees the ordering this needs: the scratch
        // writeback that fills the mailbox executes BEFORE this packet, in
        // this same drain. rexglue dispatches inline for the same reason.
        //
        // NOT paced. A 1ms sleep here was tried as a latency-model hypothesis:
        // it proved our zero latency sets the descriptor loop's RATE (65kHz ->
        // 1kHz) but not its EXIT -- and it throttled the loop's legitimate
        // pre-boot churn from a tenth of a second into minutes, so boot never
        // reached FrameRender at all. The loop's exit is content, not time.
        for (uint32_t n = 0; n < 6; ++n) {
          if (cpu_mask & (1u << n)) DispatchInterrupt(1, n);
        }
        break;
      }
      default:
        break;
    }

    // Whatever a handler did or did not read, the packet's declared length is
    // what advances the stream. Trusting a handler to consume exactly the
    // right number of words is how a parser loses sync on the first opcode it
    // gets slightly wrong.
    reader->set_read_offset(end);   // set_read_offset wraps for us
    return true;
  }

  // The vblank tick, at the guest's own refresh rate, banking debt the same
  // way rexglue does: catch up in whole intervals rather than sleeping to the
  // next one, so a late wake-up does not silently drop frames.
  void VsyncLoop() {
    rex::system::X_VIDEO_MODE mode;
    rex::kernel::xboxkrnl::VdQueryVideoMode(&mode);
    const double refresh_hz = std::max(1.0, double(float(mode.refresh_rate)));
    const uint64_t tick_frequency = rex::chrono::Clock::guest_tick_frequency();
    const uint64_t interval =
        std::max(uint64_t(1), uint64_t(double(tick_frequency) / refresh_hz));
    uint64_t last = rex::chrono::Clock::QueryGuestTickCount();

    while (workers_running_) {
      const uint64_t now = rex::chrono::Clock::QueryGuestTickCount();
      while (now - last >= interval) {
        DispatchInterrupt();
        last += interval;
      }
      DeliverPendingCsInterrupts();
      rex::thread::Sleep(std::chrono::milliseconds(1));
    }
  }

  // Is the guest's command-stream interrupt callback registered yet? The ISR
  // (sub_822346A0) reads it from [[user_data+11028]+16]; before D3D stores it,
  // a source-1 dispatch is a notification delivered to nobody. Reading the
  // same two loads the ISR performs is title-specific knowledge, which is
  // acceptable in a plugin that exists for this one title.
  bool GuestCsCallbackReady() const {
    if (interrupt_callback_data_ == 0 || memory_ == nullptr) return false;
    const uint8_t* p = memory_->TranslateVirtual<const uint8_t*>(interrupt_callback_data_ + 11028);
    if (p == nullptr) return false;
    const uint32_t block = rex::memory::load_and_swap<uint32_t>(p);
    if (block == 0) return false;
    const uint8_t* q = memory_->TranslateVirtual<const uint8_t*>(block + 16);
    if (q == nullptr) return false;
    return rex::memory::load_and_swap<uint32_t>(q) != 0;
  }

  // Deliver deferred command-stream interrupts. Called from the vsync thread
  // each tick: once the callback slot is non-null (or the bound expires), every
  // held interrupt is dispatched in submission order.
  void DeliverPendingCsInterrupts() {
    std::vector<uint32_t> masks;
    {
      std::lock_guard<std::mutex> lock(pending_cs_mutex_);
      if (pending_cs_.empty()) return;
      if (!GuestCsCallbackReady() && pending_cs_ticks_++ < 600) return;
      masks.swap(pending_cs_);
      pending_cs_ticks_ = 0;
    }
    for (uint32_t mask : masks) {
      for (uint32_t n = 0; n < 6; ++n) {
        if (mask & (1u << n)) DispatchInterrupt(1, n);
      }
    }
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXLOG_WARN("rexgpu-plume: first deferred command-stream interrupt delivered "
                  "(callback ready={})", GuestCsCallbackReady());
    }
  }

  // `source` is what the guest's handler branches on: 0 = vblank, 1 = the
  // command stream reached an INTERRUPT packet.
  void DispatchInterrupt(uint32_t source = 0, uint32_t cpu = 2) {
    if (interrupt_callback_ == 0 || function_dispatcher_ == nullptr) return;
    auto* thread = rex::system::XThread::GetCurrentThread();
    if (thread == nullptr) return;
    thread->SetActiveCpu(uint8_t(cpu));
    uint64_t args[] = {source, interrupt_callback_data_};
    function_dispatcher_->ExecuteInterrupt(thread->thread_state(), interrupt_callback_, args,
                                           uint32_t(std::size(args)));
    if (source == 0) ++vblanks_;
    if (source == 0 && (vblanks_ % 600) == 0) {
      REXLOG_WARN("rexgpu-plume: {} vblanks; {} INTERRUPT packets seen; read pointer {:08X}",
                  vblanks_, cs_interrupts_, read_ptr_);
    }
  }

  rex::runtime::FunctionDispatcher* function_dispatcher_ = nullptr;
  rex::system::KernelState* kernel_state_ = nullptr;
  rex::memory::Memory* memory_ = nullptr;

  uint32_t registers_[kRegisterCount];

  uint32_t interrupt_callback_ = 0;
  uint32_t interrupt_callback_data_ = 0;

  uint32_t ring_ptr_ = 0;
  uint32_t ring_size_ = 0;
  uint32_t read_ptr_ = 0;
  uint32_t read_ptr_writeback_ = 0;
  std::atomic<uint32_t> write_ptr_{kWritePointerUnset};

  // Indirect buffers can nest; bounded so a corrupt list cannot recurse away
  // the drain thread's stack.
  uint32_t indirect_depth_ = 0;
  uint64_t packets_ = 0;
  uint64_t indirect_buffers_ = 0;
  uint64_t cs_interrupts_ = 0;
  uint64_t waits_ = 0;
  uint64_t scratch_writebacks_ = 0;
  uint64_t fences_ = 0;
  uint32_t swaps_ = 0;
  uint64_t zpds_ = 0;
  std::mutex pending_cs_mutex_;
  std::vector<uint32_t> pending_cs_;
  uint32_t pending_cs_ticks_ = 0;
  bool opcode_seen_[128] = {};
  std::vector<uint32_t> reg_reads_ = std::vector<uint32_t>(kRegisterCount, 0);
  uint64_t wptr_writes_ = 0;

  uint64_t vblanks_ = 0;
  std::atomic<bool> workers_running_{false};
  rex::system::object_ref<rex::system::XHostThread> drain_thread_;
  rex::system::object_ref<rex::system::XHostThread> vsync_thread_;
};

}  // namespace

extern "C" {

REX_GPU_PLUGIN_EXPORT uint32_t rex_gpu_abi_version() {
  return rex::system::kGpuPluginAbiVersion;
}

REX_GPU_PLUGIN_EXPORT rex::system::IGraphicsSystem* rex_gpu_create(
    uint32_t abi_version, const rex::system::GpuCreateInfo* info) {
  if (abi_version != rex::system::kGpuPluginAbiVersion) {
    REXLOG_ERROR("rexgpu-plume: ABI {} requested, this plugin is {}", abi_version,
                 rex::system::kGpuPluginAbiVersion);
    return nullptr;
  }
  // The backend is the app's business: this plugin never creates a device.
  // dt_plume picks its own, so anything the host asks for is acceptable.
  if (info != nullptr && info->backend != nullptr) {
    REXLOG_INFO("rexgpu-plume: host asked for backend '{}' -- not used, the native renderer "
                "creates its own device", info->backend);
  }
  return new PlumeGraphicsSystem();
}

}  // extern "C"
