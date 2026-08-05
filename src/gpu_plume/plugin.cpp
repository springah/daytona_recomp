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
    if (r < kRegisterCount) registers_[r] = value;
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
          const uint32_t r = one_reg ? base : base + i;
          if (r < kRegisterCount) registers_[r] = value;
        }
        return true;
      }
      case 1: {   // two registers
        if (reader->read_count() < 2 * sizeof(uint32_t)) return false;
        const uint32_t r0 = packet & 0x7FF;
        const uint32_t r1 = (packet >> 11) & 0x7FF;
        const uint32_t v0 = reader->ReadAndSwap<uint32_t>();
        const uint32_t v1 = reader->ReadAndSwap<uint32_t>();
        if (r0 < kRegisterCount) registers_[r0] = v0;
        if (r1 < kRegisterCount) registers_[r1] = v1;
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
            if (!ExecutePacket(&sub)) break;
          } while (sub.read_count() >= sizeof(uint32_t));
          --indirect_depth_;
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
        // We have no packet-level work to count, so the vblank count is the
        // honest answer: it is the thing that actually advances here.
        StoreGuest(address, ((initiator >> 31) & 0x1) ? uint32_t(vblanks_) : value);
        break;
      }
      case 0x5B: {    // EVENT_WRITE_ZPD -- z-pass done, i.e. an occlusion query
        reader->ReadAndSwap<uint32_t>();   // initiator
        // The guest polls for the "finished" marker. Reporting zero samples is
        // a lie, but a stalled query is a hang, and this renderer does not do
        // occlusion queries yet.
        break;
      }
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
      rex::thread::Sleep(std::chrono::milliseconds(1));
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
    if (source == 0 && (vblanks_ % 3600) == 0) {
      REXLOG_INFO("rexgpu-plume: {} vblanks dispatched, ring read pointer {:08X}",
                  vblanks_, read_ptr_);
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
