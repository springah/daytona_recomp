#include "dt_null_audio.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <rex/audio/sdl/sdl_audio_driver.h>
#include <rex/logging.h>

namespace dt_null_audio {
namespace {

// The 360 submits 256-sample frames at 48 kHz, so a frame is ~5.33 ms. The guest
// audio thread blocks on the semaphore this driver releases, so the release rate
// is what paces it. Releasing immediately inside SubmitFrame would let the loop
// free-run and burn a core; pacing off a timer keeps it near real time.
constexpr auto kFramePeriod = std::chrono::microseconds(5333);

class NullAudioDriver : public rex::audio::AudioDriver {
 public:
  NullAudioDriver(rex::memory::Memory* memory, rex::thread::Semaphore* semaphore)
      : AudioDriver(memory), semaphore_(semaphore) {}

  ~NullAudioDriver() override { Shutdown(); }

  bool Initialize() {
    running_.store(true, std::memory_order_release);
    // Pre-release once so the guest's first Wait does not stall before the pump
    // thread has scheduled.
    semaphore_->Release(1, nullptr);
    pump_ = std::thread([this] {
      while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(kFramePeriod);
        semaphore_->Release(1, nullptr);
      }
    });
    return true;
  }

  void Shutdown() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    // Release once more so a pump blocked on nothing still exits promptly.
    if (pump_.joinable()) pump_.join();
  }

  // Discard the samples. TranslatePhysical(samples_ptr) would give the buffer if
  // anything ever wants to capture audio for an unattended comparison.
  void SubmitFrame(uint32_t /*samples_ptr*/) override {
    ++frames_;
  }

 private:
  rex::thread::Semaphore* semaphore_ = nullptr;
  std::atomic<bool> running_{false};
  std::thread pump_;
  uint64_t frames_ = 0;
};

}  // namespace

std::unique_ptr<rex::audio::AudioSystem> NullAudioSystem::Create(
    rex::runtime::FunctionDispatcher* function_dispatcher) {
  return std::make_unique<NullAudioSystem>(function_dispatcher);
}

NullAudioSystem::NullAudioSystem(rex::runtime::FunctionDispatcher* function_dispatcher)
    : AudioSystem(function_dispatcher) {}

NullAudioSystem::~NullAudioSystem() = default;

X_STATUS NullAudioSystem::CreateDriver(size_t /*index*/, rex::thread::Semaphore* semaphore,
                                       rex::audio::AudioDriver** out_driver) {
  if (!out_driver) return X_STATUS_INVALID_PARAMETER;
  auto* driver = new NullAudioDriver(memory_, semaphore);
  if (!driver->Initialize()) {
    delete driver;
    return X_STATUS_UNSUCCESSFUL;
  }
  *out_driver = driver;
  return X_STATUS_SUCCESS;
}

void NullAudioSystem::DestroyDriver(rex::audio::AudioDriver* driver) {
  auto* null_driver = dynamic_cast<NullAudioDriver*>(driver);
  if (!null_driver) return;
  null_driver->Shutdown();
  delete null_driver;
}

// ── ResilientAudioSystem ────────────────────────────────────────────────────

std::unique_ptr<rex::audio::AudioSystem> ResilientAudioSystem::Create(
    rex::runtime::FunctionDispatcher* function_dispatcher) {
  return std::make_unique<ResilientAudioSystem>(function_dispatcher);
}

ResilientAudioSystem::ResilientAudioSystem(rex::runtime::FunctionDispatcher* function_dispatcher)
    : AudioSystem(function_dispatcher) {}

ResilientAudioSystem::~ResilientAudioSystem() = default;

X_STATUS ResilientAudioSystem::CreateDriver(size_t /*index*/, rex::thread::Semaphore* semaphore,
                                            rex::audio::AudioDriver** out_driver) {
  if (!out_driver) return X_STATUS_INVALID_PARAMETER;

  auto* sdl = new rex::audio::sdl::SDLAudioDriver(memory_, semaphore);
  if (sdl->Initialize()) {
    *out_driver = sdl;
    return X_STATUS_SUCCESS;
  }
  sdl->Shutdown();
  delete sdl;

  // Once per driver, not once per process: the guest creates several, and if a
  // device disappears mid-session the next one should say so too.
  REXLOG_ERROR(
      "AUDIO: no playback device could be opened -- THERE WILL BE NO SOUND. The game is "
      "continuing in silence rather than exiting, which is what a failed audio init used to "
      "do here. Check that an output device is present and enabled (a sleeping display often "
      "takes an HDMI audio endpoint with it), then restart.");

  auto* silent = new NullAudioDriver(memory_, semaphore);
  if (!silent->Initialize()) {
    delete silent;
    return X_STATUS_UNSUCCESSFUL;
  }
  *out_driver = silent;
  return X_STATUS_SUCCESS;
}

void ResilientAudioSystem::DestroyDriver(rex::audio::AudioDriver* driver) {
  if (!driver) return;
  // Either kind may be live at once -- the fallback is per driver, so a session
  // can hold an SDL driver created before a device vanished and a silent one
  // created after.
  if (auto* sdl = dynamic_cast<rex::audio::sdl::SDLAudioDriver*>(driver)) {
    sdl->Shutdown();
    delete sdl;
    return;
  }
  if (auto* silent = dynamic_cast<NullAudioDriver*>(driver)) {
    silent->Shutdown();
    delete silent;
  }
}

}  // namespace dt_null_audio
