// Daytona USA -- silent-but-functional audio backend for unattended runs.
//
// NOT the SDK's rex::audio::nop::NopAudioSystem. That one's CreateDriver returns
// X_STATUS_NOT_IMPLEMENTED and its DestroyDriver calls assert_always(), so it
// reports FAILURE to the guest -- which Daytona treats as fatal (KeBugCheckEx),
// exactly like a missing audio device. It is built for tool/trace-player use
// where audio is never requested, and substituting it kills the game at startup.
// Measured: stock SDL reached 4084 frames, the SDK nop backend reached 0.
//
// This one succeeds, drains every submitted frame, and paces the guest's audio
// thread off a timer so the submit loop advances at roughly the real rate rather
// than free-running.
//
// SDL's own `dummy` target would have been simpler, but the vendored SDL3 build
// has no dummy audio backend compiled in ("Audio target 'dummy' not available").
#pragma once

#include <rex/system/xtypes.h>

#include <rex/audio/audio_driver.h>
#include <rex/audio/audio_system.h>

namespace dt_null_audio {

// X_STATUS lives in namespace rex, and the X_STATUS_* constants are macros that
// expand to ((X_STATUS)N). The SDK audio classes resolve it by enclosing-namespace
// lookup because they sit inside rex::audio; this namespace does not, so alias it
// or the macros fail to compile here.
using X_STATUS = rex::X_STATUS;

class NullAudioSystem : public rex::audio::AudioSystem {
 public:
  explicit NullAudioSystem(rex::runtime::FunctionDispatcher* function_dispatcher);
  ~NullAudioSystem() override;

  static bool IsAvailable() { return true; }
  static std::unique_ptr<rex::audio::AudioSystem> Create(
      rex::runtime::FunctionDispatcher* function_dispatcher);

  X_STATUS CreateDriver(size_t index, rex::thread::Semaphore* semaphore,
                        rex::audio::AudioDriver** out_driver) override;
  void DestroyDriver(rex::audio::AudioDriver* driver) override;
};

// The shipping backend: SDL, but silent instead of dead when there is no device.
//
// The guest treats a failed audio init as FATAL -- it calls KeBugCheckEx and the
// process dies at startup, after the window has appeared but before anything
// renders, which reads to a player as a graphics crash. The SDK's SDL system
// returns X_STATUS_UNSUCCESSFUL when SDL_OpenAudioDeviceStream cannot open the
// default playback device, and there is nothing rare about that: a machine whose
// display has slept has usually taken its HDMI audio endpoint with it, and a
// player who unplugged their headphones has no default device either.
//
// So try SDL, and if it will not open, run the silent driver above rather than
// failing the guest. LOUDLY -- the objection to an automatic fallback was that a
// silent substitution turns "my sound is broken" into an invisible config state,
// and that objection is right. It argues for saying so at ERROR, in a line that
// names the cause and the fix, not for killing the game.
//
// DT_NOAUDIO=1 still forces NullAudioSystem outright, which is what unattended
// runs should use: it never touches an audio device at all.
class ResilientAudioSystem : public rex::audio::AudioSystem {
 public:
  explicit ResilientAudioSystem(rex::runtime::FunctionDispatcher* function_dispatcher);
  ~ResilientAudioSystem() override;

  static bool IsAvailable() { return true; }
  static std::unique_ptr<rex::audio::AudioSystem> Create(
      rex::runtime::FunctionDispatcher* function_dispatcher);

  X_STATUS CreateDriver(size_t index, rex::thread::Semaphore* semaphore,
                        rex::audio::AudioDriver** out_driver) override;
  void DestroyDriver(rex::audio::AudioDriver* driver) override;
};

}  // namespace dt_null_audio
