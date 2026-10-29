# Changelog

Versions are per-repo; each release is a tag with a GitHub Release, and from
v1.1.4 on the per-OS bundles are attached to it (earlier bundles went only to a
private download share). The `v1.1.x` line is the native-renderer port; `v1.0.x`
was the stock SDK-renderer port it grew out of. See [PORTS.md](PORTS.md) for the
per-platform build and packaging notes.

## v1.1.5 — unreleased

- **`build.ps1` / `build.bat` — one command from a fresh Windows clone to a
  built `daytona.exe`.** The script does the README's Build section in order:
  checks out the SDK and plume submodules if they are missing and repairs
  libmspack's symlinks if they came out as text stubs, installs any missing
  tool through winget (Git, Python 3, CMake, Ninja, LLVM 20+, VS 2022 Build
  Tools with the C++ workload), enters the x64 developer environment the
  clang++ build needs for the MSVC headers and `rc.exe`, builds the ReXGlue
  CLI from the vendored SDK, extracts your content package with
  `scripts/extract_game.py` (`-GameSource <package>`, or the only file in
  `game/`; skipped when `assets/default.xex` and `extracted/` already exist),
  runs `scripts/regen.sh`, then configures and builds the `win-amd64-<Config>`
  preset. The two long compiles log to `out/build/*.log` instead of the
  terminal and are summarised; `-Run` launches the result against
  `extracted/`. `build.bat` is the double-clickable wrapper. Ported from the
  sibling futari port's script; the renderer and runtime are unchanged from
  v1.1.4.

## v1.1.4 — beta 1.4 — built 2026-08-14, published 2026-10-29

The first release with prebuilt bundles attached, for Windows, Steam Deck /
Linux and macOS. The binaries are the beta 1.4 builds of 2026-08-14; what was
added for publication is the packaging and the paperwork around them.

- **Texture heap recycles.** Beta 1.2's fix for the attract-loop freeze only
  enlarged the texture heap (1024 → 4096 slots behind a guard that degraded
  instead of wedging). The heap now has LRU eviction and a free list behind
  that guard, so a session of any length can no longer degrade for good.
- **The remaining Live blades get the handshake.** QuickChat, PlayerReview and
  GameInvite now receive the open/close notification pair instead of the SDK's
  silent-success stub, closing the last menu entries that could hang the menu
  the way Achievements once did.
- **macOS: the Metal layer's drawable size stays in pixels across fullscreen.**
- **Bundles are published on GitHub Releases.** Until now the per-OS bundles
  went only to a private download share and included the game data for a small
  circle of testers. The public bundles carry **no game data**: you add your
  own XBLA content package after unpacking (`RUN.txt` inside each bundle walks
  through it). They do contain the game's own program code, recompiled, so they
  are for people who own the game. See the README's Legal section.
- **`extract_game.bat`** — drag your content package onto it and it unpacks
  into `assets/` beside `daytona.exe`. `extract_game.py` learned the flattened
  bundle layout: beside the executable it defaults to `assets/` for everything
  and finds a `LIVE` package in its own folder.
- **`THIRD-PARTY-NOTICES.txt`** travels with every bundle, generated from the
  licence texts vendored in the SDK and renderer submodules by
  `scripts/gen_third_party_notices.sh`. The runtime statically links FFmpeg and
  libmspack (LGPL-2.1), whose written offer of source is in the file's header.
- **Docs and repo hygiene for a public release.** `CHANGELOG.md`, `NOTICE.md`
  and `CONTRIBUTING.md` added; `PORTS.md` rewritten with the GitHub Releases
  policy and the packaging rules; README gains "Just want to play?", Legal and
  Licence sections and stops describing `main` as the stock port (it tracks
  this branch). The Deck launcher `run.sh` is tracked under `dist/` so a bundle
  can be staged from a clean clone. CI's identity guard now runs on maintainer
  pushes only, so outside pull requests are not failed for their author.

## v1.1.3 — beta 1.3 — 2026-08-11

- **The settings overlay exists under the native renderer.** The SDK never
  stood it up when a GPU plugin declines presentation (no drawer, no F4 bind);
  the app owns the overlay stack now and the native renderer composites it.
  **Start+Select** on a pad opens it too, and `overlay_scale` scales the
  settings panes, not just the frame.
- The overlay font upload no longer records inside the present.

## v1.1.2 — beta 1.2 — 2026-08-11

- **The attract-loop freeze.** Texture heap exhaustion at 1024 slots; the cap is
  now 4096, behind a guard that degrades instead of wedging.
- **First macOS picture.** GPU plugins staged under the `.so` names the loader
  wants, volk finds the bundled MoltenVK, the window code grew its Cocoa branch.

## v1.1.1 — beta 1.1 — 2026-08-09

- Boot no longer serializes behind the swap guard (publish-before-park); the
  swapchain retries from the guest Swap hook, which fixed the Deck's headless
  boot. **The Deck renders** — five Vulkan gaps closed between a black window
  and the title screen.
- `--gpu_plugin=xenos` alone no longer crashes (choosing the xenos plugin
  implies the xenos renderer).
- First-launch note in the READMEs: the first start after a reboot is slow.

## v1.1.0 — beta 1 — 2026-08-07

The native renderer becomes the product: the game's D3D9 calls are intercepted
app-side and drawn through plume (D3D12 / Vulkan) instead of Xenos GPU
emulation, with the `gpu_plume` plugin keeping the guest's GPU bookkeeping
(vblank, interrupts, PM4 ring drain) alive. Friends-and-family Windows build;
the achievements stub that caused the menu freeze found and fixed, display
settings no longer change game speed, one window that the picture fills.

## v1.0.2 — 2026-06-24

Re-ship against the hardened rexglue-sdk v0.8.3.

## v1.0.0 — 2026-06-17

First playable stock port: boot fixed, mouse-and-keyboard controls, 720p
windowed.

## v0.1.1 / v0.1.0 — 2026-06-16

First versioned builds on the shared rexglue-sdk: bare-exe launch with an
`assets/` default and the application icon.
