# Daytona USA — Ports

Per-OS build and release notes. Build prerequisites and the Windows walkthrough
live in the [README](README.md); version history is in [CHANGELOG.md](CHANGELOG.md).

## Platforms

| OS | Build | Status |
|----|-------|--------|
| Windows (x64) | `build.ps1` (one shot: tools, SDK CLI, extract, regen, build), or by hand the `win-amd64-release` preset (clang++ GNU driver, MSVC ABI) → `out/build/win-amd64-release` | ✅ v1.1.4 (SDK v0.9.2, D3D12) |
| Steam Deck / Linux (x64) | `linux-amd64-release` preset (any x64 Linux, WSL included) | ✅ v1.1.4 (Vulkan; boot-verified on device at beta 1.3) |
| macOS (arm64) | `mac-arm64-release` preset on an Apple Silicon Mac, bundled by `mac-build/build_app.sh` | ✅ v1.1.4 (Vulkan over MoltenVK) |

The `win-amd64-*` presets name `clang`/`clang++` with `-march=x86-64-v3` and
target `x86_64-pc-windows-msvc`, so configure from a VS developer shell (MSVC
headers and libs on `INCLUDE`/`LIB`, `rc.exe` on PATH); `build.ps1` enters one
itself and puts the standalone LLVM first on PATH. The `win-amd64-vulkan`
preset builds the Vulkan backend on Windows for debugging; it is not shipped.
The `*-arm64` presets are untested.

## SDK and renderer pins

All builds link the `thirdparty/rexglue-sdk` submodule, **pinned at a tagged
upstream release** (currently v0.9.2 of
[rexglue/rexglue-sdk](https://github.com/rexglue/rexglue-sdk)); binaries stamp
`rexglue-v0.9.2-Release`. `cmake --preset win-amd64-release` resolves it with no
flags or env vars; `-DREXSDK_DIR=<path>` overrides it with a live checkout.

The native renderer is written against `thirdparty/plume`, pinned at a commit on
the `daytona/d3d12-fixes` branch of [springah/plume](https://github.com/springah/plume)
(a fork of [renderbag/plume](https://github.com/renderbag/plume) carrying the
D3D12 and MoltenVK fixes this port depends on). `thirdparty/smol-v` is the
SPIR-V compressor used on the Vulkan path.

To bump a pin: fetch in the submodule, check out the tag or commit, build and
run on every platform that ships, commit the gitlink, and regenerate the notices
(below). A pin must stay reachable from a branch or tag on its remote, or
`--recurse-submodules` clones break.

⚠️ Check what a build directory is pinned to before shipping from it —
`grep REXSDK_DIR out/build/<dir>/CMakeCache.txt`. A directory configured with an
explicit `-DREXSDK_DIR` keeps pointing at that checkout, not the submodule,
until it is reconfigured.

**macOS needs Homebrew LLVM.** The SDK's compiler gate rejects AppleClang, so
the `mac-*` presets point at `/opt/homebrew/opt/llvm/bin/clang++`. MoltenVK is
`dlopen`ed at runtime rather than vendored — `brew install molten-vk` to build
and run from a build directory; `mac-build/build_app.sh` bundles a copy into the
`.app` (with an ICD manifest under `Contents/Resources/vulkan/icd.d/`) so the
shipped bundle needs no Homebrew. `build_app.sh` reads the pinned SDK's
`out/mac-arm64` tree, never `REXSDK_DIR`.

## Releases

One SemVer tag per release; per-OS bundles are named
`Daytona-USA-PC-vX.Y.Z.zip`, `Daytona-USA-Deck-vX.Y.Z.tar.gz`,
`Daytona-USA-macOS-vX.Y.Z.zip`. No game content goes into a bundle or a tag,
ever.

Each tag gets a **GitHub Release with the per-OS bundles attached** (from v1.1.4
on; earlier bundles went only to a private download share, and those carried the
game data for a closed circle of testers — nothing like them is ever published).
The release notes are the changelog entry people can link to. A release asset is
public the moment it is uploaded, so rules 2–8 below run on a bundle *before* it
is attached, never after. A platform whose bundle is not restaged yet is simply
absent from that release; say so in the notes rather than linking an older one.

A tag must be an ancestor of `native-renderer` (and of `main`, which tracks it):
`git merge-base --is-ancestor <tag> native-renderer` is the check. The `v1.0.x`
tags and `beta1` point into the stock-renderer history and stay for the record;
nothing is released from them.

Packaging rules — each one is here because it broke once (here or in the
sibling futari port):

1. **Stage into a fresh directory.** Don't repack the previous bundle in place;
   the working copy collects debug leftovers and run logs.
2. **A public bundle carries no game data.** Ship the executable, the SDK
   runtime (`rexruntime`), the two GPU plugins (`rexgpu-plume`, `rexgpu-xenos`
   — the emulated renderer stays selectable as the reference), `daytona.ini`
   from the repo root, `extract_game.py` flattened beside the executable (plus
   `extract_game.bat` on Windows), `RUN.txt` copied from
   [`dist/RUN.release.txt`](dist/RUN.release.txt), an empty `assets/` holding a
   `PUT_GAME_FILES_HERE.txt`, and [`THIRD-PARTY-NOTICES.txt`](THIRD-PARTY-NOTICES.txt).
   Linux adds [`dist/run.sh`](dist/run.sh) and `libTracyClient.so` (the runtime
   links it). Everything on that list is in this repo or produced by the build,
   so the rule can be executed from a clean clone. The executable carries its
   icon itself (`src/daytona.ico`). `assets/` contents are the game — they never
   go in a bundle that leaves your machine. Grep the staged tree for `.xex`
   before zipping.

   ⚠️ Note what "no game data" does and does not mean: the executable *is* the
   game's own program code, recompiled (daytona is a single-module title, so it
   is all in the exe). That is unavoidable in a binary port, and it is why
   `RUN.txt` and the README say the bundles are for owners only — do not
   describe a bundle as containing no game content. Rule 2 is about *data*: no
   content package, no textures, no audio.

2b. **Licence text travels with the binary.** The runtime statically links
   FFmpeg and libmspack (both LGPL-2.1), plus ~20 permissive components, and the
   plume plugin adds plume (MIT) and smol-v; every one requires its notice to
   accompany a binary distribution. Regenerate the notices whenever a submodule
   pin changes:

   ```sh
   bash scripts/gen_third_party_notices.sh > THIRD-PARTY-NOTICES.txt
   ```

   The macOS bundle also ships MoltenVK (Apache-2.0): copy Homebrew's
   `/opt/homebrew/opt/molten-vk/LICENSE` beside the notices as
   `MoltenVK-LICENSE.txt`. The LGPL also obliges a written offer of source and
   relinkable objects; that offer is in the notices' header and points at the
   SDK submodule at the release tag. That is one more reason a tag must carry
   exactly the SDK pin the bundle was built from.
3. **Zip with 7z**, under a single top-level `Daytona-USA-PC-vX.Y.Z/` folder.
   .NET `Compress-Archive` writes non-spec backslash entries and no wrapper
   folder.
4. **Diff the new bundle's file list against the previous release** before
   publishing, and confirm every difference is intended.
5. **Linux bundles are made on Linux** (`tar czf` in WSL or on the Deck), never
   from Windows, so the Unix mode bits survive and `run.sh` and `daytona`
   arrive executable.
6. **No absolute paths in anything that ships.** `run.sh` resolves its own
   location; `daytona.ini` has none.
7. **Check what a bundle actually contains, not what you meant to put in it.**
   `strings <binary> | grep rexglue-v` reports the SDK each component was built
   against, and `Daytona USA v` the port version; every component in a bundle
   must agree.
8. **macOS: verify the app after the zip round-trip, not in place.** Zip with
   `ditto -c -k --keepParent --norsrc`; extract the result somewhere else and
   `codesign --verify --deep --strict` *that* copy. A non-Mach-O file dropped
   into `Contents/MacOS/` can pass in place and fail after the round-trip. The
   user's `assets/` goes into `Contents/MacOS/` *after* signing, which the
   ad-hoc signature tolerates (beta 1.4 shipped that way); keep everything else
   that is not code out of `MacOS/`. `spctl` reporting "rejected" is expected
   for an ad-hoc, un-notarized app; `RUN.txt` says right-click → Open.
