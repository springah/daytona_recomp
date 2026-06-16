# Daytona USA ReXGlue Recompilation

A static recompilation of **Daytona USA** (Xbox 360 / XBLA, 2011) for native PC
execution, built on the [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk)
(a Xenia-derived runtime + PPC→C++ recompiler).

No copyrighted game files are included. This tree excludes the extracted game
directory, `default.xex`, media archives, audio, images, and build outputs — you
supply your own legally obtained Daytona USA XBLA package.

This is one of a family of Xbox 360 recomps that share the same ReXGlue SDK
fork; daytona builds on the same `main` SDK branch as the
sibling ports (futari, gh, guwange, sdoj, sotn).

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| CMake | 3.25+ | |
| Ninja | any | `winget install Ninja-build.Ninja` |
| LLVM / clang | 20+ (validated on 22) | Standalone [LLVM](https://github.com/llvm/llvm-project/releases); daytona builds with the **clang++** (GNU) driver targeting the MSVC ABI |
| Git + Bash | any | Submodules; `scripts/regen.sh` is a bash script |

**Windows:** the clang++ build targets `x86_64-pc-windows-msvc`, so it needs the
MSVC headers/libs + Windows SDK on `INCLUDE`/`LIB` (a "x64 Native Tools" command
prompt sets these, or set them manually).

**Linux:** Clang 20+, GTK3 dev headers (`libgtk-3-dev` / `gtk3`), and X11/XCB
interop (`libx11-xcb-dev` / `libx11` / `libX11-devel`). The SDK renders via XCB;
on Wayland, XWayland must be running.

## Game files

Provide your own Daytona USA XBLA STFS package in `game/`, then extract it from
the repository root:

```sh
python3 scripts/extract_game.py
```

This unpacks the package into `extracted/` (the runtime game data, passed via
`--game_data_root`) and copies the shell executable to `assets/default.xex` (the
codegen input). `game/`, `extracted/`, and `assets/` are gitignored.

The package this tree was prepared against: title/media `XA-2845`, media ID
`3CA562D4`, 240058368 bytes, sha256
`db2381451a15a4e537154712213a0309e9ddc33d20bdc03421d80c9939893cef`.

## Build

### 1. Clone with submodules

```sh
git clone --recurse-submodules <repo-url> daytona_recomp
```

The `thirdparty/rexglue-sdk` submodule is the shared ReXGlue fork, pinned to
branch `main` (the same SDK the sibling recomps use). If a clean
`git submodule update --init --recursive` cannot find the pinned commit, fetch
the branch from `https://github.com/springah/rexglue-sdk.git`.

> **Windows symlink gotcha:** the SDK's `libmspack` submodule mirrors its sources
> with git symlinks under `thirdparty/libmspack/cabextract/mspack/`. With
> `core.symlinks=false` (the Windows default) git checks them out as plain text
> containing the link target, which breaks the build (`error: expected
> identifier`). Each must be a one-line `#include "<target>"` wrapper. Enable
> `git config core.symlinks true` before checkout, or replace each file's
> contents with `#include "<the path it points to>"`.

### 2. Build the ReXGlue CLI (for codegen)

```sh
cmake -S thirdparty/rexglue-sdk -B thirdparty/rexglue-sdk/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build thirdparty/rexglue-sdk/build --target rexglue
# -> thirdparty/rexglue-sdk/out/win-amd64/rexglue.exe
```

### 3. Regenerate the recompiled code

```sh
scripts/regen.sh
```

This runs `rexglue codegen daytona_manifest.toml`, emitting `generated/`
(`rexglue.cmake` + `generated/default/` with the recompiled C++). daytona is a
single-module title, so this is one codegen pass. Function-entry seeds the
analyzer can't discover on its own live in `daytona_functions.toml`. `generated/`
is gitignored.

### 4. Configure & build

```sh
cmake --preset win-amd64      # or linux-amd64
cmake --build out/build/win-amd64 --config Release --target daytona
```

The preset bakes `REXSDK_DIR=thirdparty/rexglue-sdk`, so the SDK is built from
source as part of the daytona build. Runtime DLLs (`rexruntime.dll`,
`TracyClient.dll`) are staged next to `daytona.exe` automatically.

## Run

A packaged build launches on a plain double-click: `daytona.exe` defaults its
game data root to an `assets/` folder next to the executable, matching the
sibling recomps. No launcher `.bat` or command-line flag is needed. To assemble
one, put `daytona.exe` + `rexruntime.dll` + `TracyClient.dll` (+ optional
`daytona.ini`) next to a copy of the extracted game data renamed to `assets/`.

To run the raw build-output exe (which has no `assets/` beside it), point it at
the extracted game directory with `--game_data_root`, which always overrides the
default:

```bat
out\build\win-amd64\Release\daytona.exe --game_data_root=extracted
```

```sh
# Linux
LD_LIBRARY_PATH="$PWD/thirdparty/rexglue-sdk/out/linux-amd64/Release" \
  out/build/linux-amd64/Release/daytona --game_data_root="$PWD/extracted"
```

If neither an `assets/` folder beside the exe nor `--game_data_root` is present,
startup exits with `[ERROR] --game_data_root was not provided.`

During a healthy run the log prints repeated
`NtQueryInformationFile(XFileXctdCompressionInformation) unimplemented` messages
(the game streams XCTD-compressed assets; the query is a non-fatal stub).
`logs/` and `*.log` are gitignored.

## Repository layout

```text
CMakeLists.txt          links rex::runtime via generated/rexglue.cmake
CMakePresets.json       win-amd64 / linux-amd64 presets (REXSDK_DIR baked in)
daytona_manifest.toml   ReXGlue codegen manifest
daytona_functions.toml  function-entry seeds + analysis tuning
src/                    host sources (main.cpp, stubs.cpp, keyboard_driver, debug)
ppc/                    local PPC detail headers (PPC_INCLUDE_DETAIL)
scripts/regen.sh        codegen driver
scripts/extract_game.py STFS package extractor
generated/              codegen output (gitignored, regenerate via regen.sh)
thirdparty/rexglue-sdk  shared ReXGlue SDK submodule (branch main)
```

## Roadmap

| Feature | Status |
|---------|--------|
| Arbitrary resolution support (including ultrawide 21:9 and wider) | Planned |
| Keyboard control mapping | Planned |
| Android port | Planned |
| Native rendering (replace Xbox 360 GPU emulation with a native Vulkan/D3D12 backend) | Planned |

## Acknowledgements

- [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk) — Xbox 360 runtime + recompiler (Xenia-derived)
- [Xenia](https://github.com/xenia-project) — the emulator ReXGlue's runtime is rooted in
- Original Daytona USA recompilation work by Subarasheese
