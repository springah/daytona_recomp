# Daytona USA — Ports

Recompiled PC port of the Xbox 360 XBLA title, built on the **rexglue-sdk** static
recompiler. You must supply your own game dump (`default.xex` + assets) — this repo
ships no game content.

## Platforms
| OS | CMake preset | Status |
|----|--------------|--------|
| Windows (x64) | `win-amd64` | ✅ v1.0.0 |
| Steam Deck / Linux (x64) | `linux-amd64` | ⬜ not built yet |
| macOS (arm64) | `mac-arm64` | 🛠️ build-verified v1.0.0 (`Daytona.app` via `mac-build/`, MoltenVK); runtime not yet confirmed |

## SDK
All builds link the **rexglue-sdk** submodule at `thirdparty/rexglue-sdk`,
**pinned at a tagged SDK release** (currently v0.8.5.1) — `cmake --preset
win-amd64-release` resolves it with no flags or env vars. The presets no longer
read the `REXSDK_DIR` environment variable (the persistent user-profile value
still serves the sibling ports); to build daytona against a live SDK checkout,
pass it explicitly: `cmake --preset win-amd64-release -DREXSDK_DIR=<path>`.
To bump the pin: tag a new SDK release, `git -C thirdparty/rexglue-sdk fetch
--tags && git -C thirdparty/rexglue-sdk checkout vX.Y.Z`, commit the gitlink.

## Build & stage
- **Windows:** `projects\_resync_build.ps1 -ProjDir <repo>` (configures `win-amd64`
  + builds Release). Package: `projects\_stage_release.ps1 -ProjDir <repo>` →
  `out/daytona-release/` (exe + DLLs + `assets/`, double-clickable, no flags).
- **Steam Deck / Linux:** build the `linux-amd64` preset in WSL (`~/sdk-build/`),
  deploy via `Desktop/deck_tools` (`ssh deck`). Native linux-amd64 (no Proton).
- **macOS:** build the `mac-arm64` preset on the MBP (`ssh mbp`, Homebrew LLVM),
  bundle into a `.app` with the futari `mac-build/` script pattern.

## Releases
One SemVer tag `vX.Y.Z`; per-OS zips (`daytona-windows.zip` / `-deck.zip` /
`-macos.zip`) published to the SHARED download share. The repo stays code + tags
only — no GitHub Releases, no game content.
