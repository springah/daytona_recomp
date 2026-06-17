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
All builds link the shared **rexglue-sdk** via the **`REXSDK_DIR`** environment
variable — an external dependency, not a vendored sibling or submodule. Set it
before configuring, e.g. `REXSDK_DIR=C:\Users\shant\Desktop\projects\rexglue-sdk`
(the `_resync_build.ps1` helper sets it automatically).

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
