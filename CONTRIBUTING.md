# Contributing

This is a hobby port maintained in spare time. Issues and pull requests are
welcome; please read the two rules below first, because they are the ones that
actually matter here.

## 1. Never attach game data

No file derived from the game belongs in an issue, a pull request, or this
repository — not `default.xex`, not the XBLA content package, not anything
extracted from it, not the recompiled C++ under `generated/`. CI rejects
tracked game data, and issues containing it will be deleted rather than edited.

If you are reporting a crash, attach the log from `logs/daytona_NNN.log`. Skim
it first: it contains local filesystem paths, and depending on your
`daytona.ini` it may include your user data directory. Redact anything you
would not post publicly.

## 2. Bug reports need a build and a dump you can describe

Because every copy runs on its owner's own dump, "it crashes" is not actionable
on its own. Useful reports include:

- the release version or commit you built from, and your OS;
- the SDK the binary reports — `strings <binary> | grep rexglue-v`;
- confirmation that your package matches the one in the README's "Game files"
  section (a different revision behaves differently, and that is by far the
  most common cause of odd reports);
- the tail of the log around the failure, and which screen or course you were
  on.

## Pull requests

- Keep changes scoped, and match the surrounding style. The shell scripts are
  `set -euo pipefail` and shellcheck-clean at error severity.
- `scripts/regen.sh` output is not tracked. If a change alters codegen input
  (`*.toml`), say what you regenerated and what you observed in-game.
- The native renderer lives in `src/dt_plume.cpp` and the D3D9 hook layer in
  `src/dt_d3d9_hooks.cpp`; rendering changes are judged against the emulated
  renderer (`--gpu_plugin=xenos`), which stays available as the reference.
- Runtime and recompiler bugs live in the SDK submodule and are fixed upstream
  or on the fork the pin points at, then taken by a pin bump; renderer-interface
  bugs likewise go to the plume fork.
- By contributing you agree your changes are licensed under this repository's
  [BSD 3-Clause licence](LICENSE).

## What this project will not do

Requests for game files, links to packages, or help obtaining the game by any
route other than your own console will be closed without discussion.
