#!/usr/bin/env bash
# Regenerate the recompiled C++ for daytona from daytona_manifest.toml.
#
# generated/ is gitignored (large, derived). This script reproduces it from the
# committed manifest + function hints (daytona_functions.toml) and the XEX in
# assets/. daytona is a single-module title (only the shell default.xex), so
# this is one codegen call — no per-module isolation dance (cf. futari/sdoj).
#
# Emits:
#   generated/rexglue.cmake          (SDK-discovery + rexglue_setup_target macro)
#   generated/default/sources.cmake  (entrypoint source list)
#   generated/default/daytona_*.cpp  (recompiled guest code + init/register)
#
# Usage: scripts/regen.sh   (run from anywhere)
set -euo pipefail
cd "$(dirname "$0")/.."

# SDK CLI built from the pinned submodule (thirdparty/rexglue-sdk), falling back
# to the shared sibling checkout (../rexglue-sdk). The standalone single-config
# build (README step 2) lands at out/<plat>/rexglue.exe; the Ninja Multi-Config
# preset build adds a Release/ subdir. Try both, or override with REXGLUE=<path>.
if [ -z "${REXGLUE:-}" ]; then
  for cand in \
    thirdparty/rexglue-sdk/out/win-amd64/rexglue.exe \
    thirdparty/rexglue-sdk/out/win-amd64/Release/rexglue.exe \
    thirdparty/rexglue-sdk/out/linux-amd64/rexglue \
    thirdparty/rexglue-sdk/out/linux-amd64/Release/rexglue \
    ../rexglue-sdk/out/win-amd64/rexglue.exe \
    ../rexglue-sdk/out/win-amd64/Release/rexglue.exe \
    ../rexglue-sdk/out/linux-amd64/rexglue \
    ../rexglue-sdk/out/linux-amd64/Release/rexglue; do
    if [ -f "$cand" ]; then REXGLUE="$cand"; break; fi
  done
fi

if [ -z "${REXGLUE:-}" ] || { [ ! -x "$REXGLUE" ] && [ ! -f "$REXGLUE" ]; }; then
  echo "[regen] rexglue CLI not found under thirdparty/rexglue-sdk/out/ or ../rexglue-sdk/out/." >&2
  echo "[regen] build it first (see README step 2) or set REXGLUE=<path>." >&2
  exit 1
fi
echo "[regen] using $REXGLUE"

echo "[regen] codegen daytona (single module)"
"$REXGLUE" codegen daytona_manifest.toml
echo "[regen] done -> generated/"
