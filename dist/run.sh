#!/bin/bash
# Daytona USA (recompiled) -- Steam Deck / Linux launcher. Lives beside the
# daytona binary and resolves its own directory, so the folder can go anywhere.
# X11 (XWayland) is required: the renderer presents into an X11 window.
cd "$(dirname "$0")"
# Under RecompLauncher the report goes in its data folder (RECOMPLAUNCHER_DATA),
# not into the release folder it updates.
{
  ldd --version | head -1
  ldd ./daytona 2>&1 | grep -iE "not found" || echo "libraries ok"
} > "${RECOMPLAUNCHER_DATA:-.}/deck-report.txt" 2>&1
export LD_LIBRARY_PATH=.
export SDL_VIDEODRIVER=x11
# The setup step (--prepare) opens no window, so it must not need a display.
for arg in "$@"; do
  case "$arg" in --prepare | --prepare=true) export SDL_VIDEODRIVER=offscreen ;; esac
done
exec ./daytona "$@"
