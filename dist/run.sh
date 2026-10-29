#!/bin/bash
# Daytona USA (recompiled) -- Steam Deck / Linux launcher. Lives beside the
# daytona binary and resolves its own directory, so the folder can go anywhere.
# X11 (XWayland) is required: the renderer presents into an X11 window.
cd "$(dirname "$0")"
{
  ldd --version | head -1
  ldd ./daytona 2>&1 | grep -iE "not found" || echo "libraries ok"
} > deck-report.txt 2>&1
export LD_LIBRARY_PATH=.
export SDL_VIDEODRIVER=x11
exec ./daytona "$@"
