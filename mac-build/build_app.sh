#!/usr/bin/env bash
# Assemble a self-contained Daytona.app from the mac-arm64 build + extracted assets.
# Daytona is a SINGLE-MODULE recomp (guest code linked into the exe — no guest dylibs).
# Output: ~/Desktop/projects/Daytona.app   (override paths via env vars below)
set -euo pipefail

PROJ="${PROJ:-$HOME/Desktop/projects}"
# Single-config Ninja preset (mac-arm64-release) -> binaries at the build-dir root.
BUILD="${BUILD:-$PROJ/daytona_recomp/out/build/mac-arm64-release}"
# The VENDORED SDK -- its runtime dylibs land in $SDK/out/mac-arm64[/Release].
# Deliberately NOT ${REXSDK_DIR}: that env var still points login shells at
# the external ~/Desktop/projects/rexglue-sdk checkout (the stock port's
# arrangement), which after the 08-09 vendoring is a STALE June tree --
# bundling its librexruntime.dylib against a current exe aborts at launch
# with dyld "Symbol not found" (FLAGS_metadata_root_storage_v, 08-11). This
# repo's SDK is the submodule; only an explicit SDK= overrides it.
SDK="${SDK:-$PROJ/daytona_recomp/thirdparty/rexglue-sdk}"
ASSETS_SRC="${ASSETS_SRC:-$PROJ/daytona_recomp/extracted}"
APP="${APP:-$PROJ/Daytona.app}"
ICON_SRC="${ICON_SRC:-$PROJ/daytona_recomp/src/daytona.ico}"
VER="${VER:-1.1.5}"
C="$APP/Contents"

# Copy a file from the first place it exists (build dir or SDK out dir).
copy_first() {
  local name="$1" dest="$2" c
  for c in "$BUILD/$name" "$SDK/out/mac-arm64/Release/$name" "$SDK/out/mac-arm64/$name"; do
    if [ -f "$c" ]; then cp "$c" "$dest"; return 0; fi
  done
  echo "  ERROR: $name not found in \$BUILD or \$SDK/out/mac-arm64[/Release]" >&2; return 1
}

echo "[1/7] fresh bundle skeleton"
rm -rf "$APP"
mkdir -p "$C/MacOS" "$C/lib" "$C/Resources/vulkan/icd.d"

echo "[2/7] executable + runtime dylibs (single-module: no guest dylibs)"
cp "$BUILD/daytona" "$C/MacOS/"
copy_first librexruntime.dylib "$C/MacOS/"
copy_first libTracyClient.dylib "$C/MacOS/"
# GPU plugins: the SDK loader dlopens librexgpu-<name>.so next to the exe on
# every non-Windows platform -- .so, not .dylib, and dlopen does not mind.
cp "$BUILD/librexgpu-plume.dylib" "$C/MacOS/librexgpu-plume.so"
cp "$BUILD/librexgpu-xenos.dylib" "$C/MacOS/librexgpu-xenos.so"

echo "[3/7] bundled MoltenVK (self-contained)"
MVK="$(readlink -f /opt/homebrew/lib/libMoltenVK.dylib)"
cp "$MVK" "$C/lib/libMoltenVK.dylib"
cat > "$C/Resources/vulkan/icd.d/MoltenVK_icd.json" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "../../../lib/libMoltenVK.dylib",
        "api_version" : "1.4.0",
        "is_portability_driver" : true
    }
}
JSON

echo "[4/7] game data (assets)"
if [ "${SKIP_ASSETS:-0}" = "1" ]; then
  echo "  SKIPPED (SKIP_ASSETS=1)"
else
  ditto "$ASSETS_SRC" "$C/MacOS/assets"
fi

echo "[5/7] icon"
WORK="$(mktemp -d)"
if sips -s format png "$ICON_SRC" --out "$WORK/base.png" >/dev/null 2>&1; then
  ICONSET="$WORK/AppIcon.iconset"; mkdir -p "$ICONSET"
  for s in 16 32 64 128 256 512; do
    sips -z $s $s "$WORK/base.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null 2>&1 || true
    d=$((s*2)); sips -z $d $d "$WORK/base.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null 2>&1 || true
  done
  iconutil -c icns "$ICONSET" -o "$C/Resources/AppIcon.icns" 2>/dev/null \
    && echo "  icon ok" || echo "  icon build failed (non-fatal)"
else
  echo "  could not read .ico (non-fatal, generic icon)"
fi
rm -rf "$WORK"

echo "[6/7] Info.plist"
cat > "$C/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>                  <string>Daytona USA</string>
    <key>CFBundleDisplayName</key>           <string>Daytona USA</string>
    <key>CFBundleIdentifier</key>            <string>com.springah.daytona</string>
    <key>CFBundleExecutable</key>            <string>daytona</string>
    <key>CFBundleIconFile</key>              <string>AppIcon</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key> <string>6.0</string>
    <key>CFBundleShortVersionString</key>    <string>${VER}</string>
    <key>CFBundleVersion</key>               <string>${VER}</string>
    <key>LSMinimumSystemVersion</key>        <string>13.0</string>
    <key>NSHighResolutionCapable</key>       <true/>
    <key>LSApplicationCategoryType</key>     <string>public.app-category.games</string>
</dict>
</plist>
PLIST

echo "[7/7] ad-hoc code signature (required for arm64 launch)"
codesign --force --deep --sign - "$APP" 2>&1 | tail -2 || echo "  (codesign warning, usually fine)"

echo ""
echo "DONE -> $APP"
du -sh "$APP"
