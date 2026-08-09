# daytona - ReXGlue Recompiled Project (Daytona USA / Sega Racing Classic XBLA)
#
# SDK boilerplate (SDK discovery, generated sources, rexglue_setup_target) lives
# in generated/rexglue.cmake, which `rexglue codegen` emits. Run scripts/regen.sh
# to (re)generate it together with the recompiled sources under generated/.

cmake_minimum_required(VERSION 3.25)
# 1.1.x on this branch: the native renderer replaced the Xenos GPU plugin as the
# default, which is a different product from the 1.0.x line that `main` still
# builds. The two ship side by side from separate worktrees, so the version has
# to say which one you are holding.
#
# 1.1.1 = beta 1.1: boot no longer serializes behind the swap guard (publish-
# before-park), the swapchain retries from the Swap hook (the Deck headless
# boot), --gpu_plugin=xenos alone no longer crashes, first-launch note in the
# READMEs.
project(daytona VERSION 1.1.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# SDK resolution: default to the pinned submodule when REXSDK_DIR isn't set
# (the presets pass $env{REXSDK_DIR}, which lands here as "" when unset).
# Point REXSDK_DIR elsewhere to build against a live SDK checkout instead.
if(NOT REXSDK_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rexglue-sdk/CMakeLists.txt")
    set(REXSDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rexglue-sdk"
        CACHE PATH "Path to rexglue-sdk source tree" FORCE)
endif()

# generated/ is derived and gitignored, so a fresh clone configures against a
# tree that does not have it yet. Without this, the failure is CMake's own
# "include could not find requested file: generated/rexglue.cmake", which says
# nothing about what to run. Name the command instead.
# (adopted from the sibling repo's cmake/require_codegen.cmake, moved from a
# build-time `cmake -P` to configure time because that is where ours breaks.)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/rexglue.cmake")
    message(FATAL_ERROR
        "generated/rexglue.cmake not found -- the recompiled sources have not "
        "been generated yet.\n"
        "  Run:  scripts/regen.sh\n"
        "(it needs the rexglue CLI built from thirdparty/rexglue-sdk; see README step 2)")
endif()

include(generated/rexglue.cmake)

# Hand-written host sources (the SDK app main + recompiled guest code are added
# by rexglue_setup_target below).
set(DAYTONA_SOURCES
    src/main.cpp
    src/stubs.cpp
    src/daytona_debug.cpp
    # spike/d3d9-hook-trace phase 0: pass-through D3D9 call tracer (generated).
    src/dt_d3d9_trace.cpp
    src/dt_d3d9_hooks.cpp
    # DT_NOAUDIO: silent-but-functional audio for unattended runs.
    src/dt_null_audio.cpp
    src/dt_shader_dump.cpp
    # Guarded guest-memory reads: a wrong reverse-engineered offset must log,
    # not segfault.
    src/dt_guest_mem.cpp
    # Native renderer (spike): plume bring-up + the game's 9 precompiled shaders.
    # dt_shader_blobs.cpp is GENERATED -- see _tools/daytona_d3d9_sigscan/gen_shader_blobs.py.
    src/dt_shader_blobs.cpp
    # dt_blit_blobs.cpp is GENERATED from shaders/blit.hlsl by scripts/gen_blit_blobs.py,
    # and committed, so an ordinary build needs no shader compiler. It is the only
    # shader in this port that is not the guest's: the present blit.
    src/dt_blit_blobs.cpp
    src/dt_plume.cpp
    src/dt_shader_patchtest.cpp
)

# plume (renderbag) -- the RHI the native renderer targets, replacing rexglue's
# Xenos GPU emulation. Vendored as a submodule with its own contrib deps (volk,
# Vulkan-Headers, VulkanMemoryAllocator, D3D12MemoryAllocator), so this does not
# depend on the UnleashedRecomp checkout. D3D12 is the first backend: the shader
# blobs are signed DXIL and verified.
set(PLUME_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/plume EXCLUDE_FROM_ALL)

# smol-v -- the SPIR-V decompressor. dt_shader_blobs stores its SPIR-V smol-v
# encoded because that is what XenosRecomp emits, and until this was linked the
# Vulkan branch of OnShaderCreated was a stub that failed every shader. That
# made a Linux/Deck build pointless: it would start and render nothing.
#
# Vendored (two files, MIT/public-domain, license.txt kept) from the same
# _tools/XenosRecomp checkout that GENERATED the blobs, so the decoder and the
# encoder cannot drift.
add_library(smolv STATIC thirdparty/smol-v/source/smolv.cpp)
target_include_directories(smolv PUBLIC thirdparty/smol-v/source)

# Embed the application icon + VERSIONINFO resource (visible in Explorer /
# taskbar) via a Windows resource script generated with the absolute .ico path
# and project version substituted, so the RC compiler resolves the icon
# regardless of CWD. Guarded on EXISTS so the build still works if the icon is
# ever removed.
set(DAYTONA_ICON_ABS "${CMAKE_CURRENT_SOURCE_DIR}/src/daytona.ico")
if(WIN32 AND EXISTS "${DAYTONA_ICON_ABS}")
    enable_language(RC)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/daytona.rc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/daytona.rc"
        @ONLY)
    add_executable(daytona WIN32 ${DAYTONA_SOURCES} "${CMAKE_CURRENT_BINARY_DIR}/daytona.rc")
elseif(WIN32)
    add_executable(daytona WIN32 ${DAYTONA_SOURCES})
else()
    add_executable(daytona ${DAYTONA_SOURCES})
endif()

# Links rex::runtime, adds the generated entrypoint sources + SDK windowed-app
# main, and applies platform settings (see generated/rexglue.cmake ->
# rexglue_configure_target).
# rexgpu-plume -- the GPU plugin for the native renderer. It supplies the
# guest-side bookkeeping a GPU owes the game (vblank + interrupt callback, the
# register window, draining the ring) and executes no packets: the drawing
# arrives through the D3D9 hooks instead. Staged next to the exe by
# rexglue_setup_target's GPU_PLUGINS below, so --gpu_plugin=plume vs xenos is a
# no-rebuild A/B. See src/gpu_plume/plugin.cpp for why it does not draw.
#
# Links rexruntime ONLY: it already contains the rexcore/rexui object
# libraries, and linking those again duplicates every symbol in them.
add_library(rexgpu-plume SHARED src/gpu_plume/plugin.cpp)
set_target_properties(rexgpu-plume PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(rexgpu-plume PRIVATE rexruntime)

rexglue_setup_target(daytona GPU_PLUGINS xenos plume)

# Version for the window title (see main.cpp OnPreLaunchModule). Passed unquoted
# and stringized in-source so it survives both the clang-cl and clang++ drivers.
target_compile_definitions(daytona PRIVATE DAYTONA_VERSION=${PROJECT_VERSION})

# plume. Linked but inert unless DT_PLUME is set in the environment -- see
# src/dt_plume.cpp.
target_link_libraries(daytona PRIVATE plume smolv)

# X11, for plume's Vulkan backend.
#
# plume calls XGetWindowAttributes (it reads the window size off the X11 handle
# in its Linux RenderWindow) but its CMakeLists never declares the dependency,
# so libplume.a leaves the symbol undefined and the FINAL link is where it
# surfaces -- here. Fixed here rather than in the plume fork because this is
# where the failure is, and because it costs one line instead of a submodule
# commit; worth pushing upstream to springah/plume eventually.
if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(daytona PRIVATE X11::X11)
endif()

# SDL3 headers, for the Linux window lookup in dt_plume's EnsureWindow.
#
# Explicit rather than transitive: the SDK links SDL3::SDL3 PUBLIC from its audio
# subdirectory, but daytona links `rexruntime`, which absorbs rexcore as OBJECT
# libraries -- and that is exactly the shape where usage requirements can fail to
# propagate. Relying on it would mean a header that resolves by luck.
if(UNIX AND NOT APPLE AND TARGET SDL3::SDL3)
    target_link_libraries(daytona PRIVATE SDL3::SDL3)
endif()

# The renderer spike drives plume by hand from intercepted guest calls, so a
# wrong descriptor or a mismatched pipeline shows up as a crash with no message.
# -DDT_PLUME_D3D12_DEBUG=ON turns on the D3D12 validation layer, which names the
# offending call instead. Off by default: it is slow, and it is a diagnostic.
option(DT_PLUME_D3D12_DEBUG "Enable the D3D12 debug layer inside plume" OFF)
if(DT_PLUME_D3D12_DEBUG)
    # plume's debug path also expects the two break-on-severity knobs to be
    # defined; it never provides defaults. Both false: we want the validation
    # MESSAGES, not a debugger break in an unattended run.
    target_compile_definitions(plume PRIVATE
        D3D12_DEBUG_LAYER_ENABLED
        D3D12_DEBUG_LAYER_BREAK_ON_ERROR=false
        D3D12_DEBUG_LAYER_BREAK_ON_WARNING=false)
endif()
if(WIN32)
    target_link_libraries(daytona PRIVATE d3d12 dxgi)
endif()

# ppc_config.h / ppc_detail.h are daytona's local PPC context detail headers,
# pulled in when PPC_INCLUDE_DETAIL is set.
target_include_directories(daytona PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ppc
)
target_compile_definitions(daytona PRIVATE PPC_INCLUDE_DETAIL)

# Recompiler correctness flags (the SDK applies these within its own directory
# scope, which does not reach this target): keep u8"" literals as char[], strict
# IEEE FP for guest determinism, and no strict-aliasing assumptions on the
# generated code. GNU-driver spellings (daytona builds with clang++, not clang-cl).
target_compile_options(daytona PRIVATE
    -fno-strict-aliasing
    -ffp-model=strict
    -fno-char8_t
)
