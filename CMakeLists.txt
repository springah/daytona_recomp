# daytona - ReXGlue Recompiled Project (Daytona USA / Sega Racing Classic XBLA)
#
# SDK boilerplate (SDK discovery, generated sources, rexglue_setup_target) lives
# in generated/rexglue.cmake, which `rexglue codegen` emits. Run scripts/regen.sh
# to (re)generate it together with the recompiled sources under generated/.

cmake_minimum_required(VERSION 3.25)
project(daytona VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(generated/rexglue.cmake)

# Hand-written host sources (the SDK app main + recompiled guest code are added
# by rexglue_setup_target below).
set(DAYTONA_SOURCES
    src/main.cpp
    src/stubs.cpp
    src/keyboard_driver.cpp
    src/daytona_debug.cpp
)

if(WIN32)
    # Embed a VERSIONINFO resource so the version shows in the exe's properties.
    # The .rc is generated with the project version substituted.
    enable_language(RC)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/daytona.rc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/daytona.rc"
        @ONLY)
    add_executable(daytona WIN32 ${DAYTONA_SOURCES} "${CMAKE_CURRENT_BINARY_DIR}/daytona.rc")
else()
    add_executable(daytona ${DAYTONA_SOURCES})
endif()

# Links rex::runtime, adds the generated entrypoint sources + SDK windowed-app
# main, and applies platform settings (see generated/rexglue.cmake ->
# rexglue_configure_target).
rexglue_setup_target(daytona)

# Version for the window title (see main.cpp OnPreLaunchModule). Passed unquoted
# and stringized in-source so it survives both the clang-cl and clang++ drivers.
target_compile_definitions(daytona PRIVATE DAYTONA_VERSION=${PROJECT_VERSION})

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
