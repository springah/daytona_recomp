// dt_prepare -- the windowless setup step (daytona --prepare).
//
// Unpacks the player's Daytona USA XBLA package (the LIVE/STFS file copied
// off their console) into <data_root>/game, the folder `--data_root` plays
// from, so a launcher needs no Python and no extract_game.py. The package is
// read exactly as scripts/extract_game.py reads it, and game/ ends up with the
// same files the script puts in assets/.
#pragma once

#include <filesystem>

namespace dt_prepare {

// Prints `progress <0-1> <text>` lines on stdout as it goes and the reason on
// stderr when it fails. `package` may be empty when the game is already
// unpacked: a rerun after an update finds it and exits at once. Returns the
// process exit code (0 when game/ is ready to play).
int Run(const std::filesystem::path& package, const std::filesystem::path& data_root);

}  // namespace dt_prepare
