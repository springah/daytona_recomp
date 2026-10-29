# Notice — what the licence covers

This file states the scope of [LICENSE](LICENSE). It is a clarification, not a
grant, and adds no conditions to the BSD 3-Clause licence itself.

## Covered

The contents of this repository — the host application and native renderer
(`src/`), the tooling (`scripts/`, `extract_game.bat`, `mac-build/`, `dist/`),
the codegen inputs (`*.toml`), the shaders, the build files, and the
documentation — **except for the game-derived file listed below**.

## Not covered

The licence does NOT cover, and this project does not grant any rights to:

  * **Daytona USA** — its code, data, audio, or artwork. All rights are owned
    by SEGA. No game *data* is distributed here: no content package, no game
    executable, no textures, no audio.
  * **The recompiled C++ under `generated/`.** That is a machine translation of
    the game's own executable code, produced on your machine from a copy you
    own. It is a derivative of SEGA's copyrighted work, not of this repository,
    and it is neither distributed here nor licensed by this project. The release
    bundles contain it in compiled form, which is why they are offered only to
    people who own the game, and why nothing in them is covered by the licence.
  * **`src/daytona.ico`** — the application icon, taken from the game's own
    artwork and included for identification purposes only. `CMakeLists.txt`
    embeds it in `daytona.exe` via `src/daytona.rc.in`, and `mac-build/build_app.sh`
    turns it into the `.app` icon, so every build carries it. It is inside
    `src/`, but it is **not** licensed by this project; if you fork this
    repository for anything beyond building the port for yourself, replace it
    with your own mark.

## Third-party components

The ReXGlue SDK (`thirdparty/rexglue-sdk`) is a separate project under its own
BSD 3-Clause licence, itself derived from the Xenia project (Copyright (c) 2022,
Ben Vanik and Xenia project contributors), pinned here as a submodule at a
tagged release. plume (`thirdparty/plume`, MIT) and smol-v (`thirdparty/smol-v`)
are likewise separate projects. The full set of licences that ships with a
binary bundle is in [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).
