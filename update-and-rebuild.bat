@echo off
setlocal

:: ── paths ────────────────────────────────────────────────────────────────────
set ROOT=%~dp0
set LLVM=C:\Program Files\LLVM\bin
set MSVC_BASE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC
set SDK_VER=10.0.26100.0
set SDK_BASE=C:\Program Files (x86)\Windows Kits\10

:: find latest MSVC toolchain version
for /f "delims=" %%v in ('dir /b /ad /o-n "%MSVC_BASE%" 2^>nul') do (
    set MSVC_VER=%%v
    goto :found_msvc
)
:found_msvc

set NINJA_DIR=C:\Users\shant\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe
set CMAKE_DIR=C:\Program Files\CMake\bin
set PATH=%LLVM%;%CMAKE_DIR%;%NINJA_DIR%;%PATH%
set LIB=%MSVC_BASE%\%MSVC_VER%\lib\x64;%SDK_BASE%\Lib\%SDK_VER%\um\x64;%SDK_BASE%\Lib\%SDK_VER%\ucrt\x64
set INCLUDE=%MSVC_BASE%\%MSVC_VER%\include;%SDK_BASE%\Include\%SDK_VER%\um;%SDK_BASE%\Include\%SDK_VER%\ucrt;%SDK_BASE%\Include\%SDK_VER%\shared;%SDK_BASE%\Include\%SDK_VER%\winrt

cd /d "%ROOT%"

:: ── git pull ──────────────────────────────────────────────────────────────────
echo [1/6] Pulling latest changes...
git stash push -m __autoupd >nul 2>&1
git pull
if errorlevel 1 ( echo ERROR: git pull failed & pause & exit /b 1 )
git stash list | findstr /c:"__autoupd" >nul && (
    git stash pop
    if errorlevel 1 ( echo ERROR: merge conflict restoring local changes - resolve by hand & pause & exit /b 1 )
)

echo [2/6] Updating submodules...
git -C thirdparty\rexglue-sdk diff --quiet 2>nul
if errorlevel 1 (
    echo WARNING: thirdparty\rexglue-sdk has uncommitted changes that submodule update may discard.
    echo   Commit or stash your SDK edits first, or close this window to abort.
    pause
)
git submodule update --init --recursive
if errorlevel 1 ( echo ERROR: submodule update failed & pause & exit /b 1 )

:: ── cmake configure ───────────────────────────────────────────────────────────
echo [3/6] Configuring...
cmake --preset win-amd64 -S project
if errorlevel 1 ( echo ERROR: cmake configure failed & pause & exit /b 1 )

:: ── codegen (only if game package present) ────────────────────────────────────
if not exist "assets\default.xex" goto :skip_codegen
echo [4/6] Running codegen...
cmake --build project\out\build\win-amd64 --config Release --target daytona_codegen
echo Applying codegen patch...
"C:\Program Files\Git\usr\bin\patch.exe" -p0 --forward -i patches\daytona_working_codegen.patch
if errorlevel 1 echo WARNING: codegen patch did not apply cleanly - check for .rej files
goto :after_codegen
:skip_codegen
echo [4/6] Skipping codegen (no assets\default.xex found)
:after_codegen

:: ── build ─────────────────────────────────────────────────────────────────────
echo [5/6] Building...
cmake --build project\out\build\win-amd64 --config Release
if errorlevel 1 ( echo ERROR: build failed & pause & exit /b 1 )

:: ── package ready-to-play release to Desktop ──────────────────────────────────
echo [6/6] Packaging ready-to-play release to Desktop...
set RELDIR=%USERPROFILE%\Desktop\Daytona-USA-PC
set BUILDOUT=project\out\build\win-amd64\Release
if not exist "%BUILDOUT%\daytona.exe" ( echo ERROR: built exe not found, skipping packaging & pause & exit /b 1 )
if exist "%RELDIR%" rmdir /s /q "%RELDIR%"
mkdir "%RELDIR%"
for %%f in (daytona.exe rexruntime.dll TracyClient.dll daytona.toml daytona.ini) do (
    if exist "%BUILDOUT%\%%f" copy /y "%BUILDOUT%\%%f" "%RELDIR%\" >nul
)
if exist "extracted\default.xex" (
    xcopy /e /i /q /y "extracted" "%RELDIR%\extracted" >nul
) else (
    echo WARNING: extracted\ game data not found - release will not be playable without it
)

:: daytona.exe needs --game_data_root on the command line (the toml value loads
:: too late to count), so make a Desktop shortcut that passes it.
echo Creating Desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$r='%RELDIR%'; $s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Daytona USA.lnk'); $s.TargetPath=$r+'\daytona.exe'; $s.Arguments='--game_data_root '+$r+'\extracted'; $s.WorkingDirectory=$r; $s.IconLocation=$r+'\daytona.exe,0'; $s.Save()"
if errorlevel 1 echo WARNING: shortcut creation failed

echo.
echo Done! Ready-to-play build at: %RELDIR%
echo   Launch via the "Daytona USA" shortcut on your Desktop.
pause
