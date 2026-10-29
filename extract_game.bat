@echo off
rem ===========================================================================
rem  Daytona USA (recompiled) - game extractor
rem
rem  Drag your Daytona USA XBLA content package onto this file. That is the
rem  LIVE / STFS package copied off your own Xbox 360's storage: a file with
rem  no extension and a long hexadecimal name, whose first four bytes are the
rem  letters "LIVE". Or just double-click this file when the package is
rem  sitting in this same folder.
rem
rem  It unpacks the package into "assets\" next to daytona.exe, which is where
rem  the game looks by default - after this, daytona.exe runs on a plain
rem  double-click.
rem
rem  Requires Python 3 (https://www.python.org/downloads/). This is a thin
rem  wrapper around extract_game.py.
rem ===========================================================================
setlocal enableextensions
title Daytona USA - Game Extractor

rem Work from this script's own folder so "assets\" and package auto-detect land here.
cd /d "%~dp0"

rem Locate the extractor (flattened release-zip layout vs repo layout).
set "EXTRACTOR="
if exist "%~dp0extract_game.py" set "EXTRACTOR=%~dp0extract_game.py"
if not defined EXTRACTOR if exist "%~dp0scripts\extract_game.py" set "EXTRACTOR=%~dp0scripts\extract_game.py"
if not defined EXTRACTOR (
    echo extract_game.py was not found next to this file.
    goto :fail
)

rem Find a Python 3. The Microsoft Store placeholder "python.exe" exits with
rem code 9009 and prints nothing useful, so probe with --version first.
set "PY="
for %%P in (py.exe python3.exe python.exe) do (
    if not defined PY (
        %%P --version >nul 2>&1 && set "PY=%%P"
    )
)
if not defined PY (
    echo Python 3 was not found. Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then run this again.
    goto :fail
)

if "%~1"=="" (
    echo Looking for a LIVE/STFS package in this folder...
    "%PY%" -I "%EXTRACTOR%" --out assets --assets assets
) else (
    "%PY%" -I "%EXTRACTOR%" "%~1" --out assets --assets assets
)
if errorlevel 1 goto :fail

echo.
echo Done. Double-click daytona.exe to play.
goto :end

:fail
echo.
echo Extraction did not complete. Nothing was changed outside assets\.

:end
echo.
pause
exit /b
