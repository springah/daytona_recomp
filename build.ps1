<#
.SYNOPSIS
    One-shot Windows build of daytona_recomp.

.DESCRIPTION
    Does what the README's "Build" section does by hand, in order:

      1. checks out the SDK's and plume's submodules if they are missing, and
         fixes libmspack's git symlinks if they were checked out as text stubs;
      2. installs any missing tool through winget: Git, Python 3, CMake, Ninja,
         LLVM (clang 20+) and Visual Studio 2022 Build Tools with the
         "Desktop development with C++" workload;
      3. enters the Visual Studio x64 developer environment: daytona builds
         with the clang++ GNU driver targeting the MSVC ABI, which is where it
         finds the MSVC headers and libs, and where CMake finds rc.exe;
      4. builds the ReXGlue CLI (the codegen tool) from the vendored SDK;
      5. extracts your XBLA content package into extracted\ (the game data
         root) and assets\default.xex (the codegen input), unless both already
         exist;
      6. runs scripts\regen.sh to generate the recompiled C++ (one codegen
         pass; daytona is a single-module title);
      7. configures with the win-amd64-<Config> preset and builds daytona.exe
         plus rexruntime.dll, rexgpu-plume.dll and rexgpu-xenos.dll.

    The two long compiles (SDK CLI, then the game) write their full output to
    out\build\*.log instead of the terminal, which would otherwise scroll
    through a very large number of clang warnings; the script prints Ninja's
    progress and a summary, and the error lines if a build fails.

    Nothing here downloads game data. You supply the LIVE/STFS content package
    copied off your own Xbox 360's storage -- the README's "Game files" section
    explains which file that is.

.PARAMETER GameSource
    Your Daytona USA XBLA content package: a file with no extension and a long
    hexadecimal name whose first four bytes read "LIVE". Optional when
    assets\default.xex and extracted\ already exist, or when the package is the
    only file in game\.

.PARAMETER Config
    Release (default), RelWithDebInfo or Debug. Selects the matching
    win-amd64-* preset; the build lands in out\build\win-amd64-<config>\.

.PARAMETER NoInstall
    Never run winget. Report what is missing and stop instead.

.PARAMETER Run
    Launch daytona.exe --game_data_root=extracted once the build succeeds.

.EXAMPLE
    .\build.ps1 -GameSource "D:\dumps\58410B1D\000D0000\41C4C9233343BB022C381C4E8E2EF24611E7EDDC58" -Run

.EXAMPLE
    .\build.ps1
    extracted\ and assets\default.xex already exist; just (re)generate and
    (re)build.

.NOTES
    Git symlinks on Windows need Developer Mode (Settings > System > For
    developers) or an elevated shell. winget may show UAC prompts for the
    machine-wide installers (LLVM, CMake, Build Tools).
#>
[CmdletBinding()]
param(
    [string]$GameSource,
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')]
    [string]$Config = 'Release',
    [switch]$NoInstall,
    [switch]$Run
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
# -GameSource is relative to where the user ran us from, not to the repo root
# we are about to cd into.
if ($GameSource -and -not [System.IO.Path]::IsPathRooted($GameSource)) {
    $GameSource = Join-Path (Get-Location).Path $GameSource
}
Set-Location $Root

# ---------------------------------------------------------------- helpers ---

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Note([string]$Message) { Write-Host "    $Message" }
function Fail([string]$Message) {
    Write-Host "`n[build.ps1] $Message" -ForegroundColor Red
    exit 1
}

# Run an external command and stop the script on a non-zero exit code.
function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    Write-Host "    > $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "'$Exe' exited with code $LASTEXITCODE." }
}

# Run a long compile with its output in a log file, not the terminal. The
# redirect happens inside cmd.exe so PowerShell never sees the compiler's
# stderr (under 'Stop' it would turn the first warning into an exception).
# Ninja's "[N/M]" progress is echoed from the log every few seconds.
function Invoke-LoggedBuild([string]$Label, [string[]]$Arguments, [string]$LogPath) {
    Write-Host "    > cmake $($Arguments -join ' ')" -ForegroundColor DarkGray
    Note "Full output: $LogPath"
    $logDir = Split-Path $LogPath -Parent
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
    if (Test-Path $LogPath) { Remove-Item -LiteralPath $LogPath -Force }
    $quoted = @($Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } })
    $cmdLine = "/d /c cmake $($quoted -join ' ') > `"$LogPath`" 2>&1"
    $proc = Start-Process -FilePath $env:ComSpec -ArgumentList $cmdLine -NoNewWindow -PassThru
    # Touch the handle now, or Windows PowerShell may report $null for ExitCode later.
    $null = $proc.Handle
    $last = ''
    $started = Get-Date
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 5
        $tail = @(Get-Content -LiteralPath $LogPath -Tail 40 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\[\d+/\d+\]' })
        if ($tail.Count -gt 0 -and $tail[-1] -ne $last) {
            $last = $tail[-1]
            $elapsed = [int]((Get-Date) - $started).TotalMinutes
            $progress = ($last -replace '^(\[\d+/\d+\]).*$', '$1')
            Write-Host ("`r    {0} {1} ({2} min)   " -f $Label, $progress, $elapsed) -NoNewline
        }
    }
    $proc.WaitForExit()
    Write-Host ''
    $lines = @(Get-Content -LiteralPath $LogPath -ErrorAction SilentlyContinue)
    $warnings = @($lines | Where-Object { $_ -match '(^|\s)warning(\s*:|\s[A-Z]\d{4}:)' }).Count
    $errors = @($lines | Where-Object { $_ -match '(^|\s)(error|fatal error)(\s*:|\s[A-Z]+\d{4}:)|^FAILED:|ninja: build stopped' })
    Note ("{0}: {1} lines, {2} warnings, {3} error lines" -f $Label, $lines.Count, $warnings, $errors.Count)
    if ($proc.ExitCode -ne 0) {
        Write-Host ''
        Write-Host '    ---- error lines (from the log) ----' -ForegroundColor Yellow
        $errors | Select-Object -Last 40 | ForEach-Object { Write-Host "    $_" }
        Fail "$Label failed (exit code $($proc.ExitCode)). The complete output is in $LogPath."
    }
}

# winget writes new installs into the registry PATH; this process only sees
# them after re-reading it.
function Update-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user;$env:Path"
}

function Find-Exe([string]$Name) {
    $cmd = @(Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
    if ($cmd.Count -gt 0) { return $cmd[0].Source }
    return $null
}

function Install-Tool([string]$Label, [string]$Id, [string[]]$Extra = @()) {
    if ($NoInstall) { Fail "$Label is missing (winget id $Id) and -NoInstall was given." }
    if (-not (Find-Exe 'winget')) {
        Fail "$Label is missing and winget is not available. Install it by hand (winget id: $Id) and re-run."
    }
    Step "Installing $Label (winget $Id)"
    $wingetArgs = @('install', '--id', $Id, '--exact', '--source', 'winget',
                    '--accept-source-agreements', '--accept-package-agreements',
                    '--disable-interactivity') + $Extra
    & winget @wingetArgs
    # 0x8A15002B: "already installed" -- only reachable if detection missed it.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
        Fail "winget could not install $Label (exit code $LASTEXITCODE)."
    }
    Update-PathFromRegistry
}

# Python: prefer the py launcher; otherwise a real python.exe. The Microsoft
# Store alias is a stub that exits 9009, so test by running it.
function Find-Python {
    # Windows PowerShell turns redirected stderr into errors under 'Stop'; the
    # probes below may write there, so relax it for this function only.
    $ErrorActionPreference = 'Continue'
    foreach ($candidate in @(@('py', '-3'), @('python'))) {
        if (-not (Find-Exe $candidate[0])) { continue }
        $extra = @($candidate | Select-Object -Skip 1)
        $out = & $candidate[0] @extra --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$out" -match '^Python 3\.') { return $candidate }
    }
    # Output nothing (not $null) so the caller's @() wrapper sees an empty array.
    return
}

# clang.exe (the GNU driver; the presets name bare `clang`/`clang++`). The
# standalone LLVM install first, then whatever PATH offers.
function Find-Clang {
    $candidates = @((Join-Path $env:ProgramFiles 'LLVM\bin\clang.exe'), (Find-Exe 'clang'))
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Get-ClangMajor([string]$Clang) {
    $ErrorActionPreference = 'Continue'
    $first = (& $Clang --version 2>&1 | Select-Object -First 1)
    if ("$first" -match 'clang version (\d+)') { return [int]$Matches[1] }
    return 0
}

# Visual Studio with the C++ toolset. Returns the install path or $null.
$VsCppComponent = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
function Find-VsWhere {
    $p = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $p) { return $p }
    return $null
}
function Find-VsWithCpp {
    $ErrorActionPreference = 'Continue'
    $vswhere = Find-VsWhere
    if (-not $vswhere) { return $null }
    $path = & $vswhere -latest -products * -requires $VsCppComponent -property installationPath 2>$null
    if ($path) { return "$path".Trim() }
    return $null
}

# ------------------------------------------------------------- 1. sources ---

Step 'Checking the repository'
if (-not (Test-Path (Join-Path $Root 'scripts\regen.sh'))) {
    Fail 'Run this from the daytona_recomp checkout (scripts\regen.sh not found next to build.ps1).'
}
if (-not (Test-Path (Join-Path $Root '.git'))) {
    Fail ('This folder is not a git checkout (no .git), so the SDK''s dependency submodules cannot be fetched. ' +
          'A GitHub ZIP download does not work; clone instead: ' +
          'git -c core.symlinks=true -c core.longpaths=true clone --recurse-submodules https://github.com/springah/daytona_recomp')
}

$git = Find-Exe 'git'
if (-not $git) { Install-Tool 'Git' 'Git.Git'; $git = Find-Exe 'git' }
if (-not $git) { Fail 'git is still not on PATH after installing. Open a new terminal and re-run.' }

# Long paths: harmless if already set, and the SDK's dependency trees are deep.
& $git config core.longpaths true | Out-Null

$sdk = Join-Path $Root 'thirdparty\rexglue-sdk'
$plume = Join-Path $Root 'thirdparty\plume'
if (-not (Test-Path (Join-Path $sdk 'thirdparty\fmt\CMakeLists.txt')) -or
    -not (Test-Path (Join-Path $plume 'contrib\volk\volk.h'))) {
    Note 'The SDK / plume submodules are not checked out; fetching them (about 1 GB of history, once).'
    Invoke-Checked $git @('-c', 'core.symlinks=true', '-c', 'core.longpaths=true',
                          'submodule', 'update', '--init', '--recursive')
}

# libmspack uses git symlinks. Checked out without core.symlinks they become
# text stubs and the SDK build dies with "lzxd.c(1,1): error: expected identifier".
$mspack = Join-Path $sdk 'thirdparty\libmspack'
$linkPaths = @(& $git -C $mspack ls-files -s | Where-Object { $_ -match '^120000 ' } |
               ForEach-Object { ($_ -split "`t", 2)[1] })
function Get-BrokenLinks {
    @($linkPaths | Where-Object {
        $item = Get-Item -LiteralPath (Join-Path $mspack $_) -Force -ErrorAction SilentlyContinue
        -not ($item -and $item.LinkType -eq 'SymbolicLink')
    })
}
$broken = @(Get-BrokenLinks)
if ($broken.Count -gt 0) {
    Note "libmspack's symlinks are checked out as text stubs; re-checking them out as links."
    Invoke-Checked $git @('-C', $mspack, 'config', 'core.symlinks', 'true')
    foreach ($p in $broken) { Remove-Item -LiteralPath (Join-Path $mspack $p) -Force -ErrorAction SilentlyContinue }
    Invoke-Checked $git (@('-C', $mspack, 'checkout', '--') + $broken)
    if (@(Get-BrokenLinks).Count -gt 0) {
        Fail ('git could not create symlinks. Turn on Developer Mode (Settings > System > For developers) ' +
              'or run this from an elevated PowerShell, then re-run.')
    }
}

# --------------------------------------------------------------- 2. tools ---

Step 'Checking the toolchain'

# @() keeps a one-element result an array; a bare string would index by character.
$python = @(Find-Python)
if ($python.Count -eq 0) { Install-Tool 'Python 3' 'Python.Python.3.12'; $python = @(Find-Python) }
if ($python.Count -eq 0) { Fail 'Python 3 is still not usable after installing. Open a new terminal and re-run.' }
$pyExe = $python[0]
$pyPre = @($python | Select-Object -Skip 1)
Note "Python: $pyExe $($pyPre -join ' ')"

if (-not (Find-Exe 'cmake')) { Install-Tool 'CMake' 'Kitware.CMake' }
if (-not (Find-Exe 'cmake')) {
    $cm = Join-Path $env:ProgramFiles 'CMake\bin'
    if (Test-Path (Join-Path $cm 'cmake.exe')) { $env:Path = "$cm;$env:Path" }
}
if (-not (Find-Exe 'cmake')) { Fail 'cmake is not on PATH after installing. Open a new terminal and re-run.' }
Note "CMake: $(Find-Exe 'cmake')"

if (-not (Find-Exe 'ninja')) { Install-Tool 'Ninja' 'Ninja-build.Ninja' }
if (-not (Find-Exe 'ninja')) { Fail 'ninja is not on PATH after installing. Open a new terminal and re-run.' }
Note "Ninja: $(Find-Exe 'ninja')"

$clang = Find-Clang
if (-not $clang) { Install-Tool 'LLVM (clang)' 'LLVM.LLVM'; $clang = Find-Clang }
if (-not $clang) { Fail 'clang.exe was not found under C:\Program Files\LLVM\bin or on PATH.' }
$clangMajor = Get-ClangMajor $clang
if ($clangMajor -lt 20) {
    Fail "clang at $clang is version $clangMajor; this project needs LLVM 20 or newer. Update it (winget upgrade --id LLVM.LLVM) and re-run."
}
$llvmBin = Split-Path $clang -Parent
$clangxx = Join-Path $llvmBin 'clang++.exe'
if (-not (Test-Path $clangxx)) { Fail "clang++.exe is missing next to $clang." }
Note "clang: $clang (LLVM $clangMajor)"

$vsPath = Find-VsWithCpp
if (-not $vsPath) {
    $vswhere = Find-VsWhere
    $anyVs = $null
    if ($vswhere) { $anyVs = & $vswhere -latest -products * -property installationPath }
    if ($anyVs) {
        Fail ("Visual Studio is installed at $anyVs but without the C++ toolset. Open Visual Studio Installer, " +
              "Modify, and add 'Desktop development with C++' (it brings the MSVC tools and the Windows SDK), then re-run.")
    }
    Install-Tool 'Visual Studio 2022 Build Tools (C++ workload)' 'Microsoft.VisualStudio.2022.BuildTools' @(
        '--override', '--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended')
    $vsPath = Find-VsWithCpp
}
if (-not $vsPath) { Fail 'The Visual Studio C++ toolset is still not found after installing.' }
Note "Visual Studio: $vsPath"

# bash comes with Git for Windows. Not System32\bash.exe: that is WSL, and
# regen.sh would then run inside Linux with none of these paths.
# git.exe may be found as <Git>\cmd\git.exe or <Git>\mingw64\bin\git.exe
# depending on which directory PATH names, so walk up until bash shows up.
$bash = $null
$gitRoot = Split-Path $git -Parent
for ($i = 0; $i -lt 4 -and $gitRoot -and -not $bash; $i++) {
    $gitRoot = Split-Path $gitRoot -Parent
    foreach ($rel in @('bin\bash.exe', 'usr\bin\bash.exe')) {
        $p = Join-Path $gitRoot $rel
        if (Test-Path $p) { $bash = $p; break }
    }
}
if (-not $bash) { Fail "Git's bash.exe was not found above $git. Install Git for Windows (winget id Git.Git)." }
Note "bash: $bash"

# ----------------------------------------------------------- 3. dev shell ---

Step 'Entering the Visual Studio x64 developer environment'
$devShell = Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
if (-not (Test-Path $devShell)) { Fail "Developer shell module not found at $devShell." }
# VsDevCmd shells out to vswhere.exe by bare name and prints a stray
# "'vswhere.exe' is not recognized" when the Installer folder is not on PATH.
$env:Path = "$(Split-Path (Find-VsWhere) -Parent);$env:Path"
Import-Module $devShell
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64 -no_logo' | Out-Null
if (-not (Find-Exe 'rc')) {
    Fail ('rc.exe is not on PATH inside the developer environment, so the Windows SDK component is missing. ' +
          'Add a Windows 10/11 SDK in Visual Studio Installer and re-run.')
}
Note "rc.exe: $(Find-Exe 'rc')"
# The presets name bare `clang`/`clang++`, so the LLVM we checked above must be
# first on PATH -- ahead of any clang Visual Studio bundles (it is older) and
# ahead of anything the developer environment just prepended.
$env:Path = "$llvmBin;$env:Path"
if ((Find-Exe 'clang') -ne $clang) { Fail "PATH resolves clang to $(Find-Exe 'clang'), not $clang." }
# The developer environment puts Visual Studio's own bundled CMake and Ninja
# ahead of the ones found above, so re-check what will actually run: the
# presets need CMake 3.25+ and older Build Tools bundle less than that.
$cmakeVer = ((& cmake --version | Select-Object -First 1) -replace '^cmake version\s*', '') -replace '[^0-9.].*$', ''
if (-not $cmakeVer -or [version]$cmakeVer -lt [version]'3.25') {
    Fail ("cmake inside the developer environment is '$cmakeVer' at $(Find-Exe 'cmake'); the presets need 3.25 or newer. " +
          'Update Visual Studio Build Tools, or put a newer CMake first on PATH, and re-run.')
}
Note "CMake (in dev shell): $(Find-Exe 'cmake') ($cmakeVer)"
Note "Ninja (in dev shell): $(Find-Exe 'ninja')"

# ------------------------------------------------------------ 4. SDK CLI ---

Step 'Building the ReXGlue CLI (codegen tool) from the vendored SDK'
# The SDK's own win-amd64 preset (README step 2): Ninja Multi-Config, bare
# clang/clang++ from PATH, -march=x86-64-v3. The SSSE3 byte-swap intrinsics in
# its core need that -march; a plain single-config configure without it fails
# in memory.cpp ("always_inline function '_mm_shuffle_epi8' requires target
# feature 'ssse3'").
$sdkBuild = Join-Path $sdk 'out\build\win-amd64'
$logDir = Join-Path $Root 'out\build'
Invoke-Checked 'cmake' @('-S', $sdk, '--preset', 'win-amd64')
Invoke-LoggedBuild 'SDK CLI build' @('--build', $sdkBuild, '--config', 'Release', '--target', 'rexglue') (Join-Path $logDir 'rexglue-cli-build.log')
$rexglue = Join-Path $sdk 'out\win-amd64\Release\rexglue.exe'
if (-not (Test-Path $rexglue)) { Fail "The SDK built but $rexglue is missing." }

# ----------------------------------------------------------- 5. game data ---

Step 'Game data'
$dataOk = (Test-Path (Join-Path $Root 'assets\default.xex')) -and (Test-Path (Join-Path $Root 'extracted\default.xex'))
if ($GameSource) {
    if (-not (Test-Path -LiteralPath $GameSource -PathType Leaf)) { Fail "GameSource not found (or not a file): $GameSource" }
} elseif (-not $dataOk) {
    $gameDir = Join-Path $Root 'game'
    $packages = @()
    if (Test-Path $gameDir) { $packages = @(Get-ChildItem -Path $gameDir -File) }
    if ($packages.Count -eq 1) {
        $GameSource = $packages[0].FullName
        Note "Using game\$($packages[0].Name)."
    } elseif ($packages.Count -gt 1) {
        Fail 'Several files are in game\; pass the content package with -GameSource.'
    } else {
        Fail ('No game data. Pass -GameSource <your XBLA content package> (or copy it into game\), or populate the data ' +
              'yourself so that assets\default.xex and extracted\default.xex exist. The README''s "Game files" ' +
              'section says which file on your console''s storage the package is.')
    }
}
if ($GameSource) {
    Invoke-Checked $pyExe ($pyPre + @('-I', 'scripts\extract_game.py', $GameSource, '--out', 'extracted', '--assets', 'assets'))
} else {
    Note 'assets\default.xex and extracted\ already exist; skipping extraction.'
}

# -------------------------------------------------------------- 6. codegen ---

Step 'Generating the recompiled C++ (scripts\regen.sh)'
$env:REXGLUE = 'thirdparty/rexglue-sdk/out/win-amd64/Release/rexglue.exe'
Invoke-Checked $bash @('--noprofile', '--norc', 'scripts/regen.sh')

# ---------------------------------------------------- 7. configure + build ---

$preset = "win-amd64-$($Config.ToLower())"
$buildDir = Join-Path $Root "out\build\$preset"
Step "Configuring ($preset)"
# A build directory configured earlier with another compiler (say clang-cl)
# does not survive `cmake --preset` cleanly: CMake notices the compiler change,
# wipes its cache and re-runs, and the preset's -march flags are lost on the
# way, so the SDK's SSSE3 byte-swap intrinsics fail to compile. Spot that
# state (foreign compiler, or empty CMAKE_CXX_FLAGS) and configure from scratch.
$cache = Join-Path $buildDir 'CMakeCache.txt'
if (Test-Path $cache) {
    $cacheCxx = ''
    $cacheFlags = ''
    foreach ($line in Get-Content -LiteralPath $cache) {
        if ($line -match '^CMAKE_CXX_COMPILER:\w+=(.*)$') { $cacheCxx = $Matches[1] }
        if ($line -match '^CMAKE_CXX_FLAGS:\w+=(.*)$') { $cacheFlags = $Matches[1] }
    }
    $wantCxx = $clangxx.Replace('\', '/')
    if (($cacheCxx -and $cacheCxx -ne $wantCxx) -or -not $cacheFlags) {
        Note "$buildDir was configured with '$cacheCxx' (flags '$cacheFlags'); reconfiguring it from scratch."
        Remove-Item -LiteralPath $cache -Force
        Remove-Item -LiteralPath (Join-Path $buildDir 'CMakeFiles') -Recurse -Force -ErrorAction SilentlyContinue
    }
}
Invoke-Checked 'cmake' @('--preset', $preset)

Step 'Building daytona.exe, the SDK runtime and the two GPU plugins'
Invoke-LoggedBuild 'Game build' @('--build', '--preset', $preset) (Join-Path $logDir "$preset.log")

$exe = Join-Path $buildDir 'daytona.exe'
foreach ($name in @('daytona.exe', 'rexruntime.dll', 'rexgpu-plume.dll', 'rexgpu-xenos.dll')) {
    if (-not (Test-Path (Join-Path $buildDir $name))) { Fail "Build finished but $buildDir\$name is missing." }
}

Step 'Done'
Note "Executable: $exe"
Note "Version:    $((Get-Item $exe).VersionInfo.FileVersion)"
Note "Run it with:  `"$exe`" --game_data_root=`"$Root\extracted`""
Note 'Logs go to logs\daytona_NNN.log next to the exe.'

if ($Run) {
    Step 'Launching'
    Start-Process -FilePath $exe -ArgumentList @("--game_data_root=`"$Root\extracted`"") -WorkingDirectory $buildDir
}
