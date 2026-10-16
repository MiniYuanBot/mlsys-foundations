# Installs the native-Windows part of the course environment (lab0 and lab7), at the
# versions pinned in tools/env-versions.env. PowerShell 5.1+ or 7.
#   powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 0
#
# lab1-lab6 need CUDA, Rust or MPI: do them in WSL2 with tools/setup_env.sh (see setup.md).
# Safe to re-run: every step skips what is already there.
#   C++ toolchain -> MSYS2 UCRT64 (gcc, cmake, ninja) via winget, only if g++/cmake/ninja are missing
#   Python        -> python.org 3.12 via winget, only if no Python 3.10+ is found
#   venv          -> %USERPROFILE%\venvs\mlsys (override with $env:MLSYS_VENV)
#   `mlsys`       -> a function in your PowerShell profile that activates the venv
param([int]$Lab = 7)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$Root = (Get-Location).Path

if ($Lab -lt 0 -or $Lab -gt 7) { throw 'usage: tools\setup_env.ps1 [-Lab 0-7]' }
if ($Lab -ge 1 -and $Lab -le 6) {
    Write-Host "lab$Lab needs CUDA, Rust or MPI, which this course only supports in WSL2. In WSL run:"
    Write-Host "    bash tools/setup_env.sh --lab $Lab"
    exit 1
}

$pins = @{}
Get-Content 'tools\env-versions.env' |
    Where-Object { $_ -match '^[A-Z_]+=' } |
    ForEach-Object {
        $key, $value = $_ -split '=', 2
        $pins[$key] = $value
    }

$Venv = if ($env:MLSYS_VENV) { $env:MLSYS_VENV } else { Join-Path $HOME 'venvs\mlsys' }
$VenvPy = Join-Path $Venv 'Scripts\python.exe'
$Msys = 'C:\msys64'

function Step([string]$Text) {
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Have([string]$Exe) { [bool](Get-Command $Exe -ErrorAction SilentlyContinue) }

function Need-Winget([string]$What) {
    if (-not (Have 'winget')) {
        throw "winget not found. Install $What by hand (see setup.md), open a new terminal and re-run."
    }
}

# Returns the launcher and its leading arguments, e.g. @('py', '-3.12'). The
# Microsoft Store stubs `python`/`python3` fail the version check and are skipped.
function Find-Python {
    $candidates = @(
        @('py', '-3.14'), @('py', '-3.13'), @('py', '-3.12'), @('py', '-3.11'), @('py', '-3.10'),
        @('python'), @((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'))
    )
    foreach ($cand in $candidates) {
        if (-not (Have $cand[0])) { continue }
        $rest = @($cand | Select-Object -Skip 1)
        & $cand[0] @rest -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { return ,$cand } # the comma keeps @('python') an array
    }
    return $null
}

Step 'C++ toolchain (g++, cmake, ninja)'
if ((Have 'g++') -and (Have 'cmake') -and (Have 'ninja')) {
    Write-Host "already on PATH: $((Get-Command g++).Source)"
} else {
    if (-not (Test-Path "$Msys\usr\bin\bash.exe")) {
        Need-Winget 'MSYS2 (https://www.msys2.org)'
        winget install --id MSYS2.MSYS2 -e --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw 'winget install MSYS2 failed' }
    }
    & "$Msys\usr\bin\bash.exe" -lc 'pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja'
    if ($LASTEXITCODE -ne 0) { throw 'pacman failed' }
    $bin = "$Msys\ucrt64\bin"
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $bin) {
        [Environment]::SetEnvironmentVariable('Path', "$bin;$userPath", 'User')
        Write-Host "added $bin to your user PATH"
    }
    $env:Path = "$bin;$env:Path"
}

Step "Python venv at $Venv"
if (-not (Test-Path $VenvPy)) {
    $py = Find-Python
    if (-not $py) {
        Need-Winget 'Python 3.12 (https://www.python.org/downloads/windows/)'
        winget install --id Python.Python.3.12 -e --scope user --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw 'winget install Python failed' }
        $py = Find-Python
        if (-not $py) { throw 'Python was installed; open a new terminal and re-run this script.' }
    }
    $pyRest = @($py | Select-Object -Skip 1)
    New-Item -ItemType Directory -Force (Split-Path $Venv) | Out-Null
    & $py[0] @pyRest -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "failed to create $Venv" }
    & $VenvPy -m pip install -q --upgrade pip
}
& $VenvPy -m pip install --disable-pip-version-check `
    "numpy==$($pins['NUMPY'])" "matplotlib==$($pins['MATPLOTLIB'])" "pytest==$($pins['PYTEST'])"
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
# Windows venvs only have python.exe; the copy lets the README's `python3 ...` commands
# work too instead of reaching the Microsoft Store stub.
Copy-Item $VenvPy (Join-Path $Venv 'Scripts\python3.exe') -Force

Step "PowerShell profile ($($PROFILE.CurrentUserAllHosts))"
# Activate.ps1 and the profile are scripts; the default policy (Restricted) blocks both.
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq 'Undefined' -or $policy -eq 'Restricted' -or $policy -eq 'AllSigned') {
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
    Write-Host 'set execution policy for your user to RemoteSigned (needed to activate a venv)'
}
$line = "function mlsys { & '$Venv\Scripts\Activate.ps1' }"
$profilePath = $PROFILE.CurrentUserAllHosts
if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Force $profilePath | Out-Null }
if (-not (Select-String -Path $profilePath -SimpleMatch $line -Quiet)) {
    Add-Content -Path $profilePath -Value $line
    Write-Host 'added the `mlsys` command'
} else {
    Write-Host '`mlsys` command already present'
}

foreach ($dir in @('environment') + (0..7 | ForEach-Object { "lab$_" })) {
    New-Item -ItemType Directory -Force (Join-Path $Root "results\$dir") | Out-Null
}

Step "done. Open a NEW terminal, cd into the repo, run ``mlsys``, then:  powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab $Lab"
