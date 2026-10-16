# Installs the formatters pinned in tools/formatter-versions.env (Windows, PowerShell 5.1+ or 7).
#   powershell -ExecutionPolicy Bypass -File tools\setup-format.ps1
#   powershell -ExecutionPolicy Bypass -File tools\setup-format.ps1 -Check   # report only
#
# Same layout as tools/setup-format.sh, except that Python tools live in .venv-format\Scripts.
# Safe to re-run: a tool that already matches its pin is left alone.
param([switch]$Check)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$pins = @{}
Get-Content 'tools\formatter-versions.env' |
    Where-Object { $_ -match '^[A-Z_]+=' } |
    ForEach-Object {
        $key, $value = $_ -split '=', 2
        $pins[$key] = $value
    }

$Venv = '.venv-format'
$VenvBin = Join-Path $Venv 'Scripts'
$script:Bad = $false

function Step([string]$Text) {
    Write-Host ''
    Write-Host "==> $Text" -ForegroundColor Cyan
}

# Report <name> <pinned> <installed>
function Report([string]$Name, [string]$Pinned, [string]$Found) {
    if (-not $Found) {
        Write-Host ("  [MISS]  {0,-13} want {1}" -f $Name, $Pinned)
        $script:Bad = $true
    } elseif ($Found -eq $Pinned) {
        Write-Host ("  [ OK ]  {0,-13} {1}" -f $Name, $Found)
    } else {
        Write-Host ("  [DIFF]  {0,-13} want {1}, found {2}" -f $Name, $Pinned, $Found)
        $script:Bad = $true
    }
}

# Last word of the first output line, leading "v" removed; empty if the tool is missing.
function Get-Version([string]$Exe) {
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) { return '' }
    $line = (& $Exe --version 2>$null | Select-Object -First 1)
    if (-not $line) { return '' }
    return (($line -split '\s+')[-1]) -replace '^v', ''
}

# Returns the launcher and its leading arguments, e.g. @('py', '-3.12').
function Find-Python {
    $candidates = @(
        @('py', '-3.15'), @('py', '-3.14'), @('py', '-3.13'), @('py', '-3.12'),
        @('py', '-3.11'), @('py', '-3.10'), @('python')
    )
    foreach ($cand in $candidates) {
        if (-not (Get-Command $cand[0] -ErrorAction SilentlyContinue)) { continue }
        $rest = @($cand | Select-Object -Skip 1)
        & $cand[0] @rest -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { return ,$cand } # the comma keeps @('python') an array
    }
    return $null
}

Step '1/4 Python tools in .venv-format (clang-format, gersemi, ruff)'
if (-not $Check) {
    if (-not (Test-Path (Join-Path $VenvBin 'python.exe'))) {
        $py = Find-Python
        if (-not $py) {
            throw 'Python 3.10+ is required (docs/formatting.md). Install it from python.org or with winget.'
        }
        $pyRest = @($py | Select-Object -Skip 1)
        & $py[0] @pyRest -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "failed to create $Venv" }
    }
    & (Join-Path $VenvBin 'python.exe') -m pip install -q `
        "clang-format==$($pins['CLANG_FORMAT'])" "gersemi==$($pins['GERSEMI'])" "ruff==$($pins['RUFF'])"
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}

Step '2/4 prettier (global npm)'
$found = Get-Version 'prettier'
if ($found -ne $pins['PRETTIER'] -and -not $Check) {
    if (-not (Get-Command 'npm.cmd' -ErrorAction SilentlyContinue)) {
        throw 'npm not found; install Node.js LTS first'
    }
    & npm.cmd install --global "prettier@$($pins['PRETTIER'])"
    if ($LASTEXITCODE -ne 0) { throw 'npm install prettier failed' }
    $found = Get-Version 'prettier'
}
Report 'prettier' $pins['PRETTIER'] $found

Step '3/4 shfmt (.venv-format\Scripts)'
$shfmtExe = Join-Path $VenvBin 'shfmt.exe'
$found = if (Test-Path $shfmtExe) { Get-Version $shfmtExe } else { '' }
if ($found -ne $pins['SHFMT'] -and -not $Check) {
    $url = "https://github.com/mvdan/sh/releases/download/v$($pins['SHFMT'])/shfmt_v$($pins['SHFMT'])_windows_amd64.exe"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $shfmtExe
    $found = Get-Version $shfmtExe
}
Report 'shfmt' $pins['SHFMT'] $found

Step '4/4 rustfmt (rustup)'
$rustc = $null
if (Get-Command rustc -ErrorAction SilentlyContinue) {
    $rustc = ((rustc --version) -split '\s+')[1]
}
Report 'rustc' $pins['RUST'] $rustc
if (Get-Command rustfmt -ErrorAction SilentlyContinue) {
    Write-Host ("  [ OK ]  rustfmt       {0}" -f (rustfmt --version))
} elseif (-not $Check -and (Get-Command rustup -ErrorAction SilentlyContinue)) {
    & rustup component add rustfmt
    Write-Host ("  [ OK ]  rustfmt       {0}" -f (rustfmt --version))
} else {
    Write-Host '  [MISS]  rustfmt       run: rustup component add rustfmt'
    $script:Bad = $true
}

if ($script:Bad) {
    Write-Host ''
    Write-Host 'not ready: fix the [MISS] / [DIFF] lines above and re-run' -ForegroundColor Red
    exit 1
}
Step 'all formatters match docs/formatting.md; next: python tools\format_check.py'
