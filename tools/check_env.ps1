# Checks the native-Windows course environment (lab0, lab7) and prints one line per item.
#   powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 0   # only what lab0 needs
#   powershell -ExecutionPolicy Bypass -File tools\check_env.ps1          # lab0 and lab7
# Exit code 0 iff every REQUIRED item is OK. [WARN] means the version differs from
# tools/env-versions.env. Every [FAIL] is followed by the command that fixes it.
# For lab1-lab6 use tools/check_env.sh inside WSL2.
param([int]$Lab = -1)

Set-Location (Join-Path $PSScriptRoot '..')
if ($Lab -lt -1 -or $Lab -gt 7) {
    Write-Host 'usage: tools\check_env.ps1 [-Lab 0-7]'
    exit 2
}
if ($Lab -ge 1 -and $Lab -le 6) {
    Write-Host "lab$Lab runs in WSL2 on Windows. In WSL run:  bash tools/check_env.sh --lab $Lab"
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
$Fix = "powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab $(if ($Lab -ge 0) { $Lab } else { 7 })"
$script:Fail = $false

function Ok([string]$Name, [string]$Msg) { Write-Host ('  [ OK ]  {0,-18} {1}' -f $Name, $Msg) }
function Warn([string]$Name, [string]$Msg) { Write-Host ('  [WARN]  {0,-18} {1}' -f $Name, $Msg) }
function Bad([string]$Name, [string]$Msg, [string]$How = $Fix) {
    Write-Host ('  [FAIL]  {0,-18} {1}' -f $Name, $Msg)
    Write-Host ('          {0,-18} fix: {1}' -f '', $How)
    $script:Fail = $true
}

# First output line of `<exe> --version`, or $null if the tool is missing.
function First-Line([string]$Exe) {
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) { return $null }
    return (& $Exe --version 2>$null | Select-Object -First 1)
}

function Check-Tool([string]$Name, [string]$Exe) {
    $line = First-Line $Exe
    if ($line) { Ok $Name $line } else { Bad $Name "$Exe not found on PATH" }
}

function Check-Python {
    # The usual failure: `python` resolves to another install, or to the Microsoft Store stub.
    foreach ($exe in 'python', 'python3') {
        $cmd = Get-Command $exe -ErrorAction SilentlyContinue
        $got = if ($cmd) { $cmd.Source } else { 'not found' }
        if ($got -like "$Venv\Scripts\*") {
            Ok "$exe on PATH" $got
        } elseif (Test-Path $VenvPy) {
            Bad "$exe on PATH" "$got is not the course venv" 'run: mlsys   (activates the venv in this terminal)'
        } else {
            Bad "$exe on PATH" "venv $Venv does not exist"
        }
    }
}

function Check-Package([string]$Name, [string]$Pinned) {
    if (-not (Test-Path $VenvPy)) { Bad $Name 'venv missing'; return }
    $got = & $VenvPy -c "import $Name; print($Name.__version__)" 2>&1 | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) { Bad $Name "$got" }
    elseif ("$got" -eq $Pinned) { Ok $Name "$got" }
    else { Warn $Name "$got (pinned $Pinned)" }
}

Write-Host "== $(if ($Lab -ge 0) { "lab$Lab " })environment (Windows, venv $Venv)"
if ($Lab -le 0) {
    Check-Tool 'g++' 'g++'
    Check-Tool 'cmake' 'cmake'
    Check-Tool 'ninja' 'ninja'
}
Check-Python
Check-Package 'numpy' $pins['NUMPY']
Check-Package 'matplotlib' $pins['MATPLOTLIB']
Check-Package 'pytest' $pins['PYTEST']

Write-Host ''
if ($script:Fail) {
    Write-Host 'Some required items FAILED (fix commands above).'
    exit 1
}
Write-Host 'All required items OK.'
exit 0
