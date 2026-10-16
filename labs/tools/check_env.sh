#!/usr/bin/env bash
# Checks the tools a lab needs and prints one line per item.
#   bash tools/check_env.sh --lab 2   # only what lab2 needs
#   bash tools/check_env.sh           # every lab (on macOS: lab0, lab6, lab7)
# Exit code 0 iff every REQUIRED item is OK. [WARN] means the version differs from
# tools/env-versions.env: it usually still works, but mention it in your report.
# Every [FAIL] is followed by the command that fixes it. Paste the output into your lab report.
set -uo pipefail

cd "$(dirname "$0")/.."
. tools/env-versions.env
VENV="${MLSYS_VENV:-$HOME/venvs/mlsys}"
MODELS="${MLSYS_MODELS:-$HOME/models}"
PY="$VENV/bin/python"
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"

LAB=""
case "${1:-}" in
    "") ;;
    --lab) LAB="${2:-}" ;;
    *) LAB=x ;;
esac
if [[ -n $LAB && ! $LAB =~ ^[0-7]$ ]]; then
    echo "usage: bash tools/check_env.sh [--lab 0-7]" >&2
    exit 2
fi
case $(uname -s) in
    MINGW* | MSYS* | CYGWIN*)
        echo "This is Git Bash/MSYS on Windows. For lab0/lab7 use PowerShell:" >&2
        echo "    powershell -ExecutionPolicy Bypass -File tools\check_env.ps1${LAB:+ -Lab $LAB}" >&2
        echo "For lab1-lab6 open WSL (run: wsl) and run this script there." >&2
        exit 1
        ;;
    Darwin)
        if [[ $LAB =~ ^[1-5]$ ]]; then
            echo "lab$LAB needs an NVIDIA GPU: check it inside WSL2 or on a Linux machine with one." >&2
            exit 1
        fi
        ;;
esac
FIX="bash tools/setup_env.sh --lab ${LAB:-7}"
fail=0

ok() { printf '  [ OK ]  %-18s %s\n' "$1" "$2"; }
warn() { printf '  [WARN]  %-18s %s\n' "$1" "$2"; }
bad() { # bad <name> <message> [fix]
    printf '  [FAIL]  %-18s %s\n' "$1" "$2"
    printf '          %-18s fix: %s\n' "" "${3:-$FIX}"
    fail=1
}
opt() { printf '  [ -- ]  %-18s %s\n' "$1" "$2"; }
check() { # check <required:1|0> <name> <command...>
    local req=$1 name=$2
    shift 2
    local out
    if out=$("$@" 2>&1); then
        ok "$name" "$(echo "$out" | tail -1)"
    elif [[ $req == 1 ]]; then
        bad "$name" "$(echo "$out" | tail -1)"
    else opt "$name" "not available (optional): $(echo "$out" | tail -1)"; fi
}
pyver() { # pyver <name> <module> <pinned version>
    local got
    if got=$("$PY" -c "import $2; print($2.__version__)" 2>&1); then
        if [[ $got == "$3" || $got == "$3+"* ]]; then ok "$1" "$got"; else warn "$1" "$got (pinned $3)"; fi
    else
        bad "$1" "$(echo "$got" | tail -1)"
    fi
}

item_compiler() { check 1 "c++ / cmake" bash -c 'echo "$(c++ --version | head -1) / $(cmake --version | head -1)"'; }
item_python() {
    # The usual failure: `python3 script.py` runs the system Python, which has no numpy.
    local got
    got=$(command -v python3 || echo "not found")
    if [[ $got == "$VENV/bin/python3" ]]; then
        ok "python3 on PATH" "$got"
    elif [[ -x $PY ]]; then
        bad "python3 on PATH" "$got is not the course venv" \
            "open a new terminal and cd into the repo (direnv activates it), or run: mlsys"
    else
        bad "python3 on PATH" "venv $VENV does not exist"
    fi
}
item_base() {
    pyver numpy numpy "$NUMPY"
    pyver matplotlib matplotlib "$MATPLOTLIB"
    pyver pytest pytest "$PYTEST"
}
item_gpu() {
    check 1 "GPU (driver)" bash -c "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
}
item_nvcc() {
    local got
    if got=$(nvcc --version 2>&1 | grep release); then
        if [[ $got == *"release $CUDA,"* ]]; then ok nvcc "$got"; else warn nvcc "$got (pinned $CUDA)"; fi
    else
        bad nvcc "nvcc not found"
    fi
}
item_nvrtc() { check 1 "NVRTC" bash -c "test -f /usr/local/cuda/include/nvrtc.h && ls /usr/local/cuda/lib64/libnvrtc.so.*.* | head -1"; }
item_profilers() { check 0 "ncu / nsys" bash -c "ncu --version | tail -1 && nsys --version"; }
item_mpi() { check 1 "MPI" bash -c "mpirun --version | head -1"; }
item_torch() {
    pyver torch torch "$TORCH"
    check 1 "torch + CUDA" "$PY" -c "import torch; assert torch.cuda.is_available(), 'cuda not available'; x=torch.randn(512,512,device='cuda'); (x@x).sum().item(); print(torch.version.cuda, torch.cuda.get_device_name())"
}
item_triton() {
    pyver triton triton "$TRITON"
    # Triton reads kernel source from a file, so the test kernel goes into a temp file.
    local test
    test=$(mktemp --suffix=.py)
    cat > "$test" << 'PY'
import torch, triton, triton.language as tl

@triton.jit
def add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

a = torch.randn(1000, device="cuda"); b = torch.randn_like(a); c = torch.empty_like(a)
add[(4,)](a, b, c, 1000, B=256)
assert torch.allclose(c, a + b)
print("compiled and ran a kernel")
PY
    check 1 "triton kernel" "$PY" "$test"
    rm -f "$test"
}
item_hf() {
    pyver transformers transformers "$TRANSFORMERS"
    pyver safetensors safetensors "$SAFETENSORS"
}
item_cargo() { check 1 "cargo / rustc" bash -c "cargo --version && rustc --version"; }
model() {
    check 1 "$1" "$PY" -c "
import json, pathlib, sys
d = pathlib.Path('$MODELS/$1'); cfg = json.loads((d / 'config.json').read_text())
w = sum(f.stat().st_size for f in d.glob('*.safetensors'))
assert (d / 'tokenizer.json').exists() and w > 0, 'incomplete download'
print(cfg['model_type'], 'layers', cfg['num_hidden_layers'], 'hidden', cfg['hidden_size'], f'{w/2**30:.2f} GiB')"
}
item_qwen06() { model Qwen3-0.6B; }
item_qwen17() { model Qwen3-1.7B; }

case $LAB in
    0) items="compiler python base" ;;
    1) items="compiler python base gpu nvcc profilers" ;;
    2) items="compiler python base gpu nvcc torch triton" ;;
    3) items="compiler gpu nvcc nvrtc" ;;
    4) items="python gpu nvrtc torch hf cargo qwen06" ;;
    5) items="gpu nvrtc cargo qwen06 qwen17" ;;
    6) items="compiler python base mpi" ;;
    7) items="python base" ;;
    *)
        if [[ $(uname -s) == Darwin ]]; then
            items="compiler python base mpi"
        else
            items="compiler python base gpu nvcc nvrtc profilers mpi torch triton hf cargo qwen06 qwen17"
        fi
        ;;
esac

echo "== ${LAB:+lab$LAB }environment ($(uname -s), venv $VENV)"
for i in $items; do "item_$i"; done

echo
if [[ $fail == 0 ]]; then echo "All required items OK."; else echo "Some required items FAILED (fix commands above)."; fi
exit $fail
