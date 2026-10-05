#!/usr/bin/env bash
# Checks every tool lab0-lab7 needs and prints one line per item.
#   bash tools/check_env.sh
# Exit code 0 iff every REQUIRED item is OK. Paste the output into your lab report.
set -uo pipefail

VENV="${MLSYS_VENV:-$HOME/venvs/mlsys}"
MODELS="${MLSYS_MODELS:-$HOME/models}"
PY="$VENV/bin/python"
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
fail=0

ok()   { printf '  [ OK ]  %-22s %s\n' "$1" "$2"; }
bad()  { printf '  [FAIL]  %-22s %s\n' "$1" "$2"; fail=1; }
opt()  { printf '  [ -- ]  %-22s %s\n' "$1" "$2"; }
check() {  # check <required:1|0> <name> <command...>
    local req=$1 name=$2; shift 2
    local out
    if out=$("$@" 2>&1); then ok "$name" "$(echo "$out" | tail -1)"
    elif [[ $req == 1 ]]; then bad "$name" "$(echo "$out" | tail -1)"
    else opt "$name" "not available (optional): $(echo "$out" | tail -1)"; fi
}

echo "== system"
check 1 "GPU (driver)"     bash -c "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
check 1 "nvcc"             bash -c "nvcc --version | grep release"
check 1 "NVRTC (lab3)"     bash -c "test -f /usr/local/cuda/include/nvrtc.h && ls /usr/local/cuda/lib64/libnvrtc.so.*.* | head -1"
check 1 "g++ / cmake"      bash -c "echo \$(g++ -dumpfullversion) / \$(cmake --version | head -1)"
check 0 "ncu / nsys"       bash -c "ncu --version | tail -1 && nsys --version"
check 1 "MPI (lab6)"       bash -c "mpirun --version | head -1"

echo "== python venv ($VENV)"
check 1 "python"           "$PY" -c "import sys; print(sys.version.split()[0])"
check 1 "torch + CUDA"     "$PY" -c "import torch; assert torch.cuda.is_available(), 'cuda not available'; x=torch.randn(512,512,device='cuda'); (x@x).sum().item(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
# Triton reads kernel source from a file, so the test kernel goes into a temp file.
TRITON_TEST=$(mktemp --suffix=.py)
trap 'rm -f "$TRITON_TEST"' EXIT
cat > "$TRITON_TEST" <<'PY'
import torch, triton, triton.language as tl

@triton.jit
def add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

a = torch.randn(1000, device="cuda"); b = torch.randn_like(a); c = torch.empty_like(a)
add[(4,)](a, b, c, 1000, B=256)
assert torch.allclose(c, a + b)
print("triton", triton.__version__, "compiled and ran a kernel")
PY
check 1 "triton kernel"    "$PY" "$TRITON_TEST"
check 1 "transformers"     "$PY" -c "import transformers, safetensors, numpy, matplotlib; print('transformers', transformers.__version__)"

echo "== rust"
check 1 "cargo / rustc"    bash -c "cargo --version && rustc --version"

echo "== models ($MODELS)"
for m in Qwen3-0.6B Qwen3-1.7B; do
    check 1 "$m" "$PY" -c "
import json, pathlib, sys
d = pathlib.Path('$MODELS/$m'); cfg = json.loads((d / 'config.json').read_text())
w = sum(f.stat().st_size for f in d.glob('*.safetensors'))
assert (d / 'tokenizer.json').exists() and w > 0, 'incomplete download'
print(cfg['model_type'], 'layers', cfg['num_hidden_layers'], 'hidden', cfg['hidden_size'], f'{w/2**30:.2f} GiB')"
done

echo
if [[ $fail == 0 ]]; then echo "All required items OK."; else echo "Some required items FAILED (see above)."; fi
exit $fail
