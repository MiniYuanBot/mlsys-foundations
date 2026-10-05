#!/usr/bin/env bash
# One-time environment setup for lab2-lab7 inside WSL2 Ubuntu 24.04.
# Assumes the CUDA 12.6 toolkit from setup.md is already installed (lab1).
#
#   bash tools/setup_wsl.sh            # everything below
#   bash tools/setup_wsl.sh --no-models
#
# Safe to re-run: every step skips what is already there.
# Uses sudo only for apt packages (step 1). Nothing else touches the system:
# Python goes into ~/venvs/mlsys, Rust into ~/.rustup and ~/.cargo, models into ~/models.
set -euo pipefail

VENV="$HOME/venvs/mlsys"
MODELS="$HOME/models"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"  # matches driver 566.x (CUDA <= 12.7)
WITH_MODELS=1
[[ "${1:-}" == "--no-models" ]] && WITH_MODELS=0

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

step "1/5 apt packages (needs your sudo password)"
# python3-dev: Triton compiles a small C launcher against Python.h at runtime.
# libssl-dev + pkg-config: some Rust crates link OpenSSL.
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-pip build-essential cmake \
    pkg-config libssl-dev curl git openmpi-bin libopenmpi-dev

step "2/5 Python venv at $VENV: PyTorch (CUDA 12.6) + Triton + model tooling"
if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
# torch from the cu126 index brings a matching triton as a dependency.
"$VENV/bin/python" -m pip install torch --index-url "$TORCH_INDEX"
"$VENV/bin/python" -m pip install numpy matplotlib pytest transformers safetensors "huggingface_hub[cli]"

step "3/5 Rust toolchain (rustup, stable)"
if ! command -v "$HOME/.cargo/bin/cargo" >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --component clippy,rustfmt
fi
"$HOME/.cargo/bin/rustup" update stable

step "4/5 shell setup (~/.bashrc)"
add_line() { grep -qxF "$1" "$HOME/.bashrc" || echo "$1" >> "$HOME/.bashrc"; }
add_line 'export PATH=/usr/local/cuda/bin:$PATH'
add_line 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
add_line '. "$HOME/.cargo/env"'
add_line "export MLSYS_MODELS=$MODELS"
add_line "alias mlsys='source $VENV/bin/activate'"

if [[ $WITH_MODELS == 1 ]]; then
    step "5/5 models -> $MODELS (Qwen3-0.6B ~1.5 GB, Qwen3-1.7B ~4 GB)"
    mkdir -p "$MODELS"
    for m in Qwen3-0.6B Qwen3-1.7B; do
        if [[ -f "$MODELS/$m/config.json" ]]; then
            echo "$m already present"
        else
            # If huggingface.co is slow from your network, run
            #   export HF_ENDPOINT=https://hf-mirror.com
            # before this script.
            "$VENV/bin/hf" download "Qwen/$m" --local-dir "$MODELS/$m" \
                --include "*.json" "*.safetensors" "*.txt"
        fi
    done
else
    step "5/5 models skipped (--no-models)"
fi

step "done. Open a NEW terminal (or: source ~/.bashrc), then run:  bash tools/check_env.sh"
