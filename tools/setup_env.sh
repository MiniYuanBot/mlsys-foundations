#!/usr/bin/env bash
# Installs everything lab0..labN need, at the versions pinned in tools/env-versions.env.
#   bash tools/setup_env.sh --lab 0       # C++ toolchain + Python venv (numpy, matplotlib, pytest)
#   bash tools/setup_env.sh --lab 2       # + CUDA toolkit (lab1) + PyTorch/Triton (lab2)
#   bash tools/setup_env.sh               # everything, same as --lab 7
#   bash tools/setup_env.sh --no-models   # skip the Qwen3 downloads (lab4/lab5)
#
# Platforms: WSL2 Ubuntu 24.04 (main) and native Linux with an NVIDIA driver for every lab;
# macOS (Homebrew) for lab0, lab6 and lab7. Native Windows: tools/setup_env.ps1 (lab0, lab7).
# Package installs are automatic on Ubuntu/Debian (apt) and macOS (brew); other distros
# get the list of packages to install themselves.
# Safe to re-run: every step skips what is already there, so run it again with a
# larger --lab when you reach the next lab.
# sudo is used only for apt. Python goes into ~/venvs/mlsys, Rust into ~/.rustup and
# ~/.cargo, models into ~/models. direnv activates the venv whenever you cd into the repo.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
. tools/env-versions.env
. tools/formatter-versions.env # RUST

VENV="${MLSYS_VENV:-$HOME/venvs/mlsys}"
MODELS="${MLSYS_MODELS:-$HOME/models}"
LAB=7
WITH_MODELS=1
usage() {
    echo "usage: bash tools/setup_env.sh [--lab 0-7] [--no-models]" >&2
    exit 2
}
while (($#)); do
    case $1 in
        --lab)
            LAB=${2:-}
            shift 2 || usage
            ;;
        --no-models) WITH_MODELS=0 && shift ;;
        *) usage ;;
    esac
done
[[ $LAB =~ ^[0-7]$ ]] || usage

case $(uname -s) in
    Linux) OS=linux RC="$HOME/.bashrc" SHELL_NAME=bash ;;
    Darwin) OS=mac RC="$HOME/.zshrc" SHELL_NAME=zsh ;;
    MINGW* | MSYS* | CYGWIN*)
        echo "This is Git Bash/MSYS on Windows. For lab0/lab7 use PowerShell:" >&2
        echo "    powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab $LAB" >&2
        echo "For lab1-lab6 open WSL (run: wsl) and run this script there." >&2
        exit 1
        ;;
    *)
        echo "Unsupported system $(uname -s): see setup.md." >&2
        exit 1
        ;;
esac
if [[ $OS == mac && $LAB -ge 1 && $LAB -le 5 ]]; then
    echo "lab$LAB needs an NVIDIA GPU: run this inside WSL2 or on a Linux machine with one." >&2
    exit 1
fi

need() { ((LAB >= $1)); }                  # does lab0..LAB include lab $1?
gpu() { [[ $OS == linux ]] && need "$1"; } # GPU steps are WSL/Linux only
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
add_line() { grep -qxF "$1" "$RC" 2> /dev/null || echo "$1" >> "$RC"; }

step "system packages"
if [[ $OS == linux ]] && ! command -v apt-get > /dev/null; then
    # Not Debian/Ubuntu: we do not guess package names, only check the commands.
    cmds=(c++ make cmake git curl python3 direnv)
    need 6 && cmds+=(mpirun)
    missing=()
    for c in "${cmds[@]}"; do command -v "$c" > /dev/null || missing+=("$c"); done
    if ((${#missing[@]})); then
        echo "Install these with your package manager, then re-run: ${missing[*]}" >&2
        echo "(also the Python venv module and headers, and for lab4+ OpenSSL headers and pkg-config)" >&2
        exit 1
    fi
    echo "all present"
    PYTHON=python3
elif [[ $OS == linux ]]; then
    # python3-dev: Triton compiles a small C launcher against Python.h at runtime.
    # libssl-dev + pkg-config: some Rust crates link OpenSSL.
    pkgs=(build-essential cmake git curl python3-venv python3-dev direnv)
    need 4 && pkgs+=(pkg-config libssl-dev)
    need 6 && pkgs+=(openmpi-bin libopenmpi-dev)
    missing=()
    for p in "${pkgs[@]}"; do dpkg -s "$p" > /dev/null 2>&1 || missing+=("$p"); done
    if ((${#missing[@]})); then
        echo "apt install ${missing[*]} (needs your sudo password)"
        sudo apt-get update
        sudo apt-get install -y "${missing[@]}"
    else
        echo "all present"
    fi
    PYTHON=python3
else
    xcode-select -p > /dev/null 2>&1 || {
        echo "Run 'xcode-select --install' first (C++ compiler), then re-run this script." >&2
        exit 1
    }
    command -v brew > /dev/null || {
        echo "Install Homebrew first (https://brew.sh), then re-run this script." >&2
        exit 1
    }
    pkgs=(cmake python@3.12 direnv)
    need 6 && pkgs+=(open-mpi)
    for p in "${pkgs[@]}"; do brew list --versions "$p" > /dev/null || brew install "$p"; done
    PYTHON="$(brew --prefix)/bin/python3.12"
fi

if gpu 1; then
    step "CUDA toolkit $CUDA (lab1+)"
    if [[ -x /usr/local/cuda-$CUDA/bin/nvcc ]]; then
        echo "already installed: /usr/local/cuda-$CUDA"
    elif grep -qi microsoft /proc/version; then
        # The wsl-ubuntu repository ships the toolkit without a Linux GPU driver;
        # inside WSL the Windows driver is used. Never install a driver here.
        if ! dpkg -s cuda-keyring > /dev/null 2>&1; then
            tmp=$(mktemp -d)
            curl -fsSL -o "$tmp/cuda-keyring.deb" \
                https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
            sudo dpkg -i "$tmp/cuda-keyring.deb"
            rm -rf "$tmp"
            sudo apt-get update
        fi
        sudo apt-get install -y "cuda-toolkit-${CUDA//./-}"
    elif [[ $(. /etc/os-release && echo "$ID $VERSION_ID") == "ubuntu 24.04" && $(uname -m) == x86_64 ]]; then
        # Native Ubuntu: the cuda-toolkit-* package contains no driver (unlike the `cuda`
        # metapackage). The driver is yours to install, e.g. `sudo ubuntu-drivers install`.
        if ! dpkg -s cuda-keyring > /dev/null 2>&1; then
            tmp=$(mktemp -d)
            curl -fsSL -o "$tmp/cuda-keyring.deb" \
                https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
            sudo dpkg -i "$tmp/cuda-keyring.deb"
            rm -rf "$tmp"
            sudo apt-get update
        fi
        sudo apt-get install -y "cuda-toolkit-${CUDA//./-}"
    else
        echo "Install the CUDA $CUDA toolkit (toolkit only, keep your driver) from"
        echo "https://developer.nvidia.com/cuda-12-6-0-download-archive into /usr/local/cuda-$CUDA, then re-run."
    fi
    command -v nvidia-smi > /dev/null || [[ -x /usr/lib/wsl/lib/nvidia-smi ]] \
        || echo "WARNING: nvidia-smi not found. Install an NVIDIA driver that supports CUDA $CUDA (>= 560)."
fi

step "Python venv at $VENV"
if [[ ! -x $VENV/bin/python ]]; then
    mkdir -p "$(dirname "$VENV")"
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
fi
pip_install() { "$VENV/bin/python" -m pip install --disable-pip-version-check "$@"; }
pip_install "numpy==$NUMPY" "matplotlib==$MATPLOTLIB" "pytest==$PYTEST"
if gpu 2; then
    step "PyTorch $TORCH + Triton (lab2+)"
    # The CUDA-specific index brings the matching triton as a dependency.
    pip_install "torch==$TORCH" --index-url "$TORCH_INDEX"
fi
if gpu 4; then
    step "Hugging Face tooling (lab4+)"
    pip_install "transformers==$TRANSFORMERS" "safetensors==$SAFETENSORS" "huggingface_hub==$HF_HUB"
fi

if gpu 4; then
    step "Rust $RUST (lab4+)"
    if [[ ! -x $HOME/.cargo/bin/rustup ]]; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --profile minimal --default-toolchain "$RUST" --component clippy,rustfmt
    elif [[ $("$HOME/.cargo/bin/rustc" --version 2> /dev/null | awk '{ print $2 }') != "$RUST" ]]; then
        "$HOME/.cargo/bin/rustup" toolchain install "$RUST" --profile minimal --component clippy,rustfmt
        "$HOME/.cargo/bin/rustup" default "$RUST"
    else
        echo "already installed: rustc $RUST"
    fi
fi

step "shell setup ($RC)"
add_line "eval \"\$(direnv hook $SHELL_NAME)\""
add_line "alias mlsys='source $VENV/bin/activate'"
if gpu 1; then
    add_line 'export PATH=/usr/local/cuda/bin:$PATH'
    add_line 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
fi
if gpu 4; then
    add_line '. "$HOME/.cargo/env"'
    add_line "export MLSYS_MODELS=$MODELS"
fi
# .envrc puts the venv first on PATH inside this repo, so plain `python` and
# `python3` are the course interpreter. direnv refuses an .envrc until it is allowed.
direnv allow "$ROOT"
mkdir -p results/environment results/lab{0..7}

if gpu 4 && [[ $WITH_MODELS == 1 ]]; then
    step "models -> $MODELS"
    models=(Qwen3-0.6B)            # lab4 main model, lab5 draft model (~1.5 GB)
    need 5 && models+=(Qwen3-1.7B) # lab5 target model (~4 GB)
    mkdir -p "$MODELS"
    for m in "${models[@]}"; do
        if [[ -f $MODELS/$m/config.json && -f $MODELS/$m/tokenizer.json ]]; then
            echo "$m already present"
        else
            # Downloads the whole repo; files already on disk are skipped. If
            # huggingface.co is slow, `export HF_ENDPOINT=https://hf-mirror.com` first.
            "$VENV/bin/hf" download "Qwen/$m" --local-dir "$MODELS/$m"
        fi
    done
fi

step "done. Open a NEW terminal, cd into the repo, then run:  bash tools/check_env.sh --lab $LAB"
