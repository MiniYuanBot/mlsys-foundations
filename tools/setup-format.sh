#!/usr/bin/env bash
# Installs the formatters pinned in tools/formatter-versions.env (WSL / Linux / macOS).
#   bash tools/setup-format.sh           # install or fix whatever does not match the pins
#   bash tools/setup-format.sh --check   # only report; change nothing
#
# Where things go:
#   clang-format, gersemi, ruff  -> .venv-format/bin   (pip, pinned)
#   shfmt                        -> .venv-format/bin   (GitHub release binary)
#   prettier                     -> global npm          (npm, pinned)
#   rustfmt                      -> rustup              (component of the Rust toolchain)
# .venv-format is gitignored. VS Code settings point at .venv-format/bin directly,
# so the venv does not need to be activated.
#
# Safe to re-run: a tool that already matches its pin is left alone.
set -euo pipefail

cd "$(dirname "$0")/.."
set -a
. tools/formatter-versions.env
set +a

VENV=".venv-format"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
bad=0

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# report <name> <pinned> <installed>
report() {
    if [[ -z "$3" ]]; then
        printf '  [MISS]  %-13s want %s\n' "$1" "$2"
        bad=1
    elif [[ "$3" == "$2" ]]; then
        printf '  [ OK ]  %-13s %s\n' "$1" "$3"
    else
        printf '  [DIFF]  %-13s want %s, found %s\n' "$1" "$2" "$3"
        bad=1
    fi
}

# version_of <tool>: runs "<tool> --version" and prints the last word of the first line,
# leading "v" removed. Empty if the tool is missing. "|| true" keeps set -e quiet.
version_of() {
    "$1" --version 2> /dev/null | awk 'NR == 1 { print $NF; exit }' | sed 's/^v//' || true
}

find_python() {
    local minor cand
    for minor in 15 14 13 12 11 10; do
        cand="python3.$minor"
        if command -v "$cand" > /dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
            echo "$cand"
            return 0
        fi
    done
    if command -v python3 > /dev/null 2>&1 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        echo python3
        return 0
    fi
    return 1
}

install_shfmt() {
    local os arch
    case "$(uname -s)" in
        Linux) os=linux ;;
        Darwin) os=darwin ;;
        *)
            echo "unsupported OS for automatic shfmt install: $(uname -s)" >&2
            return 1
            ;;
    esac
    case "$(uname -m)" in
        x86_64 | amd64) arch=amd64 ;;
        arm64 | aarch64) arch=arm64 ;;
        *)
            echo "unsupported CPU for automatic shfmt install: $(uname -m)" >&2
            return 1
            ;;
    esac
    local url="https://github.com/mvdan/sh/releases/download/v${SHFMT}/shfmt_v${SHFMT}_${os}_${arch}"
    curl --proto '=https' --tlsv1.2 -fsSL -o "$VENV/bin/shfmt.tmp" "$url"
    chmod +x "$VENV/bin/shfmt.tmp"
    mv "$VENV/bin/shfmt.tmp" "$VENV/bin/shfmt"
}

step "1/4 Python tools in $VENV (clang-format, gersemi, ruff)"
if [[ $CHECK_ONLY == 0 ]]; then
    if [[ ! -x "$VENV/bin/python" ]]; then
        PY=$(find_python) || {
            echo "Python 3.10+ is required (docs/formatting.md). macOS: brew install python@3.12" >&2
            echo "Debian/Ubuntu: sudo apt install python3.12-venv" >&2
            exit 1
        }
        if ! "$PY" -m venv "$VENV"; then
            rm -rf "$VENV"
            echo "failed to create $VENV; on Debian/Ubuntu run: sudo apt install python3-venv" >&2
            exit 1
        fi
    fi
    "$VENV/bin/python" -m pip install -q \
        "clang-format==$CLANG_FORMAT" "gersemi==$GERSEMI" "ruff==$RUFF"
fi
report clang-format "$CLANG_FORMAT" "$(version_of "$VENV/bin/clang-format")"
report gersemi "$GERSEMI" "$(version_of "$VENV/bin/gersemi")"
report ruff "$RUFF" "$(version_of "$VENV/bin/ruff")"

step "2/4 prettier (global npm)"
got=$(version_of prettier)
if [[ "$got" != "$PRETTIER" && $CHECK_ONLY == 0 ]]; then
    command -v npm > /dev/null 2>&1 || {
        echo "npm not found; install Node.js LTS first" >&2
        exit 1
    }
    npm install --global "prettier@$PRETTIER"
    hash -r
    got=$(version_of prettier)
fi
report prettier "$PRETTIER" "$got"

step "3/4 shfmt ($VENV/bin)"
got=$(version_of "$VENV/bin/shfmt")
if [[ "$got" != "$SHFMT" && $CHECK_ONLY == 0 ]]; then
    install_shfmt
    got=$(version_of "$VENV/bin/shfmt")
fi
report shfmt "$SHFMT" "$got"

step "4/4 rustfmt (rustup)"
got=$(rustc --version 2> /dev/null | awk '{ print $2 }' || true)
report rustc "$RUST" "$got"
if command -v rustfmt > /dev/null 2>&1; then
    echo "  [ OK ]  rustfmt       $(rustfmt --version)"
elif [[ $CHECK_ONLY == 0 ]] && command -v rustup > /dev/null 2>&1; then
    rustup component add rustfmt
    echo "  [ OK ]  rustfmt       $(rustfmt --version)"
else
    echo "  [MISS]  rustfmt       run: rustup component add rustfmt"
    bad=1
fi

if [[ $bad == 0 ]]; then
    step "all formatters match docs/formatting.md; next: python3 tools/format_check.py"
else
    printf '\n\033[1;31mnot ready: fix the [MISS] / [DIFF] lines above and re-run\033[0m\n' >&2
    exit 1
fi
