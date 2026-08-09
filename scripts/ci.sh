#!/usr/bin/env bash
# CI verification script for cycling_signatures.
# Requires cargo, a nightly toolchain, cargo-deny, cargo-machete,
# cargo-spellcheck and uv. An MPI implementation is optional: without mpicc the
# MPI steps are skipped rather than failed.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

with_gallery=1
for argument in "$@"; do
    case "$argument" in
        --skip-gallery) with_gallery=0 ;;
        *)
            echo "usage: ${0##*/} [--skip-gallery]" >&2
            exit 2
            ;;
    esac
done

step=0
run_step() {
    step=$((step + 1))
    echo -e "\n${BOLD}[$step] $1${RESET}"
    shift
    if "$@"; then
        echo -e "${GREEN}  passed${RESET}"
    else
        echo -e "${RED}  FAILED${RESET}"
        exit 1
    fi
}

skip_step() {
    step=$((step + 1))
    echo -e "\n${BOLD}[$step] $1${RESET}"
    echo -e "${YELLOW}  SKIPPED: $2${RESET}"
}

missing=()
command -v uv >/dev/null 2>&1 || missing+=("uv")
cargo +nightly --version >/dev/null 2>&1 || missing+=("the nightly toolchain (rustup toolchain install nightly)")
for subcommand in deny machete spellcheck; do
    cargo "$subcommand" --version >/dev/null 2>&1 || missing+=("cargo-$subcommand")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${RED}Missing required tools:${RESET}" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

run_step "Format check" \
    cargo +nightly fmt --all --check

for features in "--no-default-features" "--features serde"; do
    run_step "Build ($features)" \
        cargo build $features

    run_step "Tests ($features)" \
        cargo test $features

    run_step "Clippy ($features)" \
        cargo clippy -p cycling_signatures --all-targets $features -- -D warnings

    RUSTDOCFLAGS="-Dwarnings --cfg docsrs" run_step "Documentation ($features)" \
        cargo +nightly doc --no-deps --document-private-items $features
done

run_step "Clippy (Python bindings)" \
    cargo clippy -p cycling-signatures-py --all-targets -- -D warnings

RUSTDOCFLAGS="-Dwarnings --cfg docsrs" run_step "Documentation (--all-features)" \
    cargo +nightly doc --no-deps --all-features

run_step "Build (rayon backend)" \
    cargo build --features rayon

run_step "Tests (rayon backend)" \
    cargo test --features rayon

# An MPI build needs a system MPI implementation.
if command -v mpicc >/dev/null 2>&1; then
    run_step "Build (mpi backend)" \
        cargo build --features mpi

    run_step "Tests (mpi backend)" \
        cargo test --features mpi
else
    skip_step "MPI backend checks" \
        "no mpicc on PATH; install an MPI implementation to run them"
fi

run_step "Cargo deny" \
    cargo deny check

run_step "Unused dependencies" \
    cargo machete

run_step "Spellcheck" \
    cargo spellcheck check -m 1

# Python binding checks, run from the python crate. The subshell keeps the
# directory change from reaching anything after it; every path below is relative
# to python/.
(
cd python

run_step "Python build" \
    uv run maturin develop --release

run_step "Python format check" \
    uv run ruff format --check

run_step "Python lint" \
    uv run ruff check

run_step "Python type check" \
    uv run --group examples ty check

run_step "Python spellcheck" \
    uv run codespell cycling_signatures tests

run_step "Python tests" \
    uv run pytest

# sphinx-gallery keys its execution cache on each example's own source, so a
# change to the data, the shared helper, or the extension leaves stale figures
# behind a passing step. Stamp those inputs and clear the cache when they move.
gallery_stamp_path="docs/auto_examples/.gallery-data-stamp"

gallery_input_stamp() {
    for data_file in examples/lorenz/data/lorenz_storage.cyc \
                     examples/lorenz/data/lorenz_trajectory.cyc \
                     examples/lorenz/data/lorenz_raw.npy \
                     examples/dadras/data/dadras_storage.cyc \
                     examples/dadras/data/dadras_trajectory.cyc \
                     examples/dadras/data/dadras_raw.npy \
                     examples/dadras/data/dadras_times.npy; do
        if [ -f "$data_file" ]; then
            stat -c '%n %s %Y' "$data_file"
        else
            echo "$data_file absent"
        fi
    done
    # The extension is rewritten on every build, so its mtime says nothing; the
    # helper is small. Both contribute a digest instead.
    sha256sum examples/_support.py cycling_signatures/_core*.so
}

if [ "$with_gallery" -eq 0 ]; then
    skip_step "Gallery build" \
        "--skip-gallery passed"
else
    if ! current_stamp=$(gallery_input_stamp); then
        echo -e "${RED}  could not stamp the gallery inputs${RESET}" >&2
        exit 1
    fi
    if [ ! -f "$gallery_stamp_path" ] || [ "$current_stamp" != "$(cat "$gallery_stamp_path")" ]; then
        echo -e "\n${YELLOW}Gallery inputs changed; clearing sphinx-gallery cache${RESET}"
        if [ -d docs/auto_examples ]; then
            find docs/auto_examples -name '*.py.md5' -delete
        fi
        rm -f "$gallery_stamp_path"
    fi

    run_step "Gallery build" \
        uv run --group docs --group examples sphinx-build -E -W -b html docs docs/_build/html

    mkdir -p "$(dirname "$gallery_stamp_path")"
    printf '%s\n' "$current_stamp" > "$gallery_stamp_path"
fi
)

echo -e "\n${GREEN}${BOLD}All checks passed.${RESET}"
