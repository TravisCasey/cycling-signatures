#!/usr/bin/env bash
# CI verification script for cycling_signatures.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

skip_gallery=0
for argument in "$@"; do
    case "$argument" in
        --skip-gallery) skip_gallery=1 ;;
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

run_step "Format check" \
    cargo +nightly fmt --all --check

# The rayon and mpi features gate no source in this crate; they only select a
# chomp3rs execution backend. Compile-time checking therefore has just two
# configurations to cover: without serde and with it. There is no default
# feature, so the first is what a plain `cargo build` compiles.
for features in "--no-default-features" "--features serde"; do
    run_step "Build ($features)" \
        cargo build $features

    run_step "Tests ($features)" \
        cargo test $features

    run_step "Clippy ($features)" \
        cargo clippy --workspace --all-targets $features -- -D warnings

    RUSTDOCFLAGS="-Dwarnings --cfg docsrs" run_step "Documentation ($features)" \
        cargo doc --no-deps --document-private-items $features
done

# The backend features build and run the same source against a different
# chomp3rs backend, so they are worth linking and exercising once each.
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

# Python binding checks, run from the python crate.
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
                     examples/lorenz/data/lorenz_raw.npy \
                     examples/dadras/data/dadras_storage.cyc \
                     examples/dadras/data/dadras_raw.npy; do
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

if [ "$skip_gallery" -eq 1 ]; then
    skip_step "Gallery build" "--skip-gallery requested"
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

echo -e "\n${GREEN}${BOLD}All checks passed.${RESET}"
