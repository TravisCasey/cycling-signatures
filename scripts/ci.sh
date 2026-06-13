#!/usr/bin/env bash
# CI verification script for cycling_signatures.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

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

run_step "Format check" \
    cargo +nightly fmt --all --check

run_step "Build (no features)" \
    cargo build --no-default-features

run_step "Build (serde)" \
    cargo build --features serde

run_step "Build (rayon)" \
    cargo build --features rayon

run_step "Build (mpi)" \
    cargo build --features mpi

run_step "Tests (no features)" \
    cargo test --no-default-features

run_step "Tests (serde)" \
    cargo test --features serde

run_step "Clippy (workspace)" \
    cargo clippy --workspace --all-targets -- -D warnings

RUSTDOCFLAGS="-Dwarnings" run_step "Documentation" \
    cargo doc --no-deps --document-private-items --features serde

run_step "Cargo deny" \
    cargo deny check

run_step "Unused dependencies" \
    cargo machete

run_step "Spellcheck" \
    cargo spellcheck check -m 1

# Python binding checks, run from the python crate.
cd python

run_step "Python build" \
    uv run maturin develop

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

run_step "Gallery build" \
    uv run --group docs --group examples sphinx-build -E -W -b html docs docs/_build/html

echo -e "\n${GREEN}${BOLD}All checks passed.${RESET}"
