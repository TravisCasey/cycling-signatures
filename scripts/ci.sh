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
    cargo +nightly fmt --check

run_step "Clippy" \
    cargo clippy --all-targets

run_step "Tests" \
    cargo test

RUSTDOCFLAGS="-Dwarnings" run_step "Documentation" \
    cargo doc --no-deps --document-private-items

run_step "Cargo deny" \
    cargo deny check

run_step "Unused dependencies" \
    cargo machete

run_step "Spellcheck" \
    cargo spellcheck check -m 1

echo -e "\n${GREEN}${BOLD}All checks passed.${RESET}"
