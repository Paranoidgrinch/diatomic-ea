#!/usr/bin/env bash
set -euo pipefail

REMOVE_DATA=0
if [[ "${1:-}" == "--remove-data" ]]; then
    REMOVE_DATA=1
fi

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_ROOT="$DATA_HOME/DiatomicEA"
rm -f "$HOME/.local/bin/diatomic-ea-gui" "$HOME/.local/bin/diatomic-ea"
rm -f "$DATA_HOME/applications/diatomicea.desktop"
rm -rf "$APP_ROOT/app"

if [[ $REMOVE_DATA -eq 1 ]]; then
    rm -rf "$APP_ROOT"
    rm -rf "$HOME/.diatomic-ea"
    echo "DiatomicEA and calculation data were removed."
else
    echo "DiatomicEA was removed. Calculation data were kept."
fi
