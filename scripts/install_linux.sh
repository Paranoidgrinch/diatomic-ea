#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shopt -s nullglob
WHEELS=("$BUNDLE_ROOT"/diatomic_ea-*.whl)
if [[ ${#WHEELS[@]} -ne 1 ]]; then
    echo "ERROR: expected exactly one DiatomicEA wheel next to install_linux.sh." >&2
    exit 2
fi

PYTHON="${PYTHON:-python3}"
"$PYTHON" - <<'PY_CHECK'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("DiatomicEA requires Python 3.10 or newer.")
PY_CHECK

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_ROOT="$DATA_HOME/DiatomicEA"
VENV="$APP_ROOT/app"
BIN_HOME="$HOME/.local/bin"
APPLICATIONS_HOME="$DATA_HOME/applications"

mkdir -p "$APP_ROOT" "$BIN_HOME" "$APPLICATIONS_HOME"

if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV" || {
        echo "Could not create a virtual environment. Install your distribution's python3-venv package and retry." >&2
        exit 3
    }
fi

APP_PYTHON="$VENV/bin/python"
"$APP_PYTHON" -m pip install --upgrade pip
"$APP_PYTHON" -m pip install --upgrade "${WHEELS[0]}" 'PyQt5>=5.15.11,<6' 'pyscf==2.13.0' 'basis-set-exchange'

"$APP_PYTHON" -m diatomic_ea.compute_smoke

cat > "$BIN_HOME/diatomic-ea-gui" <<EOF
#!/usr/bin/env bash
exec "$APP_PYTHON" -m diatomic_ea.desktop_gui "\$@"
EOF

cat > "$BIN_HOME/diatomic-ea" <<EOF
#!/usr/bin/env bash
exec "$APP_PYTHON" -m diatomic_ea.cli "\$@"
EOF

chmod +x "$BIN_HOME/diatomic-ea-gui" "$BIN_HOME/diatomic-ea"

cat > "$APPLICATIONS_HOME/diatomicea.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=DiatomicEA
Comment=Electron-affinity calculations for diatomic molecules
Exec=$BIN_HOME/diatomic-ea-gui
Terminal=false
Categories=Science;Education;
EOF

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPLICATIONS_HOME" >/dev/null 2>&1 || true
fi

printf '\nDiatomicEA installation complete.\nGUI: %s\nCLI: %s\n' \
    "$BIN_HOME/diatomic-ea-gui" \
    "$BIN_HOME/diatomic-ea"
