# DiatomicEA installation

## Release bundle

A release bundle contains the exact Python wheel together with the Windows and Linux installers, this documentation, and the project license. The wheel is the authoritative application package used by both host and compute environments.

## Windows

Requirements:

- 64-bit Windows with WSL 2 support
- Python 3.10 or newer available through the Windows Python launcher (`py`)
- Ubuntu 24.04 under WSL for scientific calculations

From an extracted release bundle:

```powershell
.\scripts\install_windows.ps1
```

The installer creates a private application environment under `%LOCALAPPDATA%\DiatomicEA\app`, installs the desktop GUI, prepares the managed WSL compute environment, deploys the exact same DiatomicEA wheel into that environment, runs a real backend validation calculation, and creates a Start-menu shortcut.

If Ubuntu 24.04 is missing, rerun with `-InstallWSL` to explicitly request the elevated WSL installation command. The application does not restart the computer automatically. If Windows reports that a restart is required, restart manually and run the installer again.

Use `-SkipCompute` only if you intentionally want to install the desktop application without preparing the scientific backend. Use `-DesktopShortcut` to add a desktop shortcut.

Uninstall the application while keeping calculation results:

```powershell
.\scripts\uninstall_windows.ps1
```

To also remove calculation data:

```powershell
.\scripts\uninstall_windows.ps1 -RemoveData
```

## Linux

Requirements:

- Python 3.10 or newer
- Python virtual-environment support (`python3-venv` on Debian/Ubuntu families)
- a graphical desktop session for the GUI

Install from the extracted bundle:

```bash
chmod +x scripts/install_linux.sh scripts/uninstall_linux.sh
./scripts/install_linux.sh
```

The installer creates a private environment under the user's data directory, installs PyQt5, PySCF 2.13.0 and basis-set-exchange, runs a real native PySCF validation calculation, and creates GUI and CLI launchers under `~/.local/bin` plus a desktop entry.

Uninstall while keeping results:

```bash
./scripts/uninstall_linux.sh
```

Remove application and stored calculation data:

```bash
./scripts/uninstall_linux.sh --remove-data
```

## Verification

The release directory contains `SHA256SUMS.txt` and `release_manifest.json`. These identify the wheel, source archive and platform-neutral release bundle by SHA-256 digest.
