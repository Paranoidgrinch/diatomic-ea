# DiatomicEA

**Fast and reproducible electron-affinity calculations for diatomic molecules.**

DiatomicEA is an open-source desktop and command-line application for automated electron-affinity calculations of diatomic molecules. It combines a fixed, reproducible DFT workflow with calculation queues, resumable raw results, live progress telemetry, and CSV export.

## Release status

Version **1.0.0** is the first stable release. The scientific workflow, desktop calculation path, Windows/WSL distribution, and native Linux distribution have passed the release audit.

## Platforms

- **Windows:** PyQt5 desktop application on Windows, with the PySCF compute backend isolated in WSL 2 / Ubuntu 24.04.
- **Linux:** native PyQt5 desktop application and native PySCF compute backend.

Python 3.10 or newer is required.

## Install from a release bundle

### Windows

Open PowerShell in the extracted release folder and run:

```powershell
.\scripts\install_windows.ps1
```

If Ubuntu 24.04 is not yet installed under WSL, the installer explains what is missing. To explicitly request the Windows WSL installation step:

```powershell
.\scripts\install_windows.ps1 -InstallWSL
```

DiatomicEA never restarts Windows automatically.

### Linux

```bash
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh
```

The Linux installation uses PySCF directly; WSL is not involved.

Full installation details are in `docs/INSTALLATION.md`.

## Use

Launch the desktop application with `diatomic-ea-gui`, define the two atoms, choose the initial bond-length range, maximum spin and worker count, add calculations to the queue, and start the queue. The GUI reports the current calculation stage, throughput and ETA and displays the final predicted electron affinity and prediction intervals.

See `docs/USER_GUIDE.md` for the complete workflow.

## Scientific reproducibility

Production runs store raw task results, final CSV output, a run manifest and compute-backend provenance. On Windows the exact DiatomicEA wheel used by the desktop application is also deployed to the managed WSL worker environment.

## Development

```bash
python -m pip install -e ".[dev,gui]"
python -m pytest
```

Build release archives with:

```bash
python scripts/build_release.py
```

## License

DiatomicEA is distributed under **GPL-3.0-only**. See `LICENSE`.
