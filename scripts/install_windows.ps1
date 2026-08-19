param(
    [switch]$InstallWSL,
    [switch]$SkipCompute,
    [switch]$DesktopShortcut
)

$ErrorActionPreference = "Stop"
$distribution = "Ubuntu-24.04"
$bundleRoot = (Split-Path -Parent $PSScriptRoot)

Write-Host ""
Write-Host "DiatomicEA Windows installer"
Write-Host "============================"
Write-Host ""

$wheels = @(Get-ChildItem -Path $bundleRoot -Filter "diatomic_ea-*.whl" -File)
if ($wheels.Count -ne 1) {
    throw "Expected exactly one DiatomicEA wheel next to install_windows.ps1."
}
$wheel = $wheels[0].FullName

$pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
if ($null -eq $pyCommand) {
    throw "Python launcher py.exe was not found. Install Python 3.10 or newer first."
}
$py = $pyCommand.Source

& $py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "DiatomicEA requires Python 3.10 or newer."
}

$installRoot = Join-Path $env:LOCALAPPDATA "DiatomicEA"
$venv = Join-Path $installRoot "app"
$python = Join-Path $venv "Scripts\python.exe"
$pythonw = Join-Path $venv "Scripts\pythonw.exe"

New-Item -ItemType Directory -Force -Path $installRoot | Out-Null

if (-not (Test-Path $python)) {
    & $py -3 -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the DiatomicEA application environment."
    }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not update pip in the DiatomicEA environment."
}

& $python -m pip install --upgrade $wheel 'PyQt5>=5.15.11,<6'
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the DiatomicEA desktop application."
}

$version = (& $python -c "import diatomic_ea; print(diatomic_ea.__version__)").Trim()
Write-Host "Installed DiatomicEA $version"

if (-not $SkipCompute) {
    $wslCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if ($null -eq $wslCommand) {
        throw "WSL is not available on this Windows installation. Install WSL 2 and rerun this installer."
    }
    $wsl = $wslCommand.Source

    $installedDistributions = @(
        & $wsl --list --quiet 2>$null |
        ForEach-Object { ($_ -replace "`0", "").Trim() } |
        Where-Object { $_ }
    )

    if ($installedDistributions -notcontains $distribution) {
        if ($InstallWSL) {
            Write-Host ""
            Write-Host "Installing $distribution with elevated privileges..."
            $process = Start-Process `
                -FilePath $wsl `
                -ArgumentList @("--install", "--distribution", $distribution, "--no-launch") `
                -Verb RunAs `
                -Wait `
                -PassThru

            if ($process.ExitCode -ne 0) {
                throw "WSL installation did not complete successfully."
            }

            Write-Host ""
            Write-Host "$distribution was installed."
            Write-Host "Windows may require a restart before WSL 2 can start."
            Write-Host "DiatomicEA will NOT restart the computer automatically."
            Write-Host "After any required restart, run this installer again."
            exit 2
        }

        Write-Host ""
        Write-Host "$distribution is not installed."
        Write-Host "Re-run this installer with -InstallWSL to request an elevated WSL installation."
        Write-Host "No restart will be performed automatically."
        exit 2
    }

    Write-Host ""
    Write-Host "Preparing WSL compute environment..."
    & $python -m diatomic_ea.compute_bootstrap --distribution $distribution
    if ($LASTEXITCODE -ne 0) {
        throw "The WSL compute environment could not be prepared. Check the diagnostic message above."
    }

    Write-Host ""
    Write-Host "Deploying the exact application wheel to the WSL worker..."
    & $python -m diatomic_ea.compute_deploy $wheel --distribution $distribution
    if ($LASTEXITCODE -ne 0) {
        throw "The DiatomicEA compute worker could not be deployed."
    }

    Write-Host ""
    Write-Host "Running compute-backend validation..."
    & $python -m diatomic_ea.compute_smoke
    if ($LASTEXITCODE -ne 0) {
        throw "The compute-backend validation failed."
    }
}

$launcher = Join-Path $installRoot "DiatomicEA.cmd"
$launcherText = "@echo off`r`nstart `"`" `"$pythonw`" -m diatomic_ea.desktop_gui`r`n"
[System.IO.File]::WriteAllText(
    $launcher,
    $launcherText,
    [System.Text.Encoding]::ASCII
)

$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$startMenuShortcut = Join-Path $startMenu "DiatomicEA.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($startMenuShortcut)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "-m diatomic_ea.desktop_gui"
$shortcut.WorkingDirectory = $installRoot
$shortcut.Description = "DiatomicEA"
$shortcut.Save()

if ($DesktopShortcut) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $desktopShortcut = Join-Path $desktop "DiatomicEA.lnk"
    $shortcut = $shell.CreateShortcut($desktopShortcut)
    $shortcut.TargetPath = $pythonw
    $shortcut.Arguments = "-m diatomic_ea.desktop_gui"
    $shortcut.WorkingDirectory = $installRoot
    $shortcut.Description = "DiatomicEA"
    $shortcut.Save()
}

$manifest = @{
    version = $version
    installed_at_utc = [DateTime]::UtcNow.ToString("o")
    application_python = $python
    wheel = [System.IO.Path]::GetFileName($wheel)
    compute_configured = (-not $SkipCompute)
}
$manifest | ConvertTo-Json | Set-Content -Path (Join-Path $installRoot "installation.json") -Encoding UTF8

Write-Host ""
Write-Host "DiatomicEA installation complete."
Write-Host "Start menu: DiatomicEA"
Write-Host "Launcher: $launcher"
