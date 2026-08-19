param(
    [switch]$RemoveData
)

$ErrorActionPreference = "Stop"
$installRoot = Join-Path $env:LOCALAPPDATA "DiatomicEA"
$startMenuShortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\DiatomicEA.lnk"
$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "DiatomicEA.lnk"

Remove-Item $startMenuShortcut -Force -ErrorAction SilentlyContinue
Remove-Item $desktopShortcut -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $installRoot "DiatomicEA.cmd") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $installRoot "installation.json") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $installRoot "app") -Recurse -Force -ErrorAction SilentlyContinue

if ($RemoveData) {
    Remove-Item $installRoot -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "DiatomicEA and calculation data were removed."
}
else {
    Write-Host "DiatomicEA was removed. Calculation data were kept in:"
    Write-Host $installRoot
}
