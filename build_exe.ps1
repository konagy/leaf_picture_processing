param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name leaf-picture-processing `
    picture_processing_oulema.py

Write-Host ""
Write-Host "Build completed."
Write-Host "Executable: dist\\leaf-picture-processing.exe"
