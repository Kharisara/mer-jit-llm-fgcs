$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$Stage = Join-Path $ProjectRoot "MetroPT_Canonical_Sync"
$Zip = Join-Path $ProjectRoot "MetroPT_Canonical_Sync.zip"

$RequiredFiles = @(
    "paper_outputs\secondary_metropt3\preparation_manifest.json",
    "paper_outputs\secondary_metropt3\tables_figures\secondary_validation_summary.json",
    "paper_outputs\secondary_metropt3\tables_figures\secondary_policy_determinism_summary.csv",
    "paper_outputs\secondary_metropt3\tables_figures\fgcs_table_rq7_fault_detection_combined.csv",
    "paper_outputs\final_validation\final_validation_manifest.json",
    "paper_outputs\final_validation\final_results_inventory.csv",
    "paper_outputs\final_validation\final_claims_numbers.json",
    "paper_outputs\final_validation\file_sha256_manifest.csv"
)

$Missing = @()
foreach ($RelativePath in $RequiredFiles) {
    $FullPath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        $Missing += $RelativePath
    }
}

if ($Missing.Count -gt 0) {
    Write-Host "Missing required files:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

if (Test-Path -LiteralPath $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Path $Stage | Out-Null

foreach ($RelativePath in $RequiredFiles) {
    $Source = Join-Path $ProjectRoot $RelativePath
    $Destination = Join-Path $Stage $RelativePath
    $DestinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

if (Test-Path -LiteralPath $Zip) {
    Remove-Item -LiteralPath $Zip -Force
}
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -CompressionLevel Optimal

Write-Host "Created: $Zip" -ForegroundColor Green
Write-Host "Upload this ZIP together with the current post-rerun repository/release ZIP." -ForegroundColor Green