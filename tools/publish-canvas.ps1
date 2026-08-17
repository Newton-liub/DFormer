[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Source = "doc",
    [string]$Destination,
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd([char[]]"/\")

if (-not $Destination) {
    $projectKey = $ProjectRoot -replace "^([A-Za-z]):[\\/]+", '$1-'
    $projectKey = $projectKey -replace "[\\/]+", "-"
    $projectKey = $projectKey.Substring(0, 1).ToLowerInvariant() + $projectKey.Substring(1)
    $Destination = Join-Path $env:USERPROFILE ".cursor\projects\$projectKey\canvases"
}
$Destination = [IO.Path]::GetFullPath($Destination)

$sourcePath = if ([IO.Path]::IsPathRooted($Source)) {
    $Source
} else {
    Join-Path $ProjectRoot $Source
}

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Canvas source does not exist: $sourcePath"
}

$sourceItem = Get-Item -LiteralPath $sourcePath
$canvasFiles = if ($sourceItem.PSIsContainer) {
    @(Get-ChildItem -LiteralPath $sourceItem.FullName -Filter "*.canvas.tsx" -File)
} else {
    @($sourceItem)
}

$canvasFiles = @($canvasFiles | Where-Object { $_.Name -like "*.canvas.tsx" })
if ($canvasFiles.Count -eq 0) {
    throw "No .canvas.tsx files found under: $sourcePath"
}

if ($PSCmdlet.ShouldProcess($Destination, "Create Canvas destination")) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}

foreach ($canvasFile in $canvasFiles) {
    $target = Join-Path $Destination $canvasFile.Name
    if ($PSCmdlet.ShouldProcess($target, "Publish $($canvasFile.Name)")) {
        Copy-Item -LiteralPath $canvasFile.FullName -Destination $target -Force
        Write-Host "Published: $($canvasFile.FullName) -> $target"
    }
}

Write-Host "Canvas destination: $Destination"