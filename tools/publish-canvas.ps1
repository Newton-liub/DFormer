[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Source = "doc/canvases",
    [string]$Destination,
    [string]$ProjectRoot,
    [switch]$AllowUnversioned
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$versionPattern = '^(?<Version>(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*))-(?<Name>[a-z0-9]+(?:-[a-z0-9]+)*)\.canvas\.tsx$'

function Get-CanvasFileSha256 {
    param([Parameter(Mandatory)][string]$LiteralPath)

    $stream = [IO.File]::OpenRead($LiteralPath)
    try {
        $sha256 = [Security.Cryptography.SHA256]::Create()
        try {
            return -join ($sha256.ComputeHash($stream) | ForEach-Object { $_.ToString("X2") })
        } finally {
            $sha256.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

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
$canvasFiles = @($canvasFiles | Where-Object { $_.Name -like "*.canvas.tsx" } | Sort-Object Name)

if ($canvasFiles.Count -eq 0) {
    throw "No .canvas.tsx files found under: $sourcePath"
}

$versionedFiles = @()
foreach ($canvasFile in $canvasFiles) {
    $nameMatch = [regex]::Match($canvasFile.Name, $versionPattern)
    if (-not $nameMatch.Success) {
        if (-not $AllowUnversioned) {
            throw "Canvas filename must start with MAJOR.MINOR.PATCH: $($canvasFile.Name). Example: 0.0.2-weekly-progress.canvas.tsx"
        }
        Write-Warning "Publishing legacy unversioned Canvas: $($canvasFile.Name)"
        continue
    }

    $versionedFiles += [PSCustomObject]@{
        Version = $nameMatch.Groups["Version"].Value
        File = $canvasFile
    }
}

$duplicateVersions = @($versionedFiles | Group-Object Version | Where-Object { $_.Count -gt 1 })
if ($duplicateVersions.Count -gt 0) {
    $duplicates = ($duplicateVersions | ForEach-Object { $_.Name }) -join ", "
    throw "Duplicate Canvas versions found: $duplicates"
}

if ($PSCmdlet.ShouldProcess($Destination, "Create Canvas destination")) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}

foreach ($canvasFile in $canvasFiles) {
    $target = Join-Path $Destination $canvasFile.Name

    if (Test-Path -LiteralPath $target) {
        $sourceHash = Get-CanvasFileSha256 -LiteralPath $canvasFile.FullName
        $targetHash = Get-CanvasFileSha256 -LiteralPath $target

        if ($sourceHash -eq $targetHash) {
            Write-Host "Unchanged: $target"
            continue
        }

        throw "Refusing to overwrite an existing Canvas version: $target. Increment the version number instead."
    }

    if ($PSCmdlet.ShouldProcess($target, "Publish versioned Canvas")) {
        Copy-Item -LiteralPath $canvasFile.FullName -Destination $target
        Write-Host "Published: $($canvasFile.FullName) -> $target"
    }
}

Write-Host "Canvas destination: $Destination"
