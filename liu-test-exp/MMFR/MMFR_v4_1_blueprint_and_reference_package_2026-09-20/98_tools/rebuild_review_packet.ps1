[CmdletBinding()]
param(
    [string]$Profile = 'current',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoRoot = (Resolve-Path (Join-Path $packageRoot '..\..\..')).Path
$profilePath = Join-Path $packageRoot '00_control\review_profile.json'
$reproPath = Join-Path $packageRoot '02_evidence\reproducibility_current.json'
$validationPath = Join-Path $packageRoot '02_evidence\validation_report.json'
$registryPath = Join-Path $packageRoot '03_reference\MMFR_reference_registry_v4_1_2026-09-20.json'
$sourceMaterialsRoot = Join-Path $packageRoot '03_reference\source_materials'
$packetRoot = Join-Path $packageRoot '99_review_packet_current'
$expectedPacketFiles = @(
    'REVIEW_BRIEF.md',
    'CURRENT_PROTOCOL.md',
    'CURRENT_REPORT.md',
    'IMPLEMENTATION_DIFF.md',
    'EVIDENCE_SUMMARY.md',
    'packet_manifest.json'
)

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StringSha256([string]$Text) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Convert-ToPackagePath([string]$Path) {
    return $Path.Replace('\', '/')
}

function Get-PackageFile([string]$RelativePath) {
    $normalized = $RelativePath.Replace('/', '\')
    $absolute = [System.IO.Path]::GetFullPath((Join-Path $packageRoot $normalized))
    $rootPrefix = $packageRoot.TrimEnd('\') + '\'
    if (-not $absolute.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Profile source escapes package root: $RelativePath"
    }
    if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
        throw "Profile source does not exist: $RelativePath"
    }
    return $absolute
}

function Get-SourceRecord([string]$RelativePath) {
    $absolute = Get-PackageFile $RelativePath
    return [ordered]@{
        path = (Convert-ToPackagePath $RelativePath)
        sha256 = Get-Sha256 $absolute
        bytes = (Get-Item -LiteralPath $absolute).Length
    }
}

function Get-GitIdentity {
    $commitOutput = @(& git -C $repoRoot rev-parse HEAD 2>$null)
    $commitExit = $LASTEXITCODE
    if ($commitExit -ne 0 -or $commitOutput.Count -eq 0 -or [string]::IsNullOrWhiteSpace([string]$commitOutput[0])) {
        throw 'Unable to resolve the repository Git commit.'
    }
    $branchOutput = @(& git -C $repoRoot rev-parse --abbrev-ref HEAD 2>$null)
    $branchExit = $LASTEXITCODE
    if ($branchExit -ne 0 -or $branchOutput.Count -eq 0 -or [string]::IsNullOrWhiteSpace([string]$branchOutput[0])) {
        throw 'Unable to resolve the repository Git branch.'
    }
    $trackedChanges = @(& git -C $repoRoot status --porcelain --untracked-files=no 2>$null)
    $statusExit = $LASTEXITCODE
    if ($statusExit -ne 0) {
        throw 'Unable to resolve the repository tracked-workspace state.'
    }
    return [ordered]@{
        branch = ([string]$branchOutput[0]).Trim()
        commit = ([string]$commitOutput[0]).Trim()
        workspace_dirty = ($trackedChanges.Count -gt 0)
    }
}

function Convert-AuthorityPath([string]$Path) {
    return (($Path -replace '^(\.\./)+', '') -replace '\\', '/')
}

function Add-BulletSection([System.Collections.Generic.List[string]]$Lines, [string]$Heading, $Values) {
    $Lines.Add("## $Heading")
    $Lines.Add('')
    foreach ($value in @($Values)) {
        $Lines.Add("- $value")
    }
    $Lines.Add('')
}

function Test-MarkdownLinks($Files) {
    $broken = New-Object System.Collections.Generic.List[string]
    foreach ($file in $Files) {
        $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        $matches = [regex]::Matches($text, '!?(?<!\\)\[[^\]]*\]\((?<target>[^)]+)\)')
        foreach ($match in $matches) {
            $target = $match.Groups['target'].Value.Trim()
            if ($target.StartsWith('<') -and $target.EndsWith('>')) {
                $target = $target.Substring(1, $target.Length - 2)
            }
            if ($target -match '^(?i:https?|mailto|ftp):' -or $target.StartsWith('#')) {
                continue
            }
            $targetWithoutFragment = ($target -split '#', 2)[0]
            if ([string]::IsNullOrWhiteSpace($targetWithoutFragment)) {
                continue
            }
            $decoded = [System.Uri]::UnescapeDataString($targetWithoutFragment).Replace('/', '\')
            $resolved = [System.IO.Path]::GetFullPath((Join-Path $file.DirectoryName $decoded))
            if (-not (Test-Path -LiteralPath $resolved)) {
                $rootPrefix = $packageRoot.TrimEnd('\') + '\'
                if ($file.FullName.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $relativeFile = $file.FullName.Substring($packageRoot.Length + 1).Replace('\', '/')
                }
                else {
                    $relativeFile = $file.Name
                }
                $broken.Add("$relativeFile -> $target")
            }
        }
    }
    return $broken.ToArray()
}

if (-not (Test-Path -LiteralPath $profilePath -PathType Leaf)) {
    throw 'Missing 00_control/review_profile.json.'
}
if (-not (Test-Path -LiteralPath $reproPath -PathType Leaf)) {
    throw 'Missing 02_evidence/reproducibility_current.json.'
}
if (-not (Test-Path -LiteralPath $validationPath -PathType Leaf)) {
    throw 'Missing 02_evidence/validation_report.json.'
}
if (-not (Test-Path -LiteralPath $registryPath -PathType Leaf)) {
    throw 'Missing 03_reference/MMFR_reference_registry_v4_1_2026-09-20.json.'
}
if (-not (Test-Path -LiteralPath $sourceMaterialsRoot -PathType Container)) {
    throw 'Missing 03_reference/source_materials/.'
}

$profileConfig = Get-Content -LiteralPath $profilePath -Raw -Encoding UTF8 | ConvertFrom-Json
$repro = Get-Content -LiteralPath $reproPath -Raw -Encoding UTF8 | ConvertFrom-Json
$validation = Get-Content -LiteralPath $validationPath -Raw -Encoding UTF8 | ConvertFrom-Json
$registry = Get-Content -LiteralPath $registryPath -Raw -Encoding UTF8 | ConvertFrom-Json

$sourceManifest = @($registry.source_manifest)
$sourceMaterialFiles = @(Get-ChildItem -LiteralPath $sourceMaterialsRoot -File)
if ($sourceManifest.Count -ne 14 -or $sourceMaterialFiles.Count -ne 14) {
    throw "Source-material count mismatch: registry=$($sourceManifest.Count), files=$($sourceMaterialFiles.Count), expected=14."
}
$sourceMaterialFailures = New-Object System.Collections.Generic.List[string]
foreach ($entry in $sourceManifest) {
    $relative = Convert-ToPackagePath ([string]$entry.archive_relative_path)
    if (-not $relative.StartsWith('03_reference/source_materials/', [System.StringComparison]::Ordinal)) {
        $sourceMaterialFailures.Add("invalid registry path: $relative")
        continue
    }
    try {
        $absolute = Get-PackageFile $relative
        $actualBytes = (Get-Item -LiteralPath $absolute).Length
        $actualHash = Get-Sha256 $absolute
        if ($actualBytes -ne [int64]$entry.bytes) {
            $sourceMaterialFailures.Add("byte mismatch: $relative")
        }
        if ($actualHash -ne ([string]$entry.sha256).ToLowerInvariant()) {
            $sourceMaterialFailures.Add("sha256 mismatch: $relative")
        }
    }
    catch {
        $sourceMaterialFailures.Add("missing source material: $relative")
    }
}
if ($sourceMaterialFailures.Count -ne 0) {
    throw "Source-material integrity failure: $($sourceMaterialFailures -join '; ')"
}

$canonicalMarkdown = New-Object System.Collections.Generic.List[object]
foreach ($directory in @('00_control', '01_research', '02_evidence', '03_reference')) {
    foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $packageRoot $directory) -File -Filter '*.md')) {
        $canonicalMarkdown.Add($file)
    }
}
$brokenCanonicalLinks = @(Test-MarkdownLinks $canonicalMarkdown)
if ($brokenCanonicalLinks.Count -ne 0) {
    throw "Broken canonical Markdown links: $($brokenCanonicalLinks -join '; ')"
}

if ([int]$profileConfig.schema_version -ne 1) {
    throw "Unsupported review profile schema: $($profileConfig.schema_version)"
}
$effectiveProfile = if ($Profile -eq 'current') { [string]$profileConfig.active_profile } else { $Profile }
if ($effectiveProfile -ne [string]$profileConfig.active_profile) {
    throw "Requested profile '$effectiveProfile' does not match the active profile '$($profileConfig.active_profile)'."
}
if ([string]$profileConfig.official_test -ne 'sealed_unread') {
    throw 'Review packet generation requires official_test=sealed_unread.'
}

$requestedSources = New-Object System.Collections.Generic.List[string]
$requestedSources.Add('00_control/review_profile.json')
$requestedSources.Add('02_evidence/reproducibility_current.json')
foreach ($attachment in @($profileConfig.attachments)) {
    $requestedSources.Add([string]$attachment.source)
}
foreach ($summary in @($profileConfig.summaries)) {
    foreach ($source in @($summary.sources)) {
        $requestedSources.Add([string]$source)
    }
}
$requestedSources = @($requestedSources | Sort-Object -Unique)

foreach ($source in $requestedSources) {
    if ($source -match '(?i)(^|/)(official[-_ ]?test|test[-_ ]?prediction|test[-_ ]?artifact)') {
        throw "Unauthorized official-test source requested by profile: $source"
    }
    if ($source -match '^(?i)(03_reference/source_materials/|90_archive/|03_reference/MMFR_reference_(index|registry))') {
        throw "Hard-excluded source requested by L1 profile: $source"
    }
    [void](Get-PackageFile $source)
}

$sourceRecords = @($requestedSources | ForEach-Object { Get-SourceRecord $_ })
$profileHash = Get-Sha256 $profilePath
$gitIdentity = Get-GitIdentity
$identityMaterial = @(
    "profile=$effectiveProfile",
    "profile_schema=$($profileConfig.schema_version)",
    "profile_sha256=$profileHash",
    "git_commit=$($gitIdentity.commit)"
) + @($sourceRecords | Sort-Object path | ForEach-Object { "source=$($_.path):$($_.sha256)" })
$generationId = Get-StringSha256 ($identityMaterial -join "`n")

if ($DryRun) {
    [ordered]@{
        profile = $effectiveProfile
        review_level = [string]$profileConfig.review_level
        generation_id = $generationId
        source_count = $sourceRecords.Count
        planned_packet_files = $expectedPacketFiles
        official_test = [string]$profileConfig.official_test
        dry_run = $true
    } | ConvertTo-Json -Depth 6
    exit 0
}

Remove-Item -LiteralPath $packetRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $packetRoot -Force | Out-Null

$attachmentRecords = New-Object System.Collections.Generic.List[object]
foreach ($attachment in @($profileConfig.attachments)) {
    $packetFile = [string]$attachment.packet_file
    if ($packetFile -match '[\\/]' -or [string]::IsNullOrWhiteSpace($packetFile)) {
        throw "Packet attachment must be a flat file name: $packetFile"
    }
    $sourceRelative = Convert-ToPackagePath ([string]$attachment.source)
    $sourceAbsolute = Get-PackageFile $sourceRelative
    $destination = Join-Path $packetRoot $packetFile
    Copy-Item -LiteralPath $sourceAbsolute -Destination $destination -Force
    $sourceHash = Get-Sha256 $sourceAbsolute
    $packetHash = Get-Sha256 $destination
    if ([string]$attachment.transform -eq 'exact_copy' -and $sourceHash -ne $packetHash) {
        throw "Exact-copy hash mismatch for $packetFile"
    }
    $attachmentRecords.Add([ordered]@{
        packet_file = $packetFile
        role = [string]$attachment.role
        source = $sourceRelative
        source_sha256 = $sourceHash
        packet_sha256 = $packetHash
        bytes = (Get-Item -LiteralPath $destination).Length
        transform = [string]$attachment.transform
    })
}

$brief = New-Object System.Collections.Generic.List[string]
$brief.Add('# MMFR review brief')
$brief.Add('')
$brief.Add('## Review identity')
$brief.Add('')
$brief.Add(('- Profile: `{0}`' -f $effectiveProfile))
$brief.Add(('- Review level: `{0}`' -f $profileConfig.review_level))
$brief.Add(('- Generation identity: `{0}`' -f $generationId))
$brief.Add(('- Official test: `{0}`' -f $profileConfig.official_test))
$brief.Add('')
$brief.Add('## Authoritative research state')
$brief.Add('')
$brief.Add(('- Research status: `{0}`' -f (Convert-AuthorityPath ([string]$profileConfig.authoritative_status.research_status))))
$brief.Add(('- Open decisions: `{0}`' -f (Convert-AuthorityPath ([string]$profileConfig.authoritative_status.open_decisions))))
$brief.Add('')
$brief.Add('This generated packet is not a research-authorization authority. The two repository documents above remain authoritative.')
$brief.Add('')
$brief.Add('## Current task')
$brief.Add('')
$brief.Add([string]$profileConfig.review.title)
$brief.Add('')
$brief.Add([string]$profileConfig.review.current_task)
$brief.Add('')
Add-BulletSection $brief 'Current state and boundary' $profileConfig.review.state_summary
$brief.Add('## Please review')
$brief.Add('')
$questionIndex = 1
foreach ($question in @($profileConfig.review.review_questions)) {
    $brief.Add("$questionIndex. $question")
    $questionIndex++
}
$brief.Add('')
Add-BulletSection $brief 'Do not infer or authorize' $profileConfig.review.do_not_infer_or_authorize
$brief.Add('## Recommended reading')
$brief.Add('')
$readingIndex = 1
foreach ($packetFile in @($profileConfig.review.reading_order)) {
    $brief.Add("$readingIndex. [$packetFile]($packetFile)")
    $readingIndex++
}
$brief.Add('')
Add-BulletSection $brief 'Changed since previous review' $profileConfig.review.changed_since_previous_review
Add-BulletSection $brief 'Background not included' $profileConfig.review.background_not_included
$brief.Add('The excluded material remains in the canonical package and may be added only by a profile that explicitly requires it.')
Set-Content -LiteralPath (Join-Path $packetRoot 'REVIEW_BRIEF.md') -Value $brief -Encoding UTF8

$evidence = New-Object System.Collections.Generic.List[string]
$evidence.Add('# Evidence summary')
$evidence.Add('')
$evidence.Add(('- Profile: `{0}`' -f $effectiveProfile))
$evidence.Add(('- Generation identity: `{0}`' -f $generationId))
$evidence.Add('- Scope: review-relevant conclusions extracted from canonical sources; canonical files remain authoritative.')
$evidence.Add('')
foreach ($summary in @($profileConfig.summaries)) {
    $evidence.Add("## $($summary.title)")
    $evidence.Add('')
    $evidence.Add("- Status: $($summary.status)")
    $evidence.Add("- Current relevance: $($summary.current_relevance)")
    $evidence.Add('- Canonical sources:')
    foreach ($source in @($summary.sources)) {
        $record = $sourceRecords | Where-Object { $_.path -eq (Convert-ToPackagePath ([string]$source)) } | Select-Object -First 1
        $evidence.Add(('  - `{0}` - SHA-256 `{1}`' -f $record.path, $record.sha256))
    }
    $evidence.Add('- Key conclusions:')
    foreach ($conclusion in @($summary.key_conclusions)) {
        $evidence.Add("  - $conclusion")
    }
    if ([string]$summary.id -eq 'reproducibility_identity') {
        $evidence.Add('- Review identity fields:')
        $evidence.Add(('  - Git branch: `{0}`' -f $gitIdentity.branch))
        $evidence.Add(('  - Git commit: `{0}`' -f $gitIdentity.commit))
        $evidence.Add(('  - Tracked workspace dirty: `{0}`' -f $gitIdentity.workspace_dirty.ToString().ToLowerInvariant()))
        $evidence.Add('  - Evaluator: `N/A - Gate-B engineering qualification`')
        $evidence.Add('  - Checkpoint effect selection: `N/A - not performed`')
        $evidence.Add(('  - Official test: `{0}`' -f $profileConfig.official_test))
    }
    $evidence.Add('')
}
Set-Content -LiteralPath (Join-Path $packetRoot 'EVIDENCE_SUMMARY.md') -Value $evidence -Encoding UTF8

$packetRecords = New-Object System.Collections.Generic.List[object]
$briefPath = Join-Path $packetRoot 'REVIEW_BRIEF.md'
$packetRecords.Add([ordered]@{
    packet_file = 'REVIEW_BRIEF.md'
    role = 'generated_review_entry'
    source = $null
    source_sha256 = $null
    packet_sha256 = Get-Sha256 $briefPath
    bytes = (Get-Item -LiteralPath $briefPath).Length
    transform = 'generated_from_profile'
    input_sources = @('00_control/review_profile.json', '02_evidence/reproducibility_current.json')
})
foreach ($record in $attachmentRecords) {
    $packetRecords.Add($record)
}
$evidencePath = Join-Path $packetRoot 'EVIDENCE_SUMMARY.md'
$packetRecords.Add([ordered]@{
    packet_file = 'EVIDENCE_SUMMARY.md'
    role = 'generated_evidence_summary'
    source = $null
    source_sha256 = $null
    packet_sha256 = Get-Sha256 $evidencePath
    bytes = (Get-Item -LiteralPath $evidencePath).Length
    transform = 'generated_from_canonical_sources'
    input_sources = @($sourceRecords | ForEach-Object { [ordered]@{ path = $_.path; sha256 = $_.sha256 } })
})
$packetRecords.Add([ordered]@{
    packet_file = 'packet_manifest.json'
    role = 'packet_machine_identity'
    source = $null
    source_sha256 = $null
    packet_sha256 = $null
    bytes = $null
    transform = 'generated_manifest'
    self_hash = 'omitted-to-avoid-self-reference'
})

$manifest = [ordered]@{
    schema_version = 'mmfr-review-packet-v2'
    profile_schema_version = [int]$profileConfig.schema_version
    profile = $effectiveProfile
    review_level = [string]$profileConfig.review_level
    generation_id = $generationId
    deterministic_identity = 'sha256(profile identity + canonical input hashes + git commit); no wall-clock timestamp'
    source_package = 'MMFR_v4_1_blueprint_and_reference_package_2026-09-20'
    git = $gitIdentity
    authoritative_status = [ordered]@{
        research_status = Convert-AuthorityPath ([string]$profileConfig.authoritative_status.research_status)
        open_decisions = Convert-AuthorityPath ([string]$profileConfig.authoritative_status.open_decisions)
    }
    official_test = [string]$profileConfig.official_test
    packet_file_count = 6
    files = @($packetRecords | ForEach-Object { $_ })
    canonical_inputs = @($sourceRecords | ForEach-Object { $_ })
    hard_exclusions = @($profileConfig.hard_exclusions | ForEach-Object { [string]$_ })
}
$manifestPath = Join-Path $packetRoot 'packet_manifest.json'
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

[void](Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json)
$actualFiles = @(Get-ChildItem -LiteralPath $packetRoot -File | Sort-Object Name | ForEach-Object { $_.Name })
$expectedSorted = @($expectedPacketFiles | Sort-Object)
if (($actualFiles -join "`n") -ne ($expectedSorted -join "`n")) {
    throw "Packet file set mismatch. Actual: $($actualFiles -join ', ')"
}
if (@(Get-ChildItem -LiteralPath $packetRoot -Directory).Count -ne 0) {
    throw 'Review packet must be flat; a child directory was generated.'
}
$brokenPacketLinks = @(Test-MarkdownLinks @(Get-ChildItem -LiteralPath $packetRoot -File -Filter '*.md'))
if ($brokenPacketLinks.Count -ne 0) {
    throw "Broken packet links: $($brokenPacketLinks -join '; ')"
}

$tempPacketRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("mmfr-review-packet-" + $generationId)
$brokenCopiedPacketLinks = @()
try {
    Remove-Item -LiteralPath $tempPacketRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $tempPacketRoot -Force | Out-Null
    foreach ($file in @(Get-ChildItem -LiteralPath $packetRoot -File)) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $tempPacketRoot $file.Name) -Force
    }
    $brokenCopiedPacketLinks = @(Test-MarkdownLinks @(Get-ChildItem -LiteralPath $tempPacketRoot -File -Filter '*.md'))
    if ($brokenCopiedPacketLinks.Count -ne 0) {
        throw "Broken links after independent packet copy: $($brokenCopiedPacketLinks -join '; ')"
    }
}
finally {
    Remove-Item -LiteralPath $tempPacketRoot -Recurse -Force -ErrorAction SilentlyContinue
}
foreach ($record in $attachmentRecords) {
    if ($record.transform -eq 'exact_copy' -and $record.source_sha256 -ne $record.packet_sha256) {
        throw "Post-build exact-copy mismatch: $($record.packet_file)"
    }
}

$packetHashes = @(
    Get-ChildItem -LiteralPath $packetRoot -File |
        Sort-Object Name |
        ForEach-Object {
            [ordered]@{
                file = $_.Name
                sha256 = Get-Sha256 $_.FullName
                bytes = $_.Length
            }
        }
)

[ordered]@{
    review_level = [string]$profileConfig.review_level
    generation_id = $generationId
    source_material_count = [int]$repro.source_materials.count
    packet_file_count = $actualFiles.Count
    packet_files = $actualFiles
    packet_hashes = $packetHashes
    source_material_integrity = '14/14 bytes-and-sha256-match-registry'
    broken_canonical_links = $brokenCanonicalLinks.Count
    broken_packet_links = $brokenPacketLinks.Count
    broken_copied_packet_links = $brokenCopiedPacketLinks.Count
    official_test = [string]$profileConfig.official_test
    packet_path = '99_review_packet_current'
    dry_run = $false
} | ConvertTo-Json -Depth 6
