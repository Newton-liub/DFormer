# 单向同步：.cursor/ -> .codebuddy/（Rules 与 Skills）
#
# 设计原则：
#   1. 只做增量写入，从不删除、从不重建目录；.codebuddy 独有文件原样保留。
#   2. 正文路径引用自动改写 .cursor -> .codebuddy。
#   3. frontmatter 自动移除 CodeBuddy 不支持的字段（disable-model-invocation），
#      并保留目标文件里 CodeBuddy 专有字段（如 description_zh）。
#   4. 内容无变化时不写盘；重复运行结果稳定（幂等）。
#   5. 只用 PowerShell 自带能力，无外部依赖、不触网。
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File .codebuddy\sync-from-cursor.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .codebuddy\sync-from-cursor.ps1 -WhatIf

[CmdletBinding()]
param(
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root       = Split-Path -Parent $ScriptDir
$CursorRules  = Join-Path $Root '.cursor\rules'
$CursorSkills = Join-Path $Root '.cursor\skills'
$CbRules      = Join-Path $Root '.codebuddy\rules'
$CbSkills     = Join-Path $Root '.codebuddy\skills'

# CodeBuddy 不支持的 frontmatter 字段，写入前丢弃
$DropFields = @('disable-model-invocation')

$script:Created = 0
$script:Updated = 0
$script:Same    = 0

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Get-FieldKey {
    param([string]$Line)
    if ($Line -match '^\s*([A-Za-z0-9_.\-]+)\s*:') { return $Matches[1] }
    return $null
}

function Split-FrontMatter {
    param([string]$Text)
    $lines = $Text -split "`r?`n"
    if ($lines.Count -gt 0 -and $lines[0].Trim() -eq '---') {
        $header = New-Object System.Collections.ArrayList
        $i = 1
        while ($i -lt $lines.Count -and $lines[$i].Trim() -ne '---') {
            [void]$header.Add($lines[$i])
            $i++
        }
        if ($i -lt $lines.Count) { $i++ }
        $body = @()
        if ($i -lt $lines.Count) { $body = $lines[$i..($lines.Count - 1)] }
        return [pscustomobject]@{ HasHeader = $true; Header = @($header); Body = $body }
    }
    return [pscustomobject]@{ HasHeader = $false; Header = @(); Body = $lines }
}

function Convert-Body {
    param([string]$Text)
    # 路径引用改写：正反斜杠两种写法都覆盖
    $Text = $Text -replace '\.cursor/rules',  '.codebuddy/rules'
    $Text = $Text -replace '\.cursor/skills', '.codebuddy/skills'
    $Text = $Text -replace '\.cursor\\rules',  '.codebuddy\rules'
    $Text = $Text -replace '\.cursor\\skills', '.codebuddy\skills'
    return $Text
}

function Merge-FrontMatter {
    param(
        [string[]]$SourceHeader,
        [string[]]$TargetHeader
    )
    $sourceKeys = @()
    $out = New-Object System.Collections.ArrayList
    foreach ($line in $SourceHeader) {
        $key = Get-FieldKey $line
        if ($key -and ($DropFields -contains $key)) { continue }
        if ($key) { $sourceKeys += $key }
        [void]$out.Add($line)
    }
    # 保留目标文件里 CodeBuddy 专有字段
    foreach ($line in $TargetHeader) {
        $key = Get-FieldKey $line
        if (-not $key) { continue }
        if ($DropFields -contains $key) { continue }
        if ($sourceKeys -contains $key) { continue }
        [void]$out.Add($line)
    }
    return @($out)
}

function Sync-TextFile {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Label
    )
    $sourceText = Get-Content -Path $SourcePath -Raw -Encoding UTF8
    $targetText = $null
    if (Test-Path $TargetPath) {
        $targetText = Get-Content -Path $TargetPath -Raw -Encoding UTF8
    }

    $src = Split-FrontMatter $sourceText
    $tgt = Split-FrontMatter ([string]$targetText)

    if (-not $src.HasHeader) {
        $newText = Convert-Body $sourceText
    }
    else {
        $mergedHeader = Merge-FrontMatter -SourceHeader $src.Header -TargetHeader $tgt.Header
        $parts = New-Object System.Collections.ArrayList
        [void]$parts.Add('---')
        foreach ($l in $mergedHeader) { [void]$parts.Add($l) }
        [void]$parts.Add('---')
        foreach ($l in $src.Body) { [void]$parts.Add($l) }
        $newText = ($parts -join "`r`n")
        if (-not $newText.EndsWith("`r`n")) { $newText += "`r`n" }
        $newText = Convert-Body $newText
    }

    $oldNorm = $null
    if ($null -ne $targetText) { $oldNorm = ($targetText -replace "`r`n", "`n") }
    $newNorm = ($newText -replace "`r`n", "`n")

    if ($oldNorm -eq $newNorm) {
        Write-Host ("  [same]   " + $Label)
        $script:Same++
        return
    }

    if ($null -eq $oldNorm) {
        Write-Host ("  [new]    " + $Label)
        $script:Created++
    }
    else {
        Write-Host ("  [update] " + $Label)
        $script:Updated++
    }
    if (-not $WhatIf) { Write-Utf8NoBom -Path $TargetPath -Text $newText }
}

function Sync-NewFilesOnly {
    param([string]$SourceDir, [string]$TargetDir, [string]$Label)
    if (-not (Test-Path $SourceDir)) { return }
    Get-ChildItem -Path $SourceDir -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($SourceDir.Length).TrimStart('\', '/')
        $dest = Join-Path $TargetDir $rel
        if (Test-Path $dest) { return }
        Write-Host ("  [copy]   " + $Label + '/' + $rel + '  (目标不存在的附属文件)')
        if (-not $WhatIf) {
            $dir = Split-Path -Parent $dest
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
            Copy-Item -Path $_.FullName -Destination $dest
        }
        $script:Created++
    }
}

function Show-TargetOnly {
    param([string]$SourceDir, [string]$TargetDir, [string]$Label)
    if (-not (Test-Path $TargetDir)) { return }
    $srcNames = @()
    if (Test-Path $SourceDir) {
        $srcNames = Get-ChildItem -Path $SourceDir -Recurse -File -Force |
            ForEach-Object { $_.FullName.Substring($SourceDir.Length).TrimStart('\', '/') }
    }
    Get-ChildItem -Path $TargetDir -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($TargetDir.Length).TrimStart('\', '/')
        if ($srcNames -notcontains $rel) {
            Write-Host ("  [keep]   " + $Label + '/' + $rel + '  (CodeBuddy 独有，未改动)')
        }
    }
}

Write-Host ("项目根目录: " + $Root)
if ($WhatIf) { Write-Host '模式: WhatIf（只预览，不写盘）' }

Write-Host ''
Write-Host '== Rules (.cursor/rules -> .codebuddy/rules) =='
if (Test-Path $CursorRules) {
    Get-ChildItem -Path $CursorRules -Recurse -File -Force |
        Where-Object { $_.Extension -in @('.mdc', '.md') } |
        Sort-Object FullName |
        ForEach-Object {
            $rel = $_.FullName.Substring($CursorRules.Length).TrimStart('\', '/')
            Sync-TextFile -SourcePath $_.FullName -TargetPath (Join-Path $CbRules $rel) -Label $rel
        }
}
else {
    Write-Host '  (未找到 .cursor/rules，跳过)'
}
Show-TargetOnly -SourceDir $CursorRules -TargetDir $CbRules -Label 'rules'

Write-Host ''
Write-Host '== Skills (.cursor/skills -> .codebuddy/skills) =='
if (Test-Path $CursorSkills) {
    Get-ChildItem -Path $CursorSkills -Directory -Force | Sort-Object Name | ForEach-Object {
        $skillMd = Join-Path $_.FullName 'SKILL.md'
        if (Test-Path $skillMd) {
            Sync-TextFile -SourcePath $skillMd -TargetPath (Join-Path (Join-Path $CbSkills $_.Name) 'SKILL.md') `
                -Label ($_.Name + '/SKILL.md')
        }
        else {
            Write-Host ("  [skip]   " + $_.Name + '  (缺少 SKILL.md)')
        }
    }
    # 附属文件（references、scripts 等）只在目标缺失时补齐，不覆盖
    Sync-NewFilesOnly -SourceDir $CursorSkills -TargetDir $CbSkills -Label 'skills'
}
else {
    Write-Host '  (未找到 .cursor/skills，跳过)'
}
Show-TargetOnly -SourceDir $CursorSkills -TargetDir $CbSkills -Label 'skills'

Write-Host ''
$mode = ''
if ($WhatIf) { $mode = '（WhatIf 未写盘）' }
Write-Host ("完成" + $mode + "：新增 " + $script:Created + "，更新 " + $script:Updated + "，无变化 " + $script:Same)
Write-Host '说明：脚本从不删除文件；.codebuddy 独有内容（含 description_zh 等字段）始终保留。'
