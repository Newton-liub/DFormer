<# Supervise one bounded numerical job, stop GPU, retrieve only in CPU-only mode.
   Never submits/retries training or changes the configured GPU specification. #>
param(
    [Parameter(Mandatory=$true)][string]$JobId,
    [Parameter(Mandatory=$true)][string]$RemoteRoot,
    [Parameter(Mandatory=$true)][string]$LocalRoot,
    [Parameter(Mandatory=$true)][string]$SourceSha,
    [Parameter(Mandatory=$true)][long]$GpuDeadline,
    [string]$Instance = 'cpod-1vbh7faqcauq'
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
$record = [ordered]@{ instance=$Instance; source_sha=$SourceSha; job_id=$JobId; remote_root=$RemoteRoot
    gpu_deadline_unix=$GpuDeadline; started_at=[DateTimeOffset]::Now.ToString('o'); events=@() }
$recordPath = Join-Path $LocalRoot 'numerical-lifecycle.json'
function Save-Record { $record | ConvertTo-Json -Depth 18 | Set-Content -Encoding UTF8 -Path $recordPath }
function Cloud([string[]]$CloudArgs) {
    $lines = & compshare --json @CloudArgs
    $code = $LASTEXITCODE
    try { $value = ($lines -join "`n") | ConvertFrom-Json } catch { throw "Non-JSON cloud response (exit $code)" }
    $isWait = $CloudArgs.Count -ge 3 -and $CloudArgs[1] -eq 'job' -and $CloudArgs[2] -eq 'wait'
    if (-not $value.ok -or ($code -ne 0 -and -not $isWait)) {
        throw "Cloud operation failed: $($CloudArgs[0..1] -join ' '); $($value.error.code)"
    }
    return $value
}
function Event([string]$Name, $Value) {
    $record.events += [ordered]@{ event=$Name; at=[DateTimeOffset]::Now.ToString('o'); data=$Value }
    Save-Record
    Write-Output "$Name $([DateTimeOffset]::Now.ToString('o'))"
}
function Stop-Confirm([switch]$Cpu) {
    $null = Cloud @('instance','stop',$Instance,'--yes','--wait','--timeout','600')
    $show = Cloud @('instance','show',$Instance,'--status','--spec','--billing')
    $state = @($show.data.UHostSet)[0]
    Event 'direct_stopped_query' $state
    if ($state.State -ne 'Stopped' -or ($Cpu -and [int]$state.GPU -ne 0)) { throw 'Required stopped state not confirmed' }
    $record.final_state = $state.State
    $record.final_gpu_specification = [int]$state.GPU
    Save-Record
}
Save-Record
$terminal = $false
try {
    $remaining = $GpuDeadline - [DateTimeOffset]::Now.ToUnixTimeSeconds() - 600
    if ($remaining -le 0) { throw 'Insufficient time before fixed GPU shutdown insurance' }
    $wait = Cloud @('instance','job','wait',$Instance,$JobId,'--timeout',"$remaining",'--interval','10')
    Event 'numerical_job_terminal' $wait.data
    $job = $wait.data.job
    if (-not $job) { $job = $wait.data }
    if ($job.State -notin @('Succeeded','Failed')) {
        throw 'No completed/failed job; CPU restart requires parent review (possible user interruption)'
    }
    $terminal = $true
    $record.job_state = $job.State
    $record.job_exit_code = $job.ExitCode
    Save-Record
} catch {
    $record.wait_error_type = $_.Exception.GetType().Name
    $record.wait_error = $_.Exception.Message
    Save-Record
} finally { Stop-Confirm }
if (-not $terminal) { Write-Output 'GPU stopped; no automatic CPU restart after interruption'; exit 1 }
try {
    $cpuDeadline = [DateTimeOffset]::Now.AddMinutes(15).ToUnixTimeSeconds()
    $null = Cloud @('instance','schedule','set',$Instance,'--at',"$cpuDeadline")
    $schedule = Cloud @('instance','schedule','show',$Instance)
    Event 'cpu_insurance_before_start' $schedule.data
    if (-not $schedule.data.scheduled -or [long]$schedule.data.scheduler_stop_time -ne $cpuDeadline) {
        throw 'CPU insurance mismatch'
    }
    $null = Cloud @('instance','start',$Instance,'--without-gpu','A','--wait','--timeout','600')
    $show = Cloud @('instance','show',$Instance,'--status','--spec','--billing')
    $state = @($show.data.UHostSet)[0]
    Event 'direct_cpu_running_query' $state
    if ($state.State -ne 'Running' -or [int]$state.GPU -ne 0) { throw 'Retrieval requires Running/GPU0' }
    $schedule = Cloud @('instance','schedule','show',$Instance)
    Event 'cpu_insurance_after_start' $schedule.data
    if (-not $schedule.data.scheduled -or [long]$schedule.data.scheduler_stop_time -ne $cpuDeadline) {
        throw 'CPU insurance changed on start'
    }
    $copy = Cloud @('instance','cp',$Instance,":$RemoteRoot/monitor",(Join-Path $LocalRoot 'monitor'))
    Event 'cpu_monitor_transfer_complete' $copy.data
    $record.monitor_retrieved = $true
    Save-Record
} catch {
    $record.retrieval_error = $_.Exception.Message
    Save-Record
} finally { Stop-Confirm -Cpu }
$record.finished_at = [DateTimeOffset]::Now.ToString('o')
Save-Record
if (-not $record.monitor_retrieved) { exit 1 }
Write-Output 'NUMERICAL_COLLECT_COMPLETE; INSTANCE_STOPPED_GPU0'
