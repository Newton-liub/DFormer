<# Local supervisor for an already submitted, explicitly authorized formal job.
   This never starts GPU jobs, retries training, changes specifications or extends insurance.
   It actively stops GPU after job completion/failure and performs only CPU-only retrieval.
#>
param(
    [Parameter(Mandatory=$true)][string]$JobId,
    [Parameter(Mandatory=$true)][string]$RemoteRoot,
    [Parameter(Mandatory=$true)][string]$LocalRoot,
    [Parameter(Mandatory=$true)][string]$FormalSha,
    [Parameter(Mandatory=$true)][long]$GpuDeadline,
    [string]$Instance = 'cpod-1vbh7faqcauq'
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
$record = [ordered]@{
    instance = $Instance; formal_run_sha = $FormalSha; job_id = $JobId
    remote_root = $RemoteRoot; gpu_shutdown_schedule_unix = $GpuDeadline
    supervisor_started_at = [DateTimeOffset]::Now.ToString('o')
    events = @(); final_bill = 'not_verified'
}
$recordPath = Join-Path $LocalRoot 'lifecycle.json'
function Save-Record {
    $record | ConvertTo-Json -Depth 15 | Set-Content -Encoding UTF8 -Path $recordPath
}
function Invoke-Cloud([string[]]$CloudArgs) {
    $lines = & compshare --json @CloudArgs
    $exitCode = $LASTEXITCODE
    try { $value = ($lines -join "`n") | ConvertFrom-Json } catch {
        throw "Non-JSON cloud response (exit $exitCode)"
    }
    if (-not $value.ok) {
        throw "Cloud command failed: $($CloudArgs[0..2] -join ' '); $($value.error.code): $($value.error.message)"
    }
    # A terminal failed job can return valid JSON with ok=true but a nonzero exit.
    $isJobWait = $CloudArgs.Count -ge 3 -and $CloudArgs[1] -eq 'job' -and $CloudArgs[2] -eq 'wait'
    if ($exitCode -ne 0 -and -not $isJobWait) {
        throw "Cloud command returned exit $exitCode despite ok=true: $($CloudArgs[0..1] -join ' ')"
    }
    return $value
}
function Add-Event([string]$Name, $Value) {
    $record.events += [ordered]@{ event=$Name; at=[DateTimeOffset]::Now.ToString('o'); data=$Value }
    Save-Record
    Write-Output "$Name $([DateTimeOffset]::Now.ToString('o'))"
}
function Stop-And-Confirm([switch]$RequireGpuZero) {
    $stop = Invoke-Cloud -CloudArgs @('instance','stop',$Instance,'--yes','--timeout','600')
    $show = Invoke-Cloud -CloudArgs @('instance','show',$Instance,'--status','--spec','--billing')
    $state = @($show.data.UHostSet)[0]
    Add-Event 'stopped_direct_query' $state
    if ($state.State -ne 'Stopped' -or ($RequireGpuZero -and [int]$state.GPU -ne 0)) {
        throw 'Instance did not directly confirm the required stopped state'
    }
    # After a GPU-mode stop the API can retain GPU=1 as configured specification.
    # CPU-only final shutdown must directly show both Stopped and GPU=0.
    $record.final_state = $state.State
    $record.final_gpu = [int]$state.GPU
    Save-Record
}
Save-Record
$gpuStopped = $false
try {
    # Leave ten minutes before the already fixed platform deadline. No extension.
    $remaining = $GpuDeadline - [DateTimeOffset]::Now.ToUnixTimeSeconds() - 600
    if ($remaining -le 0) { throw 'Too close to fixed GPU insurance deadline' }
    $wait = Invoke-Cloud -CloudArgs @('instance','job','wait',$Instance,$JobId,'--timeout',"$remaining",'--interval','30')
    Add-Event 'formal_job_terminal' $wait.data
    $job = $wait.data.job
    if (-not $job) { $job = $wait.data }
    if ($job.State -notin @('Succeeded','Failed','Cancelled','Interrupted')) {
        throw 'Job wait did not provide a recognized terminal state'
    }
    if ($job.State -ne 'Succeeded' -or [int]$job.ExitCode -ne 0) {
        throw "Formal job terminated with state $($job.State), exit $($job.ExitCode); no automatic restart"
    }
} catch {
    $record.wait_error = $_.Exception.Message
    Save-Record
    Write-Output "Formal supervisor interrupted: $($record.wait_error)"
} finally {
    Stop-And-Confirm
    $gpuStopped = $true
}
# A failed run can still have necessary failure receipts/logs to retrieve.
# Never automatically restart after a manually stopped/interrupted job.
if ($record.wait_error) {
    Write-Output 'GPU stopped; job wait was unsuccessful. CPU restart requires parent review.'
    exit 1
}
$cpuStarted = $false
try {
    $cpuDeadline = [DateTimeOffset]::Now.AddMinutes(30).ToUnixTimeSeconds()
    $schedule = Invoke-Cloud -CloudArgs @('instance','schedule','set',$Instance,'--at',"$cpuDeadline")
    $confirmed = Invoke-Cloud -CloudArgs @('instance','schedule','show',$Instance)
    Add-Event 'cpu_30_minute_schedule' $confirmed.data
    if ($confirmed.data.scheduled -ne $true -or [long]$confirmed.data.scheduler_stop_time -ne $cpuDeadline) {
        throw 'CPU shutdown insurance did not confirm the exact deadline'
    }
    # Respect an unexpected prior stop: only retrieve after a known terminal job.
    $start = Invoke-Cloud -CloudArgs @('instance','start',$Instance,'--without-gpu','A','--timeout','600')
    $cpuStarted = $true
    $show = Invoke-Cloud -CloudArgs @('instance','show',$Instance,'--status','--spec','--billing')
    $state = @($show.data.UHostSet)[0]
    Add-Event 'cpu_running_direct_query' $state
    if ($state.State -ne 'Running' -or [int]$state.GPU -ne 0) { throw 'CPU retrieval did not confirm Running/GPU0' }
    $confirmation = Invoke-Cloud -CloudArgs @('instance','schedule','show',$Instance)
    Add-Event 'cpu_schedule_after_start' $confirmation.data
    if ($confirmation.data.scheduled -ne $true -or [long]$confirmation.data.scheduler_stop_time -ne $cpuDeadline) {
        throw 'CPU shutdown insurance changed after start'
    }
    $copy = Invoke-Cloud -CloudArgs @('instance','cp',$Instance,":$RemoteRoot/monitor",(Join-Path $LocalRoot 'monitor'))
    Add-Event 'monitor_transfer_complete' $copy.data
    # Only a successful formal run is expected to have all three fixed finals.
    $receipt = Get-Content -Raw -Path (Join-Path $LocalRoot 'monitor/formal-run-receipt.json') | ConvertFrom-Json
    $record.formal_status = $receipt.status
    Save-Record
    if ($receipt.status -eq 'SUCCEEDED') {
        foreach ($strategy in @('Natural','Grid','Replay')) {
            $remote = "$RemoteRoot/checkpoints/NaturalMissing-$strategy-formal/development/seed-772961337/checkpoint/update-2560.pth"
            $destination = Join-Path $LocalRoot "checkpoints/$strategy/update-2560.pth"
            New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
            $copy = Invoke-Cloud -CloudArgs @('instance','cp',$Instance,":$remote",$destination)
            Add-Event "$strategy-final-transfer-complete" $copy.data
        }
    }
} catch {
    $record.retrieval_error = $_.Exception.Message
    Save-Record
    Write-Output "CPU retrieval error: $($record.retrieval_error)"
} finally {
    # Include start timeout/partial start: querying and stopping is safer than assuming.
    Stop-And-Confirm -RequireGpuZero
}
$record.supervisor_finished_at = [DateTimeOffset]::Now.ToString('o')
Save-Record
if ($record.retrieval_error -or $record.formal_status -ne 'SUCCEEDED') { exit 1 }
Write-Output 'FORMAL_COLLECT_COMPLETE; INSTANCE_STOPPED_GPU0'
