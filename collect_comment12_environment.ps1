param(
    [string]$BaseImage = '',
    [string]$FinalImageUri = ''
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Get-ProjectRoot {
    if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        return [System.IO.Path]::GetFullPath($PSScriptRoot)
    }
    return [System.IO.Path]::GetFullPath((Get-Location).Path)
}

function Write-Utf8NoBomFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Content
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'Write-Utf8NoBomFile received an empty path.'
    }

    $absolutePath = [System.IO.Path]::GetFullPath($Path)
    $parent = [System.IO.Path]::GetDirectoryName($absolutePath)

    if ([string]::IsNullOrWhiteSpace($parent)) {
        throw "Could not determine the parent directory for: $absolutePath"
    }

    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $encoding = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllText($absolutePath, $Content, $encoding)
}

function Get-CommandPathOrNull {
    param([Parameter(Mandatory = $true)][string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $null
    }

    if (-not [string]::IsNullOrWhiteSpace($command.Path)) {
        return $command.Path
    }

    if (-not [string]::IsNullOrWhiteSpace($command.Source)) {
        return $command.Source
    }

    return $null
}

function Invoke-Trimmed {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)

    try {
        return ((& $Command 2>$null | Out-String).Trim())
    }
    catch {
        return $null
    }
}

function Get-ExistingEnvironmentManifest {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }

    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

$ProjectRoot = Get-ProjectRoot
$OutDir = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot 'paper_outputs\environment')
)
[System.IO.Directory]::CreateDirectory($OutDir) | Out-Null

$lockPath = [System.IO.Path]::GetFullPath(
    (Join-Path $OutDir 'pip_freeze_lock.txt')
)
$manifestPath = [System.IO.Path]::GetFullPath(
    (Join-Path $OutDir 'execution_environment_manifest.json')
)

$existingManifest = Get-ExistingEnvironmentManifest -Path $manifestPath

$pythonCommand = Get-Command python -ErrorAction Stop
$pythonExe = if (-not [string]::IsNullOrWhiteSpace($pythonCommand.Path)) {
    $pythonCommand.Path
}
else {
    $pythonCommand.Source
}

if ([string]::IsNullOrWhiteSpace($pythonExe) -or -not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Could not resolve the active Python executable.'
}

$pythonVersion = (& python --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pythonVersion)) {
    throw 'Could not obtain the active Python version.'
}

$pythonHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $pythonExe
).Hash.ToLowerInvariant()

$pipFreezeLines = @(& python -m pip freeze 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "python -m pip freeze failed:`n$($pipFreezeLines -join [Environment]::NewLine)"
}

$pipFreezeText = ($pipFreezeLines | ForEach-Object { [string]$_ }) -join [Environment]::NewLine
$pipFreezeText = $pipFreezeText.TrimEnd("`r", "`n") + [Environment]::NewLine
Write-Utf8NoBomFile -Path $lockPath -Content $pipFreezeText

$lockHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath
).Hash.ToLowerInvariant()

$requirementsPath = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot 'requirements.txt')
)
if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
    throw "Missing required file: $requirementsPath"
}

$requirementsHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $requirementsPath
).Hash.ToLowerInvariant()

$os = Get-CimInstance Win32_OperatingSystem
$computerSystem = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1

if ($null -eq $os -or $null -eq $computerSystem -or $null -eq $cpu) {
    throw 'Could not collect the required Windows system information.'
}

try {
    $physicalDisks = @(Get-PhysicalDisk -ErrorAction Stop | ForEach-Object {
        [ordered]@{
            friendly_name = if ([string]::IsNullOrWhiteSpace($_.FriendlyName)) {
                'Unknown physical disk'
            }
            else {
                [string]$_.FriendlyName
            }
            media_type = if ([string]::IsNullOrWhiteSpace([string]$_.MediaType)) {
                'Unspecified'
            }
            else {
                [string]$_.MediaType
            }
            bus_type = if ([string]::IsNullOrWhiteSpace([string]$_.BusType)) {
                'Unspecified'
            }
            else {
                [string]$_.BusType
            }
            size_bytes = [int64]$_.Size
        }
    })
}
catch {
    $physicalDisks = @(Get-CimInstance Win32_DiskDrive | ForEach-Object {
        [ordered]@{
            friendly_name = if ([string]::IsNullOrWhiteSpace($_.Model)) {
                'Unknown disk drive'
            }
            else {
                [string]$_.Model
            }
            media_type = if ([string]::IsNullOrWhiteSpace($_.MediaType)) {
                'Unspecified'
            }
            else {
                [string]$_.MediaType
            }
            bus_type = if ([string]::IsNullOrWhiteSpace($_.InterfaceType)) {
                'Unspecified'
            }
            else {
                [string]$_.InterfaceType
            }
            size_bytes = [int64]$_.Size
        }
    })
}

if ($physicalDisks.Count -eq 0) {
    throw 'No storage devices were detected.'
}

try {
    $powerScheme = (& powercfg /getactivescheme 2>&1 | Out-String).Trim()
}
catch {
    $powerScheme = 'Unavailable'
}
if ([string]::IsNullOrWhiteSpace($powerScheme)) {
    $powerScheme = 'Unavailable'
}

$torchVersion = Invoke-Trimmed { python -c "import torch; print(torch.__version__)" }
$torchNumThreads = Invoke-Trimmed { python -c "import torch; print(torch.get_num_threads())" }
$torchNumInteropThreads = Invoke-Trimmed { python -c "import torch; print(torch.get_num_interop_threads())" }
$torchCudaAvailable = Invoke-Trimmed { python -c "import torch; print(torch.cuda.is_available())" }

$threadInfo = [ordered]@{
    logical_cpu_count = [Environment]::ProcessorCount
    environment = [ordered]@{
        OMP_NUM_THREADS = $env:OMP_NUM_THREADS
        MKL_NUM_THREADS = $env:MKL_NUM_THREADS
        OPENBLAS_NUM_THREADS = $env:OPENBLAS_NUM_THREADS
        NUMEXPR_NUM_THREADS = $env:NUMEXPR_NUM_THREADS
        PYTHONHASHSEED = $env:PYTHONHASHSEED
    }
    torch_num_threads = if ([string]::IsNullOrWhiteSpace($torchNumThreads)) {
        $null
    }
    else {
        [int]$torchNumThreads
    }
    torch_num_interop_threads = if ([string]::IsNullOrWhiteSpace($torchNumInteropThreads)) {
        $null
    }
    else {
        [int]$torchNumInteropThreads
    }
    torch_version = $torchVersion
    torch_cuda_available = if ([string]::IsNullOrWhiteSpace($torchCudaAvailable)) {
        $null
    }
    else {
        [System.Convert]::ToBoolean($torchCudaAvailable)
    }
}

$gitCommit = Invoke-Trimmed { git -C $ProjectRoot rev-parse HEAD }
if ([string]::IsNullOrWhiteSpace($gitCommit)) {
    throw 'Could not resolve the current Git commit.'
}

$gitStatus = Invoke-Trimmed { git -C $ProjectRoot status --porcelain }
$gitWorktreeDirty = -not [string]::IsNullOrWhiteSpace($gitStatus)

$dockerfilePath = Join-Path $ProjectRoot 'Dockerfile'
if ([string]::IsNullOrWhiteSpace($BaseImage) -and (Test-Path -LiteralPath $dockerfilePath -PathType Leaf)) {
    $fromLine = Get-Content -LiteralPath $dockerfilePath |
        Where-Object { $_ -match '^\s*FROM\s+' } |
        Select-Object -First 1

    if (-not [string]::IsNullOrWhiteSpace($fromLine)) {
        $BaseImage = (($fromLine -split '\s+')[1]).Trim()
    }
}

if ([string]::IsNullOrWhiteSpace($BaseImage)) {
    throw 'The Docker base image could not be resolved. Pass -BaseImage explicitly or provide a Dockerfile with a FROM line.'
}

$dockerReady = $false
$baseImageRepoDigest = $null
$baseImageDigestSource = $null
$dockerPath = Get-CommandPathOrNull -Name 'docker'

if (-not [string]::IsNullOrWhiteSpace($dockerPath)) {
    try {
        & docker info *> $null
        if ($LASTEXITCODE -eq 0) {
            $dockerReady = $true
        }
    }
    catch {
        $dockerReady = $false
    }
}

if ($dockerReady) {
    try {
        & docker pull $BaseImage *> $null
        if ($LASTEXITCODE -eq 0) {
            $candidateDigest = (
                & docker inspect --format='{{index .RepoDigests 0}}' $BaseImage 2>$null |
                    Out-String
            ).Trim()

            if ($candidateDigest -match '^.+@sha256:[0-9a-fA-F]{64}$') {
                $baseImageRepoDigest = $candidateDigest.ToLowerInvariant()
                $baseImageDigestSource = 'docker_pull_and_inspect'
            }
        }
    }
    catch {
        $baseImageRepoDigest = $null
    }
}

if ([string]::IsNullOrWhiteSpace($baseImageRepoDigest) -and $null -ne $existingManifest) {
    try {
        $existingBaseImage = [string]$existingManifest.container.dockerfile_base_image
        $existingBaseDigest = [string]$existingManifest.container.base_image_repo_digest

        if (
            $existingBaseImage -eq $BaseImage -and
            $existingBaseDigest -match '^.+@sha256:[0-9a-fA-F]{64}$'
        ) {
            $baseImageRepoDigest = $existingBaseDigest.ToLowerInvariant()
            $baseImageDigestSource = 'previous_validated_manifest'
        }
    }
    catch {
        $baseImageRepoDigest = $null
    }
}

if ([string]::IsNullOrWhiteSpace($baseImageRepoDigest)) {
    throw "Could not resolve the immutable repository digest for base image '$BaseImage'. Start Docker Desktop and rerun."
}

$finalImageDigest = $null
$finalImageDigestSource = $null

if ($FinalImageUri -match '@(sha256:[0-9a-fA-F]{64})$') {
    $finalImageDigest = $Matches[1].ToLowerInvariant()
    $finalImageDigestSource = 'pinned_uri'
}
else {
    $gcloudPath = Get-CommandPathOrNull -Name 'gcloud'
    if (-not [string]::IsNullOrWhiteSpace($gcloudPath) -and -not [string]::IsNullOrWhiteSpace($FinalImageUri)) {
        try {
            $candidateFinalDigest = (
                & gcloud artifacts docker images describe $FinalImageUri --format='value(image_summary.digest)' 2>$null |
                    Out-String
            ).Trim()

            if ($candidateFinalDigest -match '^sha256:[0-9a-fA-F]{64}$') {
                $finalImageDigest = $candidateFinalDigest.ToLowerInvariant()
                $finalImageDigestSource = 'gcloud_artifact_registry'
                $FinalImageUri = $FinalImageUri.TrimEnd('/') + '@' + $finalImageDigest
            }
        }
        catch {
            $finalImageDigest = $null
        }
    }
}

if ([string]::IsNullOrWhiteSpace($FinalImageUri)) {
    throw 'FinalImageUri is required. Pass the exact GAR image URI pinned with @sha256:<64 hex>.'
}

if ([string]::IsNullOrWhiteSpace($finalImageDigest)) {
    throw 'Could not resolve the final container image digest. Pass FinalImageUri pinned with @sha256:<64 hex>.'
}

if (-not $FinalImageUri.ToLowerInvariant().EndsWith('@' + $finalImageDigest)) {
    throw 'FinalImageUri is not pinned to the resolved final image digest.'
}

$manifest = [ordered]@{
    captured_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    git_commit = $gitCommit
    git_worktree_dirty_at_capture = $gitWorktreeDirty
    local = [ordered]@{
        os_caption = [string]$os.Caption
        os_version = [string]$os.Version
        os_build_number = [string]$os.BuildNumber
        os_architecture = [string]$os.OSArchitecture
        cpu_name = ([string]$cpu.Name).Trim()
        cpu_manufacturer = [string]$cpu.Manufacturer
        physical_cores = [int]$cpu.NumberOfCores
        logical_processors = [int]$cpu.NumberOfLogicalProcessors
        installed_ram_bytes = [int64]$computerSystem.TotalPhysicalMemory
        installed_ram_gib = [math]::Round(
            ([double]$computerSystem.TotalPhysicalMemory / 1GB),
            2
        )
        storage_devices = $physicalDisks
        active_power_scheme = $powerScheme
    }
    python = [ordered]@{
        version = $pythonVersion
        executable = $pythonExe
        executable_sha256 = $pythonHash
        requirements_txt_sha256 = $requirementsHash
        pip_freeze_lock = 'paper_outputs/environment/pip_freeze_lock.txt'
        pip_freeze_lock_sha256 = $lockHash
    }
    threads = $threadInfo
    container = [ordered]@{
        dockerfile_base_image = $BaseImage
        docker_daemon_available = $dockerReady
        base_image_repo_digest = $baseImageRepoDigest
        base_image_digest_source = $baseImageDigestSource
        final_image_uri = $FinalImageUri
        final_image_digest = $finalImageDigest
        final_image_digest_source = $finalImageDigestSource
    }
}

$manifestJson = $manifest | ConvertTo-Json -Depth 12
$manifestJson = $manifestJson.TrimEnd("`r", "`n") + [Environment]::NewLine
Write-Utf8NoBomFile -Path $manifestPath -Content $manifestJson

$manifestHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath
).Hash.ToLowerInvariant()

Write-Host ''
Write-Host '[PASS] Comment 12 execution environment captured'
Write-Host "[OUT]  $manifestPath"
Write-Host "[SHA]  $manifestHash"
Write-Host "[OUT]  $lockPath"
Write-Host "[SHA]  $lockHash"
Write-Host "[GIT]  $gitCommit"
Write-Host "[DIRTY] $gitWorktreeDirty"
Write-Host "[BASE] $baseImageRepoDigest"
Write-Host "[FINAL] $finalImageDigest"

if ($gitWorktreeDirty) {
    Write-Warning 'The working tree was dirty at capture time. Commit the collector fix, then rerun to produce a clean-commit environment manifest.'
}