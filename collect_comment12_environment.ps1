param(
    [string]$BaseImage = '',
    [string]$FinalImageUri = ''
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = (Get-Location).Path
$OutDir = Join-Path $ProjectRoot 'paper_outputs\environment'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Get-CommandPathOrNull {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { return $null }
    return $cmd.Source
}

function Invoke-Trimmed {
    param([scriptblock]$Command)
    try {
        return ((& $Command 2>$null | Out-String).Trim())
    }
    catch {
        return $null
    }
}

$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1

try {
    $physicalDisks = @(Get-PhysicalDisk | ForEach-Object {
        [ordered]@{
            friendly_name = $_.FriendlyName
            media_type = [string]$_.MediaType
            bus_type = [string]$_.BusType
            size_bytes = [int64]$_.Size
        }
    })
}
catch {
    $physicalDisks = @(Get-CimInstance Win32_DiskDrive | ForEach-Object {
        [ordered]@{
            friendly_name = $_.Model
            media_type = $_.MediaType
            bus_type = $_.InterfaceType
            size_bytes = [int64]$_.Size
        }
    })
}

$powerScheme = (& powercfg /getactivescheme | Out-String).Trim()

$pythonExe = (Get-Command python).Source
$pythonHash = (Get-FileHash -Algorithm SHA256 -Path $pythonExe).Hash.ToLowerInvariant()
$pythonVersion = (& python --version 2>&1 | Out-String).Trim()

$lockPath = Join-Path $OutDir 'pip_freeze_lock.txt'
$pipFreeze = (& python -m pip freeze | Out-String).TrimEnd() + [Environment]::NewLine
[System.IO.File]::WriteAllText($lockPath, $pipFreeze, $Utf8NoBom)
$lockHash = (Get-FileHash -Algorithm SHA256 -Path $lockPath).Hash.ToLowerInvariant()

$requirementsPath = Join-Path $ProjectRoot 'requirements.txt'
$requirementsHash = $null
if (Test-Path $requirementsPath) {
    $requirementsHash = (Get-FileHash -Algorithm SHA256 -Path $requirementsPath).Hash.ToLowerInvariant()
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
    torch_num_threads = if ($torchNumThreads) { [int]$torchNumThreads } else { $null }
    torch_num_interop_threads = if ($torchNumInteropThreads) { [int]$torchNumInteropThreads } else { $null }
    torch_version = $torchVersion
    torch_cuda_available = if ($torchCudaAvailable) {
        [System.Convert]::ToBoolean($torchCudaAvailable)
    } else {
        $null
    }
}

$gitCommit = Invoke-Trimmed { git rev-parse HEAD }

if ([string]::IsNullOrWhiteSpace($BaseImage) -and (Test-Path (Join-Path $ProjectRoot 'Dockerfile'))) {
    $fromLine = Get-Content (Join-Path $ProjectRoot 'Dockerfile') |
        Where-Object { $_ -match '^\s*FROM\s+' } |
        Select-Object -First 1
    if ($fromLine) {
        $BaseImage = (($fromLine -split '\s+')[1]).Trim()
    }
}

$dockerReady = $false
$baseImageRepoDigest = $null
$dockerPath = Get-CommandPathOrNull -Name 'docker'

if ($dockerPath) {
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

if ($dockerReady -and -not [string]::IsNullOrWhiteSpace($BaseImage)) {
    try {
        & docker pull $BaseImage *> $null
        if ($LASTEXITCODE -eq 0) {
            $baseImageRepoDigest = (& docker inspect --format='{{index .RepoDigests 0}}' $BaseImage 2>$null | Out-String).Trim()
            if ([string]::IsNullOrWhiteSpace($baseImageRepoDigest)) {
                $baseImageRepoDigest = $null
            }
        }
    }
    catch {
        $baseImageRepoDigest = $null
    }
}

$finalImageDigest = $null
$gcloudPath = Get-CommandPathOrNull -Name 'gcloud'
if ($gcloudPath -and -not [string]::IsNullOrWhiteSpace($FinalImageUri)) {
    try {
        $finalImageDigest = (& gcloud artifacts docker images describe $FinalImageUri --format='value(image_summary.digest)' 2>$null | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($finalImageDigest)) {
            $finalImageDigest = $null
        }
    }
    catch {
        $finalImageDigest = $null
    }
}

$manifest = [ordered]@{
    captured_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    git_commit = $gitCommit
    local = [ordered]@{
        os_caption = $os.Caption
        os_version = $os.Version
        os_build_number = $os.BuildNumber
        os_architecture = $os.OSArchitecture
        cpu_name = $cpu.Name.Trim()
        cpu_manufacturer = $cpu.Manufacturer
        physical_cores = [int]$cpu.NumberOfCores
        logical_processors = [int]$cpu.NumberOfLogicalProcessors
        installed_ram_bytes = [int64]$cs.TotalPhysicalMemory
        installed_ram_gib = [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
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
        final_image_uri = $FinalImageUri
        final_image_digest = $finalImageDigest
    }
}

$manifestPath = Join-Path $OutDir 'execution_environment_manifest.json'
$manifestJson = $manifest | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText(
    $manifestPath,
    $manifestJson + [Environment]::NewLine,
    $Utf8NoBom
)

$manifestHash = (Get-FileHash -Algorithm SHA256 -Path $manifestPath).Hash.ToLowerInvariant()

Write-Host "Created: $manifestPath"
Write-Host "Manifest SHA-256: $manifestHash"
Write-Host "Created: $lockPath"
Write-Host "Lock SHA-256: $lockHash"

if (-not $dockerReady) {
    Write-Warning 'Docker Desktop is not running. Local fields were captured, but the base-image digest remains unresolved.'
}
elseif ($null -eq $baseImageRepoDigest) {
    Write-Warning 'Docker is available, but the base-image digest was not resolved.'
}

if ($null -eq $finalImageDigest) {
    Write-Warning "Final GAR image digest was not resolved. Rerun with -FinalImageUri '<exact GAR image URI:tag-or-digest>'."
}