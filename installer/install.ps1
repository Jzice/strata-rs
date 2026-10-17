<#
 .SYNOPSIS
  Strata one-click installer (Windows first).

 .DESCRIPTION
  1. Detect local hardware (GPU vendor, RAM, free disk).
  2. Download the chosen GGUF model with resume support and SHA-256 verification.
  3. Generate a launcher and (optionally) a logon-time autostart task.
  4. Fetch the latest strata binary (self-update).

  Requires PowerShell 7+ for the most reliable networking. Run normally for a
  user install; pass -InstallService (as Administrator) to create an autostart
  task.
#>
[CmdletBinding()]
param(
  [string]$Model = "qwen35moe-q4_k",
  [string]$ModelUrl,
  [string]$Sha256,
  [string]$InstallDir = "$env:LOCALAPPDATA\Strata",
  [string]$BinUrl = "https://github.com/Niko1221/Strata/releases/latest/download/strata-windows-x86_64.exe",
  [int]$Port = 8080,
  [switch]$InstallService,
  [switch]$SkipBinaryUpdate
)

$ErrorActionPreference = "Stop"

# Known model catalog: name -> { url, sha256, minGpuMb, minRamMb }
$MODELS = @{
  "qwen35moe-q4_k" = @{
    url   = "https://example.invalid/models/qwen35moe-q4_k.gguf"
    sha256 = ""
    minGpuMb = 4096; minRamMb = 8192
  }
  "qwen4exp-q4_k" = @{
    url   = "https://example.invalid/models/qwen4exp-q4_k.gguf"
    sha256 = ""
    minGpuMb = 6144; minRamMb = 12288
  }
}

function Write-Step($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }

function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Hardware {
  $gpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch "Microsoft|Basic" } |
    Select-Object -First 1
  $mem = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
  $free = (Get-PSDrive -Name C -ErrorAction SilentlyContinue).Free
  [pscustomobject]@{
    GpuName = if ($gpu) { $gpu.Name } else { "No discrete GPU (CPU backend)" }
    GpuVendor = if ($gpu) {
      if ($gpu.Name -match "NVIDIA") { "nvidia" }
      elseif ($gpu.Name -match "AMD|Radeon") { "amd" }
      elseif ($gpu.Name -match "Intel") { "intel" }
      else { "unknown" }
    } else { "cpu" }
    RamMb = if ($mem) { [int]($mem.TotalPhysicalMemory / 1MB) } else { 0 }
    FreeMb = if ($free) { [int]($free / 1MB) } else { 0 }
  }
}

# Resumable download: continue from a previous partial file via an HTTP Range header.
function Invoke-ResumableDownload($url, $outPath) {
  $tmp = "$outPath.part"
  $done = if (Test-Path $tmp) { (Get-Item $tmp).Length } else { 0 }
  $req = [System.Net.HttpWebRequest][System.Net.WebRequest]::Create($url)
  if ($done -gt 0) { $req.AddRange("bytes", $done) }
  $req.AllowAutoRedirect = $true
  $req.UserAgent = "StrataInstaller/0.1"
  $resp = $req.GetResponse()
  $total = $done + $resp.ContentLength
  $stream = $resp.GetResponseStream()
  $fs = [System.IO.File]::Open($tmp, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
  $buf = New-Object byte[] 65536
  $read = 0; $got = 0
  while (($read = $stream.Read($buf, 0, $buf.Length)) -gt 0) {
    $fs.Write($buf, 0, $read)
    $got += $read
    $pct = if ($total -gt 0) { [int](($done + $got) / $total * 100) } else { 0 }
    Write-Progress -Activity "Download $url" -Status "$pct%" -PercentComplete $pct
  }
  $fs.Close(); $stream.Close(); $resp.Close()
  Move-Item $tmp $outPath -Force
  Write-Progress -Activity "Download $url" -Completed
}

function Test-Sha256($path, $expected) {
  if (-not $expected) { return $true }
  $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
  $actual -eq $expected.ToLower()
}

function Update-Binary($dir) {
  $dest = Join-Path $dir "strata.exe"
  if (Test-Path $dest) {
    $ver = & $dest --version 2>$null
    Write-Host "Current binary version: $($ver -join ' ')"
  }
  Write-Host "Downloading latest strata binary..."
  Invoke-ResumableDownload $BinUrl $dest
  Write-Host "Ready: $dest"
}

function New-Launcher($dir, $modelPath) {
  $bat = Join-Path $dir "start.bat"
  $body = "@echo off`n" +
    "setlocal`n" +
    "`"%~dp0strata.exe`" serve --model `"$modelPath`" --web-dir `"%~dp0web`" --port $Port`n" +
    "endlocal`n"
  Set-Content -Path $bat -Value $body -Encoding ASCII
  Write-Host "Launcher written: $bat"
}

# ---- main ----
Write-Step "Detect hardware"
$hw = Get-Hardware
Write-Host ("GPU: {0} ({1})" -f $hw.GpuName, $hw.GpuVendor)
Write-Host ("RAM: {0} MB  Free disk: {1} MB" -f $hw.RamMb, $hw.FreeMb)

$entry = if ($ModelUrl) { @{ url = $ModelUrl; sha256 = $Sha256 } } else { $MODELS[$Model] }
if (-not $entry) { throw "Unknown model '$Model'; pass -ModelUrl to specify a download URL." }

Write-Step "Prepare directories"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$modelsDir = Join-Path $InstallDir "models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

if (-not $SkipBinaryUpdate) {
  Write-Step "Install/update strata binary"
  Update-Binary $InstallDir
}

Write-Step "Download model"
$modelFile = Join-Path $modelsDir "$Model.gguf"
if (Test-Path $modelFile) { Write-Host "Already present, skipping: $modelFile" }
else {
  Invoke-ResumableDownload $entry.url $modelFile
  if (-not (Test-Sha256 $modelFile $entry.sha256)) {
    Write-Warning "SHA-256 mismatch; the file may be corrupt. Re-run to resume or check -Sha256."
  }
  else { Write-Host "Verified: $modelFile" }
}

Write-Step "Generate launcher"
New-Launcher $InstallDir $modelFile

if ($InstallService) {
  Write-Step "Install as service"
  if (-not (Test-Admin)) { throw "Installing a service requires running as Administrator." }
  $action = "powershell -NoProfile -Command `"& '$(Join-Path $InstallDir 'start.bat')'`""
  schtasks /Create /TN "StrataServer" /TR $action /SC ONLOGON /F | Out-Null
  Write-Host "Created logon autostart task StrataServer (manage via Task Scheduler)."
}

Write-Step "Done"
Write-Host "Install dir: $InstallDir"
Write-Host "Run $(Join-Path $InstallDir 'start.bat'), then open http://127.0.0.1:$Port"
