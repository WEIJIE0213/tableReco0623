param(
    [string]$HostName = "10.200.97.195",
    [string]$User = "ywj",
    [int]$Port = 22,
    [string]$RemoteDir = "/home/ywj/projects/tableReco0623",
    [string]$SshExe = "C:\Windows\System32\OpenSSH\ssh.exe",
    [string]$ScpExe = "C:\Windows\System32\OpenSSH\scp.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel).Trim()
Set-Location $repoRoot

$dirty = (& git status --porcelain)
if ($dirty) {
    Write-Warning "Working tree has uncommitted changes. This script syncs committed HEAD only."
}

$head = (& git rev-parse --short HEAD).Trim()
$syncDir = Join-Path $repoRoot ".sync"
New-Item -ItemType Directory -Force -Path $syncDir | Out-Null

$archive = Join-Path $syncDir "tableReco0623-$head.tar"
if (Test-Path $archive) {
    Remove-Item -LiteralPath $archive -Force
}

Write-Host "[local] archiving HEAD $head"
& git archive --format=tar -o $archive HEAD

$remoteUploads = "~/tableReco0623_uploads"
$remoteArchive = "$remoteUploads/tableReco0623-$head.tar"
$target = "${User}@${HostName}"

Write-Host "[server] creating remote directories"
& $SshExe -p $Port -o BatchMode=yes $target "mkdir -p $remoteUploads $RemoteDir"

Write-Host "[local->server] uploading $archive"
& $ScpExe -P $Port $archive "${target}:$remoteArchive"

Write-Host "[server] extracting into $RemoteDir"
& $SshExe -p $Port -o BatchMode=yes $target "mkdir -p $RemoteDir && tar -xf $remoteArchive -C $RemoteDir && cd $RemoteDir && echo synced_commit=$head && ls -la | sed -n '1,20p'"

Write-Host "[done] server code sync complete"
