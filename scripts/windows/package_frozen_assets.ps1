param(
    [string]$Source = "outputs/dataset_frozen_v1",
    [string]$Destination = "dist/release-assets"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SourcePath = (Resolve-Path (Join-Path $Root $Source)).Path
$DestinationPath = Join-Path $Root $Destination
$ArchiveName = "scamweb-v1.0-frozen-inputs.tar.gz"
$ArchivePath = Join-Path $DestinationPath $ArchiveName
$ChecksumPath = "$ArchivePath.sha256"
$SourceName = Split-Path -Leaf $SourcePath
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$MemberFiles = @(
    "boxes.jsonl",
    "dataset_version.json",
    "mhtml_audit.jsonl",
    "mhtml_recapture.jsonl",
    "sample_ids.txt",
    "scamweb_v_manifest.jsonl",
    "verification_report.json",
    "web_evidence.jsonl"
)

New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
$env:PYTHONPATH = Join-Path $Root "src"
& $Python -m scammark.frozen_assets --root $SourcePath
if ($LASTEXITCODE -ne 0) {
    throw "Frozen asset verification failed"
}

if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
if (Test-Path -LiteralPath $ChecksumPath) {
    Remove-Item -LiteralPath $ChecksumPath -Force
}

@'
import gzip
import sys
import tarfile
from pathlib import Path

source = Path(sys.argv[1])
archive = Path(sys.argv[2])
source_name = sys.argv[3]
members = sys.argv[4:]

with archive.open("wb") as raw_output:
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=raw_output,
        mtime=0,
    ) as compressed:
        with tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as bundle:
            for filename in members:
                path = source / filename
                info = tarfile.TarInfo(f"{source_name}/{filename}")
                info.size = path.stat().st_size
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o644
                with path.open("rb") as handle:
                    bundle.addfile(info, handle)
'@ | & $Python - $SourcePath $ArchivePath $SourceName @MemberFiles
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create frozen input archive"
}
$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash.ToLowerInvariant()
"$Hash  $ArchiveName" | Set-Content -Encoding ascii -NoNewline -LiteralPath $ChecksumPath

[pscustomobject]@{
    archive = $ArchivePath
    checksum = $ChecksumPath
    sha256 = $Hash
    bytes = (Get-Item -LiteralPath $ArchivePath).Length
} | ConvertTo-Json
