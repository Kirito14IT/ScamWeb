param(
    [string]$Repository = "nlixsys/ScamWeb-V",
    [string]$Tag = "experiment-qwen3-dashscope-full-v1",
    [string]$Destination = "outputs/server_results/qwen3-dashscope-full-v1"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DownloadDir = Join-Path $Root "dist\result-downloads"
$DestinationPath = Join-Path $Root $Destination
New-Item -ItemType Directory -Force -Path $DownloadDir, $DestinationPath | Out-Null

gh auth status | Out-Null
gh release download $Tag --repo $Repository --dir $DownloadDir --clobber `
    --pattern "scammark-qwen3-dashscope-full-v1-inference.tar.gz" `
    --pattern "scammark-qwen3-dashscope-full-v1-inference.tar.gz.sha256" `
    --pattern "scammark-qwen3-dashscope-full-v1-final-evaluation.tar.gz" `
    --pattern "scammark-qwen3-dashscope-full-v1-final-evaluation.tar.gz.sha256"
if ($LASTEXITCODE -ne 0) {
    throw "GitHub Release download failed"
}

$Archives = Get-ChildItem -LiteralPath $DownloadDir -Filter "*.tar.gz"
foreach ($Archive in $Archives) {
    $ChecksumFile = "$($Archive.FullName).sha256"
    if (-not (Test-Path -LiteralPath $ChecksumFile)) {
        throw "Missing checksum for $($Archive.Name)"
    }
    $Expected = ((Get-Content -LiteralPath $ChecksumFile -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive.FullName).Hash.ToLowerInvariant()
    if ($Expected -ne $Actual) {
        throw "SHA256 mismatch for $($Archive.Name)"
    }
    tar -xzf $Archive.FullName -C $DestinationPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract $($Archive.Name)"
    }
}

$FinalMetricsFiles = @(Get-ChildItem -LiteralPath $DestinationPath -Recurse -Filter "metrics_final.json")
if ($FinalMetricsFiles.Count -ne 1) {
    throw "Expected exactly one metrics_final.json, found $($FinalMetricsFiles.Count)"
}
$FinalMetrics = Get-Content -LiteralPath $FinalMetricsFiles[0].FullName -Raw | ConvertFrom-Json
$PredictionCount = $FinalMetrics.engineering.prediction_sample_count
$BertScoreStatus = $FinalMetrics.rationale.end_to_end.bertscore_status
if ($PredictionCount -ne 5658) {
    throw "Expected prediction_sample_count=5658, found $PredictionCount"
}
if ($BertScoreStatus -ne "computed") {
    throw "Expected bertscore_status=computed, found $BertScoreStatus"
}

git -C $Root fetch origin "results/qwen3-dashscope-full-v1:results/qwen3-dashscope-full-v1"

[pscustomobject]@{
    destination = $DestinationPath
    release = "https://github.com/$Repository/releases/tag/$Tag"
    result_branch = "results/qwen3-dashscope-full-v1"
    metrics_final = $FinalMetricsFiles[0].FullName
    prediction_sample_count = $PredictionCount
    bertscore_status = $BertScoreStatus
} | ConvertTo-Json
