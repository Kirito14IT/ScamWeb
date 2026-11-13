param(
    [string]$Repository = "nlixsys/ScamWeb-V",
    [string]$Tag = "experiment-qwen3-dashscope-full-v2",
    [string]$Destination = "outputs/server_results/qwen3-dashscope-full-v2",
    [string]$ArchiveDirectory = "",
    [switch]$VerifyOnly,
    [switch]$SkipResultsBranchFetch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$AllowedDestinationRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $Root "outputs/server_results")
)
$PathComparison = if (
    [System.IO.Path]::DirectorySeparatorChar -eq [char]92
) {
    [System.StringComparison]::OrdinalIgnoreCase
}
else {
    [System.StringComparison]::Ordinal
}
$ResultsBranch = "results/qwen3-dashscope-full-v2"
$InferenceArchiveName = "scammark-qwen3-dashscope-full-v2-inference.tar.gz"
$FinalArchiveName = "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz"
$AssetNames = @(
    "scammark-qwen3-dashscope-full-v2-inference.tar.gz",
    "scammark-qwen3-dashscope-full-v2-inference.tar.gz.sha256",
    "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz",
    "scammark-qwen3-dashscope-full-v2-final-evaluation.tar.gz.sha256"
)
$InferenceRoot = "scammark-qwen3-dashscope-full-v2-inference"
$FinalRoot = "scammark-qwen3-dashscope-full-v2-final-evaluation"
$InferencePayload = @(
    "full/predictions.jsonl",
    "full/metrics.json",
    "full/manifest.jsonl",
    "full/preflight.json",
    "full/full_summary.md",
    "full/human_evaluation_candidates.csv",
    "audit/run_config.sanitized.json",
    "audit/api_call_ledger.sanitized.jsonl",
    "logs/run_events.jsonl",
    "logs/full_run.log",
    "config/qwen3_dashscope_server_full_v2.toml",
    "frozen/dataset_version.json",
    "frozen/verification_report.json",
    "provenance.json"
)
$FinalPayload = @(
    "full/metrics.json",
    "full/metrics_final.json",
    "full/final_evaluation_summary.md",
    "environment/bertscore_environment.txt",
    "environment/bertscore_asset_status.json",
    "logs/bertscore_finalize.log",
    "provenance.json"
)
$Utf8Strict = [System.Text.UTF8Encoding]::new($false, $true)
$ConvertFromJsonSupportsDateKind = (
    Get-Command ConvertFrom-Json
).Parameters.ContainsKey("DateKind")

function ConvertFrom-JsonPreservingStrings {
    param([Parameter(Mandatory = $true)][string]$Text)

    if ($ConvertFromJsonSupportsDateKind) {
        return $Text | ConvertFrom-Json -Depth 100 -DateKind String
    }
    return $Text | ConvertFrom-Json -Depth 100
}

function Get-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ([System.IO.Path]::IsPathFullyQualified($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Value))
}

function Assert-SafeDestinationPath {
    param([Parameter(Mandatory = $true)][string]$DestinationPath)

    $AllowedVolumeRoot = [System.IO.Path]::GetPathRoot(
        $AllowedDestinationRoot
    )
    $DestinationVolumeRoot = [System.IO.Path]::GetPathRoot($DestinationPath)
    if (
        [string]::IsNullOrEmpty($AllowedVolumeRoot) -or
        [string]::IsNullOrEmpty($DestinationVolumeRoot) -or
        -not [string]::Equals(
            $AllowedVolumeRoot,
            $DestinationVolumeRoot,
            $PathComparison
        )
    ) {
        throw "Destination must use the controlled results volume"
    }

    $RelativeDestination = [System.IO.Path]::GetRelativePath(
        $AllowedDestinationRoot,
        $DestinationPath
    )
    $ParentPrefix = "..$([System.IO.Path]::DirectorySeparatorChar)"
    $AlternateParentPrefix = "..$([System.IO.Path]::AltDirectorySeparatorChar)"
    if (
        $RelativeDestination -eq "." -or
        $RelativeDestination -eq ".." -or
        [System.IO.Path]::IsPathFullyQualified($RelativeDestination) -or
        $RelativeDestination.StartsWith($ParentPrefix, $PathComparison) -or
        $RelativeDestination.StartsWith($AlternateParentPrefix, $PathComparison)
    ) {
        throw "Destination must be a non-root child of outputs/server_results"
    }

    $CurrentPath = $DestinationPath
    while ($true) {
        $Item = Get-Item `
            -LiteralPath $CurrentPath `
            -Force `
            -ErrorAction SilentlyContinue
        if ($null -ne $Item) {
            if (
                ($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) `
                    -ne 0
            ) {
                throw "Destination ancestor is a reparse point"
            }
            if (-not $Item.PSIsContainer) {
                throw "Destination ancestor is not a directory"
            }
        }
        if ([string]::Equals($CurrentPath, $Root, $PathComparison)) {
            break
        }
        $Parent = [System.IO.Directory]::GetParent($CurrentPath)
        if ($null -eq $Parent) {
            throw "Destination does not resolve below the repository root"
        }
        $CurrentPath = $Parent.FullName
    }
}

function Assert-RegularFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing"
    }
    $Item = Get-Item -LiteralPath $Path -Force
    if (
        $Item.PSIsContainer -or
        (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)
    ) {
        throw "$Label is not a regular non-reparse file"
    }
}

function Assert-ExactAssetDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "ArchiveDirectory is missing"
    }
    $DirectoryItem = Get-Item -LiteralPath $Path -Force
    if (($DirectoryItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "ArchiveDirectory must not be a reparse point"
    }
    $Entries = @(Get-ChildItem -LiteralPath $Path -Force)
    if ($Entries.Count -ne $AssetNames.Count) {
        throw "ArchiveDirectory must contain exactly two archives and two checksums"
    }
    $Expected = @($AssetNames | Sort-Object -CaseSensitive)
    $Actual = @($Entries.Name | Sort-Object -CaseSensitive)
    if ([string]::Join("`n", $Actual) -cne [string]::Join("`n", $Expected)) {
        throw "ArchiveDirectory asset names are not exact"
    }
    foreach ($Name in $AssetNames) {
        Assert-RegularFile -Path (Join-Path $Path $Name) -Label "Release asset"
    }
}

function Copy-PrivateAssetSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDirectory,
        [Parameter(Mandatory = $true)][string]$SnapshotDirectory
    )

    $null = New-Item -ItemType Directory -Path $SnapshotDirectory
    foreach ($Name in $AssetNames) {
        $SourcePath = Join-Path $SourceDirectory $Name
        $SnapshotPath = Join-Path $SnapshotDirectory $Name
        Assert-RegularFile -Path $SourcePath -Label "Release asset source"
        [System.IO.File]::Copy($SourcePath, $SnapshotPath, $false)
        Assert-RegularFile -Path $SnapshotPath -Label "Private release asset snapshot"
    }
    Assert-ExactAssetDirectory -Path $SnapshotDirectory
    return $SnapshotDirectory
}

function Assert-ExactChecksum {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$ArchiveName
    )

    $ChecksumPath = "$ArchivePath.sha256"
    Assert-RegularFile -Path $ArchivePath -Label "Result archive"
    Assert-RegularFile -Path $ChecksumPath -Label "Result checksum"
    $ChecksumBytes = [System.IO.File]::ReadAllBytes($ChecksumPath)
    try {
        $ChecksumText = $Utf8Strict.GetString($ChecksumBytes)
    }
    catch {
        throw "Result checksum is not strict UTF-8"
    }
    $Pattern = "\A([0-9a-f]{64})  $([regex]::Escape($ArchiveName))`n\z"
    $Match = [regex]::Match($ChecksumText, $Pattern)
    if (-not $Match.Success) {
        throw "Result checksum content is not exact"
    }
    $ExpectedHash = $Match.Groups[1].Value
    $ExpectedBytes = [System.Text.Encoding]::ASCII.GetBytes(
        "$ExpectedHash  $ArchiveName`n"
    )
    $BytesMatch = $ChecksumBytes.Length -eq $ExpectedBytes.Length
    if ($BytesMatch) {
        for ($Index = 0; $Index -lt $ChecksumBytes.Length; $Index += 1) {
            if ($ChecksumBytes[$Index] -ne $ExpectedBytes[$Index]) {
                $BytesMatch = $false
                break
            }
        }
    }
    if (-not $BytesMatch) {
        throw "Result checksum bytes are not exact"
    }
    $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash.ToLowerInvariant()
    if ($ActualHash -cne $ExpectedHash) {
        throw "Result archive SHA256 mismatch"
    }
    return $ExpectedHash
}

function Assert-SafeArchiveMember {
    param(
        [Parameter(Mandatory = $true)][string]$Member,
        [Parameter(Mandatory = $true)][string]$ExpectedRoot
    )

    if (
        [string]::IsNullOrWhiteSpace($Member) -or
        $Member.StartsWith("/", [System.StringComparison]::Ordinal) -or
        $Member.StartsWith("\", [System.StringComparison]::Ordinal) -or
        $Member.Contains("\") -or
        $Member -match "\A[A-Za-z]:"
    ) {
        throw "Result archive contains an absolute or malformed member"
    }
    $Parts = $Member.Split("/")
    if ($Parts -contains ".." -or $Parts -contains "." -or $Parts -contains "") {
        throw "Result archive contains path traversal"
    }
    if ($Parts[0] -cne $ExpectedRoot) {
        throw "Result archive root is not exact"
    }
    $DenylistedSegments = @(
        ".env", "cache", "caches", "checkpoint", "checkpoints",
        "image_cache", "model_weights", "prompt", "prompts", "raw",
        "raw_request", "raw_response", "request", "response", "secrets"
    )
    foreach ($Part in $Parts) {
        if ($DenylistedSegments -ccontains $Part.ToLowerInvariant()) {
            throw "Result archive contains a denylisted member"
        }
    }
}

function Assert-ArchiveMembers {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$ExpectedRoot,
        [Parameter(Mandatory = $true)][string[]]$Payload
    )

    $Members = @(& tar -tzf $ArchivePath 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list result archive members"
    }
    $VerboseMembers = @(& tar -tvzf $ArchivePath 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect result archive member types"
    }
    if ($Members.Count -ne $Payload.Count -or $VerboseMembers.Count -ne $Payload.Count) {
        throw "Result archive member count is not exact"
    }
    foreach ($Line in $VerboseMembers) {
        if ([string]::IsNullOrEmpty($Line) -or $Line[0] -cne "-") {
            throw "Result archive contains a non-regular member type"
        }
    }
    $ExpectedMembers = @($Payload | ForEach-Object { "$ExpectedRoot/$_" })
    $UniqueMembers = @($Members | Sort-Object -CaseSensitive -Unique)
    if ($UniqueMembers.Count -ne $Members.Count) {
        throw "Result archive contains duplicate member names"
    }
    foreach ($Member in $Members) {
        Assert-SafeArchiveMember -Member $Member -ExpectedRoot $ExpectedRoot
    }
    $ExpectedSorted = @($ExpectedMembers | Sort-Object -CaseSensitive)
    $ActualSorted = @($Members | Sort-Object -CaseSensitive)
    if ([string]::Join("`n", $ActualSorted) -cne [string]::Join("`n", $ExpectedSorted)) {
        throw "Result archive member allowlist is not exact"
    }
}

function Expand-ValidatedArchive {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    & tar -xzf $ArchivePath -C $DestinationPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to extract validated result archive"
    }
}

function Assert-ExtractedTree {
    param(
        [Parameter(Mandatory = $true)][string]$ExtractedRoot,
        [Parameter(Mandatory = $true)][string[]]$Payload
    )

    if (-not (Test-Path -LiteralPath $ExtractedRoot -PathType Container)) {
        throw "Extracted result root is missing"
    }
    $RootItem = Get-Item -LiteralPath $ExtractedRoot -Force
    if (($RootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Extracted result root is a reparse point"
    }
    $Files = @()
    foreach ($Item in Get-ChildItem -LiteralPath $ExtractedRoot -Recurse -Force) {
        if (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Extracted result contains a reparse point"
        }
        if (-not $Item.PSIsContainer) {
            $Relative = [System.IO.Path]::GetRelativePath(
                $ExtractedRoot,
                $Item.FullName
            ).Replace("\", "/")
            $Files += $Relative
        }
    }
    $ExpectedSorted = @($Payload | Sort-Object -CaseSensitive)
    $ActualSorted = @($Files | Sort-Object -CaseSensitive)
    if ([string]::Join("`n", $ActualSorted) -cne [string]::Join("`n", $ExpectedSorted)) {
        throw "Extracted result file allowlist is not exact"
    }
}

function Assert-UniqueJsonElement {
    param(
        [Parameter(Mandatory = $true)][System.Text.Json.JsonElement]$Element,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if ($Element.ValueKind -eq [System.Text.Json.JsonValueKind]::Object) {
        $Names = [System.Collections.Generic.HashSet[string]]::new(
            [System.StringComparer]::Ordinal
        )
        foreach ($Property in $Element.EnumerateObject()) {
            if (-not $Names.Add($Property.Name)) {
                throw "$Label contains a duplicate JSON property"
            }
            Assert-UniqueJsonElement -Element $Property.Value -Label $Label
        }
    }
    elseif ($Element.ValueKind -eq [System.Text.Json.JsonValueKind]::Array) {
        foreach ($Item in $Element.EnumerateArray()) {
            Assert-UniqueJsonElement -Element $Item -Label $Label
        }
    }
}

function Assert-StrictJsonObjectText {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $Options = [System.Text.Json.JsonDocumentOptions]::new()
    $Options.AllowTrailingCommas = $false
    $Options.CommentHandling = [System.Text.Json.JsonCommentHandling]::Disallow
    $Options.MaxDepth = 100
    $Document = $null
    try {
        $Document = [System.Text.Json.JsonDocument]::Parse($Text, $Options)
        if ($Document.RootElement.ValueKind -ne [System.Text.Json.JsonValueKind]::Object) {
            throw "$Label must be a JSON object"
        }
        Assert-UniqueJsonElement -Element $Document.RootElement -Label $Label
    }
    catch {
        if ($_.Exception.Message -like "$Label*") {
            throw
        }
        throw "$Label is not valid strict JSON"
    }
    finally {
        if ($null -ne $Document) {
            $Document.Dispose()
        }
    }
}

function Read-JsonObject {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    Assert-RegularFile -Path $Path -Label $Label
    try {
        $Text = [System.IO.File]::ReadAllText($Path, $Utf8Strict)
        Assert-StrictJsonObjectText -Text $Text -Label $Label
        $Value = ConvertFrom-JsonPreservingStrings -Text $Text
    }
    catch {
        throw "$Label is not valid strict JSON"
    }
    if ($null -eq $Value -or $Value -is [array] -or $Value -isnot [pscustomobject]) {
        throw "$Label must be a JSON object"
    }
    return $Value
}

function Get-ExactProperty {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $Names = @($Object.PSObject.Properties.Name)
    if (-not ($Names -ccontains $Name)) {
        throw "$Label is missing $Name"
    }
    return $Object.PSObject.Properties[$Name].Value
}

function Assert-ExactInteger {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][int64]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (
        ($Value -isnot [int] -and $Value -isnot [int64]) -or
        [int64]$Value -ne $Expected
    ) {
        throw "$Label is not $Expected"
    }
}

function Read-SampleIds {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    Assert-RegularFile -Path $Path -Label $Label
    $Ids = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal
    )
    $OrderedIds = [System.Collections.Generic.List[string]]::new()
    $Count = 0
    try {
        foreach ($Line in [System.IO.File]::ReadLines($Path, $Utf8Strict)) {
            if ([string]::IsNullOrWhiteSpace($Line)) {
                throw "$Label contains an empty row"
            }
            Assert-StrictJsonObjectText -Text $Line -Label $Label
            $Row = ConvertFrom-JsonPreservingStrings -Text $Line
            if ($null -eq $Row -or $Row -is [array] -or $Row -isnot [pscustomobject]) {
                throw "$Label row is not an object"
            }
            $SampleId = Get-ExactProperty -Object $Row -Name "sample_id" -Label $Label
            if (
                $SampleId -isnot [string] -or
                [string]::IsNullOrWhiteSpace($SampleId) -or
                -not [string]::Equals(
                    $SampleId,
                    $SampleId.Trim(),
                    [System.StringComparison]::Ordinal
                )
            ) {
                throw "$Label contains an invalid sample_id"
            }
            if (-not $Ids.Add($SampleId)) {
                throw "$Label contains a duplicate sample_id"
            }
            $OrderedIds.Add($SampleId)
            $Count += 1
        }
    }
    catch {
        if ($_.Exception.Message -like "$Label*") {
            throw
        }
        throw "$Label is not valid strict JSONL"
    }
    if ($Count -ne 5658) {
        throw "$Label must contain exactly 5658 unique sample IDs"
    }
    return [pscustomobject]@{
        Count = $Count
        Ids = $Ids
        OrderedIds = $OrderedIds
    }
}

function Assert-MetricsContract {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $Metrics = Read-JsonObject -Path $Path -Label $Label
    $Engineering = Get-ExactProperty -Object $Metrics -Name "engineering" -Label $Label
    if ($Engineering -isnot [pscustomobject]) {
        throw "$Label engineering contract is invalid"
    }
    Assert-ExactInteger `
        -Value (Get-ExactProperty -Object $Engineering -Name "sample_count" -Label $Label) `
        -Expected 5658 `
        -Label "$Label sample_count"
    Assert-ExactInteger `
        -Value (Get-ExactProperty -Object $Engineering -Name "prediction_sample_count" -Label $Label) `
        -Expected 5658 `
        -Label "$Label prediction_sample_count"
    $Gates = Get-ExactProperty -Object $Engineering -Name "acceptance_gates" -Label $Label
    if ($Gates -isnot [pscustomobject]) {
        throw "$Label acceptance_gates contract is invalid"
    }
    $Passed = Get-ExactProperty -Object $Gates -Name "passed" -Label $Label
    if ($Passed -isnot [bool] -or $Passed -ne $true) {
        throw "$Label acceptance_gates.passed must be true"
    }
    $Checks = Get-ExactProperty -Object $Gates -Name "checks" -Label $Label
    if ($Checks -isnot [pscustomobject] -or @($Checks.PSObject.Properties).Count -eq 0) {
        throw "$Label acceptance_gates.checks is invalid"
    }
    $FullCountGate = Get-ExactProperty `
        -Object $Checks `
        -Name "full_sample_count_exact" `
        -Label $Label
    if ($FullCountGate -isnot [bool] -or $FullCountGate -ne $true) {
        throw "$Label full_sample_count_exact must be true"
    }
    foreach ($Property in $Checks.PSObject.Properties) {
        if ($Property.Value -isnot [bool] -or $Property.Value -ne $true) {
            throw "$Label contains a failed acceptance gate"
        }
    }
    return $Metrics
}

function Assert-ProvenanceContract {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Mode
    )

    $Provenance = Read-JsonObject -Path $Path -Label "provenance.json"
    $ExpectedFields = @(
        "schema", "mode", "repository", "release_tag", "git_commit",
        "packaged_at_utc", "secrets_included"
    )
    $ActualFields = @($Provenance.PSObject.Properties.Name | Sort-Object -CaseSensitive)
    $ExpectedSorted = @($ExpectedFields | Sort-Object -CaseSensitive)
    if ([string]::Join("`n", $ActualFields) -cne [string]::Join("`n", $ExpectedSorted)) {
        throw "provenance.json fields are not exact"
    }
    if ((Get-ExactProperty -Object $Provenance -Name "schema" -Label "provenance.json") -cne "scammark-result-provenance-v2") {
        throw "provenance.json schema is invalid"
    }
    if ((Get-ExactProperty -Object $Provenance -Name "mode" -Label "provenance.json") -cne $Mode) {
        throw "provenance.json mode is invalid"
    }
    if ((Get-ExactProperty -Object $Provenance -Name "repository" -Label "provenance.json") -cne $Repository) {
        throw "provenance.json repository is invalid"
    }
    if ((Get-ExactProperty -Object $Provenance -Name "release_tag" -Label "provenance.json") -cne $Tag) {
        throw "provenance.json release tag is invalid"
    }
    $Commit = Get-ExactProperty -Object $Provenance -Name "git_commit" -Label "provenance.json"
    if ($Commit -isnot [string] -or $Commit -cnotmatch "\A(?:[0-9a-f]{6,40}|[0-9a-f]{64})\z") {
        throw "provenance.json git commit is invalid"
    }
    $PackagedAt = Get-ExactProperty -Object $Provenance -Name "packaged_at_utc" -Label "provenance.json"
    $ParsedTimestamp = [datetimeoffset]::MinValue
    if ($PackagedAt -isnot [string] -or -not [datetimeoffset]::TryParse($PackagedAt, [ref]$ParsedTimestamp)) {
        throw "provenance.json packaged timestamp is invalid"
    }
    $SecretsIncluded = Get-ExactProperty -Object $Provenance -Name "secrets_included" -Label "provenance.json"
    if ($SecretsIncluded -isnot [bool] -or $SecretsIncluded -ne $false) {
        throw "provenance.json secrets flag is invalid"
    }
    return $Provenance
}

function Assert-ResultContent {
    param([Parameter(Mandatory = $true)][string]$StagedDestination)

    $InferencePath = Join-Path $StagedDestination $InferenceRoot
    $FinalPath = Join-Path $StagedDestination $FinalRoot
    Assert-ExtractedTree -ExtractedRoot $InferencePath -Payload $InferencePayload
    Assert-ExtractedTree -ExtractedRoot $FinalPath -Payload $FinalPayload

    $RunEvents = @(Get-ChildItem -LiteralPath $StagedDestination -Recurse -Force -File -Filter "run_events.jsonl")
    $RunLogs = @(Get-ChildItem -LiteralPath $StagedDestination -Recurse -Force -File -Filter "full_run.log")
    $ProvenanceFiles = @(Get-ChildItem -LiteralPath $StagedDestination -Recurse -Force -File -Filter "provenance.json")
    if ($RunEvents.Count -ne 1 -or $RunLogs.Count -ne 1 -or $ProvenanceFiles.Count -ne 2) {
        throw "Result event, log, or provenance file count is not exact"
    }

    $Manifest = Read-SampleIds `
        -Path (Join-Path $InferencePath "full/manifest.jsonl") `
        -Label "manifest.jsonl"
    $Predictions = Read-SampleIds `
        -Path (Join-Path $InferencePath "full/predictions.jsonl") `
        -Label "predictions.jsonl"
    if ($Manifest.Count -ne $Predictions.Count) {
        throw "Manifest and prediction counts differ"
    }
    foreach ($SampleId in $Manifest.Ids) {
        if (-not $Predictions.Ids.Contains($SampleId)) {
            throw "Manifest and prediction sample IDs differ"
        }
    }
    for ($Index = 0; $Index -lt $Manifest.OrderedIds.Count; $Index += 1) {
        if (-not [string]::Equals(
            $Manifest.OrderedIds[$Index],
            $Predictions.OrderedIds[$Index],
            [System.StringComparison]::Ordinal
        )) {
            throw "Manifest and prediction sample ID order differs"
        }
    }

    $InferenceMetricsPath = Join-Path $InferencePath "full/metrics.json"
    $FinalMetricsPath = Join-Path $FinalPath "full/metrics.json"
    $null = Assert-MetricsContract -Path $InferenceMetricsPath -Label "inference metrics.json"
    $null = Assert-MetricsContract -Path $FinalMetricsPath -Label "final metrics.json"
    $InferenceMetricsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $InferenceMetricsPath).Hash
    $FinalMetricsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $FinalMetricsPath).Hash
    if ($InferenceMetricsHash -cne $FinalMetricsHash) {
        throw "Inference and final metrics.json bytes differ"
    }

    $FinalMetrics = Read-JsonObject `
        -Path (Join-Path $FinalPath "full/metrics_final.json") `
        -Label "metrics_final.json"
    $FinalEngineering = Get-ExactProperty -Object $FinalMetrics -Name "engineering" -Label "metrics_final.json"
    if ($FinalEngineering -isnot [pscustomobject]) {
        throw "metrics_final.json engineering contract is invalid"
    }
    Assert-ExactInteger `
        -Value (Get-ExactProperty -Object $FinalEngineering -Name "prediction_sample_count" -Label "metrics_final.json") `
        -Expected 5658 `
        -Label "metrics_final.json prediction_sample_count"
    $Rationale = Get-ExactProperty -Object $FinalMetrics -Name "rationale" -Label "metrics_final.json"
    if ($Rationale -isnot [pscustomobject]) {
        throw "metrics_final.json rationale contract is invalid"
    }
    $EndToEnd = Get-ExactProperty -Object $Rationale -Name "end_to_end" -Label "metrics_final.json"
    if ($EndToEnd -isnot [pscustomobject]) {
        throw "metrics_final.json end_to_end contract is invalid"
    }
    if ((Get-ExactProperty -Object $EndToEnd -Name "bertscore_status" -Label "metrics_final.json") -cne "computed") {
        throw "metrics_final.json bertscore_status must be computed"
    }

    $InferenceProvenance = Assert-ProvenanceContract `
        -Path (Join-Path $InferencePath "provenance.json") `
        -Mode "inference"
    $FinalProvenance = Assert-ProvenanceContract `
        -Path (Join-Path $FinalPath "provenance.json") `
        -Mode "final"
    foreach ($Name in @("repository", "release_tag", "git_commit")) {
        $InferenceValue = Get-ExactProperty -Object $InferenceProvenance -Name $Name -Label "inference provenance"
        $FinalValue = Get-ExactProperty -Object $FinalProvenance -Name $Name -Label "final provenance"
        if ($InferenceValue -cne $FinalValue) {
            throw "Inference and final provenance differ"
        }
    }
}

function Install-StagedDestination {
    param(
        [Parameter(Mandatory = $true)][string]$StagedDestination,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    Assert-SafeDestinationPath -DestinationPath $DestinationPath
    $Parent = Split-Path -Parent $DestinationPath
    $Leaf = Split-Path -Leaf $DestinationPath
    $Backup = Join-Path $Parent ".$Leaf.rollback-$([guid]::NewGuid().ToString('N'))"
    $BackupName = [System.IO.Path]::GetFileName($Backup)
    $HadPrevious = Test-Path -LiteralPath $DestinationPath
    if ($HadPrevious) {
        $Existing = Get-Item -LiteralPath $DestinationPath -Force
        if (-not $Existing.PSIsContainer -or (($Existing.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "Destination is not a safe directory"
        }
    }
    if ($HadPrevious) {
        Move-Item -LiteralPath $DestinationPath -Destination $Backup
    }
    try {
        Move-Item -LiteralPath $StagedDestination -Destination $DestinationPath
    }
    catch {
        if (Test-Path -LiteralPath $DestinationPath) {
            try {
                Remove-Item -LiteralPath $DestinationPath -Recurse -Force
            }
            catch {
                if ($HadPrevious -and (Test-Path -LiteralPath $Backup)) {
                    throw "result installation recovery required: preserve rollback backup $BackupName"
                }
                throw "Result installation cleanup failed before completion"
            }
        }
        if ($HadPrevious -and (Test-Path -LiteralPath $Backup)) {
            try {
                Move-Item -LiteralPath $Backup -Destination $DestinationPath
            }
            catch {
                throw "result installation recovery required: preserve rollback backup $BackupName"
            }
            throw "Unable to atomically install validated results; previous destination restored"
        }
        throw "Unable to atomically install validated results"
    }
    if ($HadPrevious) {
        try {
            Remove-Item -LiteralPath $Backup -Recurse -Force
        }
        catch {
            throw "Validated results are installed; rollback backup cleanup required for $BackupName"
        }
    }
}

$DestinationPath = Get-AbsolutePath -Value $Destination
$DestinationParent = Split-Path -Parent $DestinationPath
$DestinationLeaf = Split-Path -Leaf $DestinationPath
if ([string]::IsNullOrWhiteSpace($DestinationLeaf)) {
    throw "Destination must name a child directory"
}
Assert-SafeDestinationPath -DestinationPath $DestinationPath

if ($VerifyOnly) {
    $OperationParent = [System.IO.Path]::GetTempPath()
}
else {
    $null = New-Item -ItemType Directory -Force -Path $DestinationParent
    Assert-SafeDestinationPath -DestinationPath $DestinationPath
    $DestinationParentVolume = [System.IO.Path]::GetPathRoot(
        $DestinationParent
    )
    $DestinationVolume = [System.IO.Path]::GetPathRoot($DestinationPath)
    if (-not [string]::Equals(
        $DestinationParentVolume,
        $DestinationVolume,
        $PathComparison
    )) {
        throw "Destination staging must remain on the destination volume"
    }
    $OperationParent = $DestinationParent
}
$OperationRoot = Join-Path $OperationParent ".full-v2-verify-$([guid]::NewGuid().ToString('N'))"
$null = New-Item -ItemType Directory -Path $OperationRoot
$StagedDestination = Join-Path $OperationRoot "validated-results"
$null = New-Item -ItemType Directory -Path $StagedDestination

try {
    if ([string]::IsNullOrWhiteSpace($ArchiveDirectory)) {
        $DownloadDir = Join-Path $OperationRoot "release-assets"
        $null = New-Item -ItemType Directory -Path $DownloadDir
        & gh auth status --hostname github.com 1>$null 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "GitHub CLI authentication failed"
        }
        & gh release download $Tag --repo $Repository --dir $DownloadDir --clobber --pattern $($AssetNames[0]) --pattern $($AssetNames[1]) --pattern $($AssetNames[2]) --pattern $($AssetNames[3]) 1>$null 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "GitHub Release download failed"
        }
        $AssetDirectory = $DownloadDir
    }
    else {
        $AssetDirectory = (Resolve-Path -LiteralPath $ArchiveDirectory).Path
    }
    Assert-ExactAssetDirectory -Path $AssetDirectory
    $PrivateAssetDirectory = Join-Path $OperationRoot "private-assets"
    $AssetDirectory = Copy-PrivateAssetSnapshot `
        -SourceDirectory $AssetDirectory `
        -SnapshotDirectory $PrivateAssetDirectory

    $InferenceArchive = Join-Path $AssetDirectory $InferenceArchiveName
    $FinalArchive = Join-Path $AssetDirectory $FinalArchiveName
    $InferenceSha256 = Assert-ExactChecksum `
        -ArchivePath $InferenceArchive `
        -ArchiveName $InferenceArchiveName
    $FinalSha256 = Assert-ExactChecksum `
        -ArchivePath $FinalArchive `
        -ArchiveName $FinalArchiveName
    Assert-ArchiveMembers `
        -ArchivePath $InferenceArchive `
        -ExpectedRoot $InferenceRoot `
        -Payload $InferencePayload
    Assert-ArchiveMembers `
        -ArchivePath $FinalArchive `
        -ExpectedRoot $FinalRoot `
        -Payload $FinalPayload
    Expand-ValidatedArchive `
        -ArchivePath $InferenceArchive `
        -DestinationPath $StagedDestination
    Expand-ValidatedArchive `
        -ArchivePath $FinalArchive `
        -DestinationPath $StagedDestination
    try {
        $PostInferenceSha256 = Assert-ExactChecksum `
            -ArchivePath $InferenceArchive `
            -ArchiveName $InferenceArchiveName
        $PostFinalSha256 = Assert-ExactChecksum `
            -ArchivePath $FinalArchive `
            -ArchiveName $FinalArchiveName
    }
    catch {
        throw "post-extraction archive SHA256 verification failed"
    }
    if (
        $PostInferenceSha256 -cne $InferenceSha256 -or
        $PostFinalSha256 -cne $FinalSha256
    ) {
        throw "post-extraction archive SHA256 changed"
    }
    Assert-ResultContent -StagedDestination $StagedDestination

    if (-not $VerifyOnly -and -not $SkipResultsBranchFetch) {
        & git -C $Root fetch origin "+$ResultsBranch`:refs/remotes/origin/$ResultsBranch" 1>$null 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Results branch fetch failed"
        }
    }

    if (-not $VerifyOnly) {
        Assert-SafeDestinationPath -DestinationPath $DestinationPath
        Install-StagedDestination `
            -StagedDestination $StagedDestination `
            -DestinationPath $DestinationPath
    }

    $Result = [ordered]@{
        status = "validated"
        verify_only = [bool]$VerifyOnly
        destination = $DestinationPath
        release = "https://github.com/$Repository/releases/tag/$Tag"
        result_branch = $ResultsBranch
        prediction_sample_count = 5658
        bertscore_status = "computed"
        inference_sha256 = $InferenceSha256
        final_sha256 = $FinalSha256
    }
}
finally {
    if (Test-Path -LiteralPath $OperationRoot) {
        Remove-Item -LiteralPath $OperationRoot -Recurse -Force
    }
}

$Result | ConvertTo-Json -Compress
