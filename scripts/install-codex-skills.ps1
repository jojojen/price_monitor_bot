param(
    [string]$CodexHome = $env:CODEX_HOME
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $repoRoot "ai\skills"

if (-not (Test-Path -LiteralPath $sourceRoot)) {
    throw "Skill source directory not found: $sourceRoot"
}

if ([string]::IsNullOrWhiteSpace($CodexHome)) {
    $CodexHome = Join-Path $HOME ".codex"
}

$targetRoot = Join-Path $CodexHome "skills"
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

$targetRootFull = [System.IO.Path]::GetFullPath($targetRoot)
$skillDirs = Get-ChildItem -LiteralPath $sourceRoot -Directory

foreach ($skillDir in $skillDirs) {
    $targetPath = Join-Path $targetRoot $skillDir.Name
    $targetPathFull = [System.IO.Path]::GetFullPath($targetPath)

    if (-not $targetPathFull.StartsWith($targetRootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to write outside Codex skills directory: $targetPathFull"
    }

    if (Test-Path -LiteralPath $targetPath) {
        Remove-Item -LiteralPath $targetPath -Recurse -Force
    }

    Copy-Item -LiteralPath $skillDir.FullName -Destination $targetPath -Recurse -Force
    Write-Host "Installed skill: $($skillDir.Name) -> $targetPath"
}

Write-Host "Restart Codex to pick up the updated skills."
