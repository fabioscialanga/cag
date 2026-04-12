param(
    [string]$OutputDir = ".\artifacts\eval_runs",
    [string]$CompareOutputDir = ".\artifacts\eval_comparisons",
    [ValidateSet("auto", "off", "required")]
    [string]$JudgeMode = "off",
    [int]$Runs = 1
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python virtualenv not found at $pythonExe"
}

$env:PYTHONPATH = "src"

function Invoke-EvalRun {
    param([string]$SystemName)

    Write-Host "Running eval for $SystemName ..."
    & $pythonExe -m cag.eval.run --system $SystemName --runs $Runs --judge-mode $JudgeMode --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed for $SystemName"
    }

    return Get-ChildItem $OutputDir -Directory |
        Where-Object { $_.Name -like "*_${SystemName}" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

$ragRun = Invoke-EvalRun -SystemName "rag_baseline"
$cagNoSelectionRun = Invoke-EvalRun -SystemName "cag_no_selection"
$cagRun = Invoke-EvalRun -SystemName "cag"

Write-Host "Building comparison ..."
& $pythonExe -m cag.eval.compare --runs $ragRun.FullName $cagNoSelectionRun.FullName $cagRun.FullName --output-dir $CompareOutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Comparison generation failed"
}

$comparisonDir = Get-ChildItem $CompareOutputDir -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Write-Host ""
Write-Host "Triplet benchmark complete."
Write-Host "rag_baseline:     $($ragRun.FullName)"
Write-Host "cag_no_selection: $($cagNoSelectionRun.FullName)"
Write-Host "cag:              $($cagRun.FullName)"
Write-Host "comparison:       $($comparisonDir.FullName)"
