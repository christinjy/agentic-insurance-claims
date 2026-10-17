param(
  [int]$LimitCases = 5,
  [int]$MaxSteps = 8,
  [double]$Temperature = 0.2
)

Set-Location $PSScriptRoot
$outputDir = Join-Path $PSScriptRoot "artifacts\quick_regression"

python -m claim_sandbox.harness.regression `
  --limit-cases $LimitCases `
  --max-steps $MaxSteps `
  --temperature $Temperature `
  --outdir $outputDir
