param(
  [string]$CaseId = "claim_pending_docs_0142",
  [int]$MaxSteps = 8
)

Set-Location $PSScriptRoot
python -m claim_sandbox.harness.serving `
  --case-id $CaseId `
  --max-steps $MaxSteps
