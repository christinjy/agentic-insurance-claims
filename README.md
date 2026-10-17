# Agentic Insurance Claims Platform

An engineering reproduction and extension project for reliable, tool-using LLM agents in insurance claims service.

The project combines the **SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning** reference implementation with an insurance claims agent sandbox. I participated in organizing the end-to-end workflow, extending the local serving and evaluation utilities, and documenting the data and training pipeline.

## Project Motivation

Insurance claims service is a multi-step decision task. A useful agent must understand the case, select the right business skill, inspect case data, perform only permitted actions, and handle escalation when the case is uncertain or high risk. A fluent answer alone is not sufficient: the system must also enforce protocol, business boundaries, and auditability.

## System Workflow

```text
case and user message
        |
        v
skill routing -> protocol-checked action -> simulated business tool
        |                                      |
        +---------- trajectory and observations
                                               |
                                               v
                              verifier -> reward and regression report
                                               |
                                               v
                                  SFT / RL training records
```

The sandbox supports a structured agent protocol:

```text
load_skills -> load_action (zero or more) -> respond
```

The harness validates JSON structure, skill and action permissions, tool budgets, duplicate side effects, and safe termination. The verifier evaluates the complete trajectory, including required actions, guardrails, and final disposition.

## Contributions

- Organized an insurance claims agent workflow around skills, tools, trajectory state, verification, and reward.
- Implemented configurable serving demos for replaying different claim cases.
- Added a quick regression runner that saves summary metrics and replayable case records.
- Structured step-level SFT records and episode-level RL rollout records for downstream training.
- Added protocol and safety boundaries for missing documents, partial payment, disputes, risk flags, and mandatory human review.
- Connected the sandbox concepts to SEED's hindsight-skill SFT and OPD-assisted RL pipeline.

## Repository Structure

| Directory | Description |
| --- | --- |
| `insurance_claim_agent_sandbox/` | Insurance claims agent, simulated tools, harness, verifier, regression runner, and training-data exporters. |
| `SEED/` | Reference implementation and training scripts for SEED. |
| `design_files/` | Design notes, setup instructions, and project documentation. |

## Quick Start

Configure the model API through environment variables, then run a single case:

```powershell
powershell -ExecutionPolicy Bypass -File .\insurance_claim_agent_sandbox\run_demo.ps1
```

Run a selected case:

```powershell
powershell -ExecutionPolicy Bypass -File .\insurance_claim_agent_sandbox\run_demo.ps1 `
  -CaseId claim_high_risk_review_0308 `
  -MaxSteps 8
```

Run a small regression suite and save replay artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File .\insurance_claim_agent_sandbox\run_regression.ps1 `
  -LimitCases 5 `
  -MaxSteps 8
```

Detailed implementation notes are available in the README files inside `SEED/` and `insurance_claim_agent_sandbox/`.

## Attribution and Scope

`SEED/` is included as the reference research code for the SEED project. This repository documents my participation in the surrounding engineering reproduction, sandbox extension, evaluation workflow, and project organization. It does not claim ownership of the original SEED research or code. Private API keys, model credentials, and large checkpoints are excluded.
