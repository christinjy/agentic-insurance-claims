# Agent Learning Projects

This repository collects my study and reproduction work on LLM agents and agent training.

## Contents

### `SEED/`

The reference implementation for **SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning**. It contains the paper code, environment integrations, data preparation scripts, SFT/RL training scripts, and evaluation utilities.

### `insurance_claim_agent_sandbox/`

A small insurance-claims agent sandbox used to study a complete agent workflow:

```text
case -> skill routing -> tool calls -> response -> verifier -> reward -> training data
```

It includes business skills, simulated tools, protocol validation, trajectory recording, reward verification, regression evaluation, and SFT/RL dataset exporters.

### `design_files/`

Study notes, setup instructions, and project design materials.

## Main Ideas

- Skills constrain what an agent is allowed to do.
- Tool observations are recorded as part of an episode trajectory.
- A verifier checks both task outcome and safety guardrails.
- Step-level records can be used for SFT; complete rollouts can be used for RL.
- Hindsight skills and on-policy distillation connect the agent trajectory to later training.

## Notes

This repository is a learning and reproduction project. The included training outputs and evaluation artifacts are examples from the sandbox; large model checkpoints and private API credentials are intentionally excluded.

See the README files inside `SEED/` and `insurance_claim_agent_sandbox/` for detailed instructions.
