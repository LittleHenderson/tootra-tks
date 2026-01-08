# Mission: Backend & Runtime Engineer (agent_be)

**Objective:** Implement the core runtime modules and CLI.

## Critical Tasks
- [ ] Move 'tks_llm_core_v4.py' to 'src/tks/core/' and refactor imports.
- [ ] Implement 'src/tks/governance' (High-Stakes gates, Clearance Tokens).
- [ ] Implement 'src/tks/planning' (RPM nesting logic, Regulator triggers).
- [ ] Create CLI entrypoint 'src/cli/run_episode.py'.

## Acceptance Criteria
> Can run 'python -m tks.cli.run_episode' end-to-end with a stub tool.

## Context
Refer to `specs/TKS_Master_README_CLI_Runbook.md` for full operational details.