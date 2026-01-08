# Mission: Data & Canon Steward (agent_data)

**Objective:** Build and validate the Source of Truth (Canon).

## Critical Tasks
- [ ] Scan 'canon/raw_pdfs' and extract normalized definitions into 'canon/normalized/*.json'.
- [ ] Implement 'scripts/canon/build_canon.py' and 'validate_canon.py'.
- [ ] Maintain 'canon/gold' with undeniable examples (JSONL).
- [ ] Ensure 'canon_index.json' contains hashes to prevent drift.

## Acceptance Criteria
> Canon validation script passes with 0 errors. Hashes match.

## Context
Refer to `specs/TKS_Master_README_CLI_Runbook.md` for full operational details.