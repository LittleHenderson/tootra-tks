# TKS Training Packs (Operators + RPM/Regulators)

## Files
- `tks_operator_training_pack_2000.jsonl` — 2,000 chat-style operator records
- `tks_rpm_fm_acbe_mvr_training_pack_800.jsonl` — 800 chat-style RPM/regulator records
- `tks_training_packs_SCHEMA.md` — this schema/notes

## Common JSONL record format
Each line is a JSON object:

- `type`: task type string
- `meta`: metadata for filtering (difficulty, ids, operators, etc.)
- `messages`: chat array with:
  - system
  - user
  - assistant

## Pack 1: Operator tasks (`tks_operator_training_pack_2000.jsonl`)
Types included:
- `operator_grounding`
- `operator_minimal_pair`
- `operator_selection`
- `substitution_test`
- `pemdas_order`
- `grouping_effect`
- `operator_error_correction`
- `operator_equivalence_check`

Operator semantics enforced:
- `+` Association (link/coexist/support)
- `-` Disassociation (remove the second from the first)
- `×` Intensification (amplify/fuse effect)
- `÷` Opposition (contrast/tension/polarity)

## Pack 2: RPM + Regulators (`tks_rpm_fm_acbe_mvr_training_pack_800.jsonl`)
Types included:
- `fm_female_authority` (Noetic 5): “reservoir/authority/expert/embodiment” (generic; no real-person names)
- `fm_male_superset` (Noetic 6): “larger set / related ideas”
- `acbe_cause_effect` (Noetics 8/9): cause chain + effect chain for unknown capabilities
- `mvr_activation_plan` (Noetics 1/4/7): Mind → Vibration → Rhythm activation to build desire
- `rpm_cascade_generate`: produce 2–3 layer RPM cascade (colon nesting Option A Outer:Inner)
- `rpm_trigger_classify`: select which regulator should activate

### Foundation stacks
Pack 2 uses simplified foundation stacks like `F1:F5:F2` to express nested drive context.
Replace with your full 28 sub-foundations as needed (e.g., `1a:5b:2b`).

## Notes / Next extensions
- Replace synthetic symbols with canonical element/acquisition text from your corpus.
- Add “hard negatives” and “minimal pairs” for regulators (e.g., FM vs ACBE confusion).
- Add real RPM cases from your practice + calculus-derived interpretations for higher fidelity.
