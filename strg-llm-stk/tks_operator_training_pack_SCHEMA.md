# TKS Operator Training Pack (250 JSONL records)

Files:
- `tks_operator_training_pack_250.jsonl` — chat-style JSONL records
- `tks_operator_training_pack_SCHEMA.md` — this schema/notes

## Record format
Each line is a JSON object with:

- `type` : one of
  - `operator_grounding`
  - `operator_minimal_pair`
  - `operator_selection`
  - `substitution_test`
  - `pemdas_order`
  - `grouping_effect`
- `meta` : metadata for filtering/analysis (operator, difficulty, ids)
- `messages` : chat messages array suitable for SFT-style training

### `messages` layout
- system: instructs the model to follow operator semantics and be concise
- user: prompt
- assistant: target response

## Operator semantics used (canonical)
- `+` Association: combine/link so ideas coexist/support
- `-` Disassociation: remove/sever the second idea’s influence from the first
- `×` Intensification: amplify/fuse the interaction (stronger effect)
- `÷` Opposition: set in polarity/contrast (tension)

## Notes
- Symbols are synthetic but structurally valid (Element + Noetic superscript + Foundation subscript).
- No numeric computation is expected; precedence tasks describe *order of operations* only.
- Designed to be extended with your canonical element/acquisition text and real RPM cases.
