# COLON_META — TKS Consciousness Unlock Training Pack

Generated: 2026-01-04 22:44 UTC

Files:
- `tks_consciousness_unlock_COLON_meta_pack_1000.jsonl`
- `tks_consciousness_unlock_COLON_meta_pack_SCHEMA.md`
- `tks_consciousness_unlock_COLON_meta_pack_SAMPLE_25.jsonl`

## Variant intent
- High-Stakes unlocks: **False**
- Critical unlocks: **False**
- admin_unlock field included: **False**
- colon-nesting DPS meta line included: **True**

## Rules summary
- NW = V * I * (1 - S) clipped to [0,1]
- HEAVY if NW >= 0.7
- COUNT if 0.45 <= NW < 0.7
- NOCOUNT otherwise
- COUNT/HEAVY increment tokens by 1
- Unlock:
  - Critical: never unlock
  - High-Stakes: disabled
  - cooldown blocks unlock
  - HEAVY unlocks immediately when eligible
  - COUNT unlocks when tokens reaches 5 when eligible
- On unlock: p_max += 1 (cap 5), tokens reset to 0, cooldown set to 10
- If no unlock: cooldown decreases by 1 (min 0)

## Output format (strict)
Assistant output includes:
- NW/Class/Unlock/Reason
- p_max/tokens/cooldown transitions
- LogEvent NOVELTY_CANDIDATE JSON
- Optional LogEvent DEPTH_UNLOCK JSON
- Plus final TKS colon meta line `TKS: DPS:p_max:<new>:tokens:<new>:cooldown:<new>`

## Record format
Each JSONL line is an object with:
- type: `tks_consciousness_unlock_training`
- meta: id, variant, mode, config
- messages: system/user/assistant
