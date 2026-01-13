# TKS Consciousness Unlock Training Pack (1,200 JSONL)

Files:
- `tks_consciousness_unlock_training_pack_1200.jsonl`
- `tks_consciousness_unlock_training_pack_SCHEMA.md`
- `tks_consciousness_unlock_training_pack_SAMPLE_30.jsonl`

## Purpose
Train the model to:
1) Compute **Novelty Weight**: `NW = V * I * (1 - S)` clipped to `[0,1]`
2) Classify novelty: **HEAVY / COUNT / NOCOUNT** using thresholds
3) Update the **Depth Permission System (DPS)** state:
   - `novelty_tokens` increments for COUNT or HEAVY
   - Unlock rules (non-bypassable):
     - Critical: never unlock
     - High-Stakes: unlocks disabled in this pack
     - cooldown blocks unlock
     - HEAVY can unlock (p_max + 1, cap 5)
     - COUNT unlock when tokens reach 5
   - On unlock: tokens reset to 0, cooldown set to 10
   - If no unlock: cooldown decreases by 1 (min 0)
4) Emit required **LogEvent JSON** lines:
   - `NOVELTY_CANDIDATE` always
   - `DEPTH_UNLOCK` only if unlock occurs

## Output format (strict)
Assistant output must be exactly:
- `NW: <3 decimals>`
- `Class: HEAVY|COUNT|NOCOUNT`
- `Unlock: YES|NO`
- `Reason: <reason>`
- `p_max: <old> -> <new>`
- `tokens: <old> -> <new>`
- `cooldown: <old> -> <new>`
- `LogEvent: <NOVELTY_CANDIDATE JSON>`
- (Optional) `LogEvent: <DEPTH_UNLOCK JSON>` when unlocked

## Record format
Each JSONL line is an object with:
- `type`: `"tks_consciousness_unlock_training"`
- `meta`: id/mode/config timestamps
- `messages`: system/user/assistant
