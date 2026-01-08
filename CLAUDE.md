# Claude Code Project Instructions

## Required Reading

Before starting work, read:
1. `AI_INSTRUCTIONS.md` - General AI assistant rules
2. `docs/EXPLAIN_FORMAT_STANDARD.md` - How to explain concepts to user

## Explanation Format

When user asks to explain/define/break down technical concepts, use the format in `docs/EXPLAIN_FORMAT_STANDARD.md`:

- Three-column table: **Metaphor | Technical | 6th Grade**
- "Like this:" everyday analogies
- Bullet points with numbers
- End with Quick Reference Card

## Training

- GPU ONLY - no CPU training allowed
- Use `train_v5.py` - don't create new scripts
- Scripts will exit if no CUDA GPU

## User Preferences

- No emojis unless requested
- Concise CLI-friendly output
- Read before editing files
