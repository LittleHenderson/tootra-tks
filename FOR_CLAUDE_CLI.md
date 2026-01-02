# Claude CLI Handoff: TKS Teacher Generation (600k -> 1M)

This file is a handoff for Claude CLI to continue the work.

## Goal
Generate 600k (then 1M) teacher examples with both providers per chunk, using
8 parallel workers and a supervisor that merges results and checks completeness,
then train on GPU with the non-Transformer TKS pipeline.

## Repo
`/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS`

## What’s already done
- Added worker chunk-range support + skip-combine to `scripts/run_teacher_agents.py`.
  - New flags: `--chunk-dir`, `--chunk-start`, `--chunk-end`, `--chunk-indices`,
    `--skip-combine`, `--chunk-dir` input mode.
  - Defaults set to `gemini:gemini-3-pro-preview` and `anthropic:claude-opus-4-5`.
  - Keys loader supports `export KEY=...` lines and only fills missing env vars.
- Added supervisor script `scripts/run_teacher_supervisor.py`:
  - Splits input JSONL into `equations_<idx>.jsonl` chunks.
  - Launches N worker processes (both providers per worker).
  - Collects outputs, checks for missing files, merges into `teacher_all.jsonl`.
  - Writes logs in `output/<run>/logs`.
- Disabled Transformer training path in `scripts/train_cuda.py`:
  - Default model is `tks_pipeline`.
  - If `--model-type simple` is used, the script exits.
- Python compile check passed:
  `python3 -m py_compile scripts/run_teacher_agents.py scripts/run_teacher_supervisor.py`

## Provider keys (user has local keys)
User stores keys in `~/.config/tks/keys.env` with:
```
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```
Loaded either via `source ~/.config/tks/keys.env` or `--keys-from-config`.

## Dependencies
If not installed:
```
python3 -m pip install --user --break-system-packages anthropic google-generativeai
```

## Hardware note (user: 16 GB RAM, RTX 4070)
- Data generation (cloud API): GPU not used; CPU/RAM are the constraints. 8 workers is fine.
- Training: uses GPU via `scripts/train_cuda.py`.

## 600k run (8 workers, both providers)
1) Generate seed file (only if not done yet):
```
python3 scripts/generate_new_canonical_equations.py \
  --count 600000 \
  --output data/equations_600k.jsonl
```

2) Supervisor run:
```
python3 scripts/run_teacher_supervisor.py \
  --data data/equations_600k.jsonl \
  --output-dir output/teacher_600k \
  --workers 8 \
  --chunks 80 \
  --keys-from-config ~/.config/tks/keys.env \
  --min-canon 0.9 \
  --append-fractals
```
Notes:
- Chunk files in `output/teacher_600k/chunks`.
- Worker logs in `output/teacher_600k/logs`.
- Outputs per provider: `output/teacher_600k/teacher_<provider>_<idx>.jsonl`.
- Final merge: `output/teacher_600k/teacher_all.jsonl`.
- If chunks already exist, use `--reuse-chunks` or `--overwrite-chunks`.
- If you hit rate limits, increase `--chunks` to 96 or 120 to reduce per-chunk size.

## 1M run (later)
Use the same command with `--data data/equations_1m.jsonl` and a larger
`--chunks` count. Suggested:
- 1M with 8 workers: `--chunks 120` (start) or `--chunks 160` (rate-limit friendly).

## Resume / re-run missing chunks
If supervisor reports missing outputs:
- Re-run the supervisor with `--reuse-chunks` to avoid re-splitting.
- Or re-run a worker range directly:
```
python3 scripts/run_teacher_agents.py \
  --chunk-dir output/teacher_600k/chunks \
  --chunk-start <start> \
  --chunk-end <end> \
  --output-dir output/teacher_600k \
  --keys-from-config ~/.config/tks/keys.env \
  --min-canon 0.9 \
  --skip-combine
```
Then re-run the supervisor with `--reuse-chunks` to re-merge.

### Automated resume helper
Use the resume script to re-run only missing outputs (per provider, batched):
```
python3 scripts/run_teacher_resume.py \
  --output-dir output/teacher_600k \
  --keys-from-config ~/.config/tks/keys.env \
  --min-canon 0.9 \
  --batch-size 10 \
  --max-parallel 2
```
After resume, re-run supervisor with `--reuse-chunks` to re-merge.

## GPU training (non-Transformer enforced)
`scripts/train_cuda.py` is now locked to the non-Transformer pipeline:
```
python3 scripts/train_cuda.py \
  --data output/teacher_600k/teacher_all.jsonl \
  --output-dir output/cuda_models_600k \
  --epochs 5 \
  --batch-size 16 \
  --model-type tks_pipeline
```
Notes:
- Transformer training is disabled; any attempt to use `--model-type simple` exits.
- If VRAM pressure occurs, reduce `--batch-size` or add `--gradient-checkpointing`.

## Known behavior
- Worker runs skip combine; supervisor does final merge.
- Append-fractals is handled by supervisor on the final combined file.
- Default providers already set to the requested models.

## What we’re trying to accomplish
Produce a large canon-validated teacher corpus using both Gemini and Claude,
with WSL-friendly parallelization and a deterministic supervisor that
verifies completeness and merges outputs.
