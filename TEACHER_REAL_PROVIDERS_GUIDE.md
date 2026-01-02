# Teacher Generation with Real LLM Providers - Quick Guide

This guide shows how to run teacher generation with real LLM providers (OpenAI, Anthropic, Gemini) instead of the mock provider used in Phase 5.

---

## Prerequisites

### 1. Install Required Dependencies

The teacher system already has the necessary dependencies, but ensure you have:
- `openai` package (for OpenAI/GPT-4)
- `anthropic` package (for Claude)
- `google-generativeai` package (for Gemini)

### 2. Configure API Keys

Set environment variables for the providers you want to use:

**Windows (PowerShell)**:
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GOOGLE_API_KEY = "..."  # or GEMINI_API_KEY
```

**Windows (Command Prompt)**:
```cmd
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...
set GOOGLE_API_KEY=...
```

**Linux/Mac**:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

---

## Single Provider Examples

### OpenAI (GPT-4)
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs_gpt4.jsonl \
  --providers openai:gpt-4
```

**Supported OpenAI Models**:
- `gpt-4` - GPT-4 (recommended for quality)
- `gpt-4-turbo` - GPT-4 Turbo (faster, cheaper)
- `gpt-3.5-turbo` - GPT-3.5 Turbo (fastest, cheapest)

### Anthropic (Claude)
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs_claude.jsonl \
  --providers anthropic:claude-3-sonnet-20240229
```

**Supported Anthropic Models**:
- `claude-3-opus-20240229` - Claude 3 Opus (highest quality)
- `claude-3-sonnet-20240229` - Claude 3 Sonnet (balanced)
- `claude-3-haiku-20240307` - Claude 3 Haiku (fastest)

### Google Gemini
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs_gemini.jsonl \
  --providers gemini:gemini-1.5-pro
```

**Supported Gemini Models**:
- `gemini-1.5-pro` - Gemini 1.5 Pro (recommended)
- `gemini-pro` - Gemini Pro
- `gemini-ultra` - Gemini Ultra (when available)

---

## Multi-Provider Ensemble (Recommended)

For best results, use multiple providers and let the teacher ensemble select the best interpretations:

### Full Ensemble (3 providers)
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs_ensemble.jsonl \
  --providers openai:gpt-4 anthropic:claude-3-sonnet-20240229 gemini:gemini-1.5-pro
```

**Benefits**:
- Higher quality through consensus
- Canonical validation across multiple interpretations
- Agreement scoring to detect inconsistencies
- Redundancy if one provider fails

### Two-Provider Ensemble
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs_ensemble.jsonl \
  --providers openai:gpt-4 anthropic:claude-3-sonnet-20240229
```

---

## Configuration Options

### Minimum Canon Score
Set the minimum canon score threshold (default: 0.8):
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs.jsonl \
  --providers openai:gpt-4 \
  --min-canon 0.9
```

### Lenient Validation
Use lenient validation mode (allows more flexibility):
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs.jsonl \
  --providers openai:gpt-4 \
  --lenient
```

### Specific Task Types
Generate only specific task types:
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs.jsonl \
  --providers openai:gpt-4 \
  --tasks e2i i2e  # Only E2I and I2E tasks
```

**Available Task Types**:
- `e2i` - Equation to Interpretation
- `i2e` - Interpretation to Equation
- `s2e` - Scenario to Equation
- `e2rpm` - Equation to RPM
- `e2f` - Equation to Foundations
- `full` - Full pipeline

---

## Validation After Generation

Always validate outputs after generation:

```bash
# Validate the generated outputs
python scripts/validate_teacher_output.py output/teacher_outputs_gpt4.jsonl

# Expected output:
# Pass rate: 100.0% (or close to it)
# No canonical violations
```

---

## Cost Estimation

Approximate costs for generating 60 examples (15 equations × 4 tasks):

### OpenAI
- **GPT-4**: ~$0.50-1.00 (input + output)
- **GPT-4 Turbo**: ~$0.20-0.40
- **GPT-3.5 Turbo**: ~$0.05-0.10

### Anthropic
- **Claude 3 Opus**: ~$0.80-1.50
- **Claude 3 Sonnet**: ~$0.15-0.30
- **Claude 3 Haiku**: ~$0.05-0.10

### Google Gemini
- **Gemini 1.5 Pro**: ~$0.30-0.60 (or free tier)

### Multi-Provider Ensemble
- **3 Providers (GPT-4 + Claude Sonnet + Gemini Pro)**: ~$1.00-2.00 total

**Note**: Actual costs depend on prompt length, response length, and current API pricing.

---

## Troubleshooting

### API Key Errors
```
Error: API key not found for provider 'openai'
```
**Solution**: Set the `OPENAI_API_KEY` environment variable

### Rate Limiting
```
Error: Rate limit exceeded
```
**Solution**: The teacher system includes automatic retry logic with exponential backoff. For high-volume generation, consider:
- Using multiple API keys (if allowed by provider)
- Reducing batch size
- Adding delays between requests

### Canonical Validation Failures
```
Canonical validation failed: Invalid world 'Y'
```
**Solution**: This indicates the input equations have non-canonical elements. Check `data/equations.jsonl` and ensure only A, B, C, D worlds are used.

### Low Agreement Scores
```
Warning: Low agreement score (0.45)
```
**Solution**: This is normal when using multiple providers with different interpretations. The system will select the best response. If agreement is consistently low:
- Check if equations are ambiguous
- Verify providers are using the same TKS canon
- Consider using `--min-canon` to raise the bar

---

## Example Workflow

### 1. Prepare Input
```bash
# Review/edit equations
cat data/equations.jsonl
```

### 2. Set API Keys
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 3. Generate with Ensemble
```bash
python scripts/run_teacher.py generate \
  data/equations.jsonl \
  --output output/teacher_outputs_ensemble.jsonl \
  --providers openai:gpt-4 anthropic:claude-3-sonnet-20240229 \
  --min-canon 0.85
```

### 4. Validate Outputs
```bash
python scripts/validate_teacher_output.py \
  output/teacher_outputs_ensemble.jsonl
```

### 5. Review Statistics
The generation command will print:
- Total queries
- Success rate
- Canonical rejections
- Average canon score
- Provider statistics

---

## Advanced Usage

### Interpret Single Equation (Interactive)
```bash
python scripts/run_teacher.py interpret "B4 + C10" \
  --providers openai:gpt-4 anthropic:claude-3-sonnet-20240229
```

### Batch Process (No Training Data)
```bash
python scripts/run_teacher.py batch \
  data/equations.jsonl \
  --output output/interpretations.jsonl \
  --providers openai:gpt-4
```

### Test with Mock Provider (No API Keys Needed)
```bash
python scripts/run_teacher.py test
```

---

## Production Recommendations

For production use with real LLM providers:

1. **Use ensemble mode** with at least 2 providers for quality
2. **Set min-canon ≥ 0.85** for strict canonical compliance
3. **Always validate** outputs before using in training
4. **Monitor costs** by tracking API usage
5. **Log statistics** to detect quality drift
6. **Use caching** to avoid re-generating identical equations
7. **Version outputs** to track which provider/model generated each example

---

## Next Steps

After generating with real providers:

1. **Validate** outputs: `python scripts/validate_teacher_output.py ...`
2. **Review quality**: Check canon scores and agreement scores
3. **Expand dataset**: Add more equations to `data/equations.jsonl`
4. **Feed to training**: Use outputs in `scripts/train_with_augmented.py`
5. **Iterate**: Refine based on model performance

---

**Note**: Phase 5 used mock provider to demonstrate the pipeline without requiring API keys. For production quality, use real LLM providers as described above.
