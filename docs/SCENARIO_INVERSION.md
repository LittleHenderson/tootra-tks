# TKS Scenario Inversion CLI

## Usage

```bash
python scripts/run_scenario_inversion.py --story "<text>" --axes N,E,W,F,S,A,P --mode soft|hard|targeted [--from-foundation ... --to-foundation ... --from-world ... --to-world ...] [--format json|text]
```

## Parameters

- **Input (required, mutually exclusive):**
  - `--story "text"` - Natural language story input
  - `--equation "expr"` - TKS equation (e.g., "B5,+T,D3" or "B5 +T D3")

- **Inversion configuration:**
  - `--axes N,E,W,F,S,A,P` - Comma-separated axes (default: E)
    - `N` = Noetic (involution pairs: 2<->3, 5<->6, 8<->9)
    - `E` = Element (full element inversion: world + noetic)
    - `W` = World (world mirror: A<->D, B<->C)
    - `F` = Foundation (1<->7, 2<->6, 3<->5, 4 self-dual)
    - `S` = SubFoundation (foundation + world compound)
    - `A` = Acquisition (negation toggle)
    - `P` = Polarity (valence flip)
  - `--mode soft|hard|targeted` - Inversion mode (default: soft)
    - `soft` - Invert only where canonical dual/opposite exists
    - `hard` - Apply on all selected axes unconditionally
    - `targeted` - Apply TargetProfile remaps; others unchanged

- **Targeted mode parameters (optional):**
  - `--from-foundation N` - Source foundation (1-7)
  - `--to-foundation N` - Target foundation (1-7)
  - `--from-world X` - Source world (A/B/C/D)
  - `--to-world X` - Target world (A/B/C/D)

- **Output:**
  - `--format json|text` - Output format (default: text)

- **Validation mode:**
  - `--lenient` - Allow unknown tokens/operators with warnings (default: strict mode rejects unknown tokens)
  - **Default behavior**: Strict mode is enabled by default to ensure input validity
  - When strict mode detects unknown tokens, it provides helpful error messages suggesting valid alternatives

## Examples

### Story inversion (soft mode)
```bash
python scripts/run_scenario_inversion.py --story "She loved him" --axes W,N --mode soft
```

### Equation inversion (hard mode)
```bash
python scripts/run_scenario_inversion.py --equation "B5,+T,D3" --axes E --mode hard
```

### Targeted inversion
```bash
python scripts/run_scenario_inversion.py --story "Power corrupts" --axes F --mode targeted --from-foundation 5 --to-foundation 2
```

### JSON output
```bash
python scripts/run_scenario_inversion.py --story "She loved him" --axes W,N --mode soft --format json
```

### Lenient mode (allow unknown tokens)
```bash
# By default, strict mode rejects unknown tokens with helpful error messages
python scripts/run_scenario_inversion.py --story "The quantum superposition collapsed" --axes E --mode soft
# Error: Unknown tokens detected: quantum, superposition, collapsed
# Valid token categories:
#   - Words in LEXICON (e.g., 'woman', 'man', 'love', 'fear', 'power')
#   ...
# Use --lenient flag to allow unknown tokens with warnings.

# Use --lenient to allow unknown tokens with warnings
python scripts/run_scenario_inversion.py --story "The quantum superposition collapsed" --axes E --mode soft --lenient
```

## Anti-Attractor Synthesis

The `--anti-attractor` flag enables anti-attractor synthesis, which generates a counter-scenario by analyzing and inverting the attractor signature of the input expression.

### What is Anti-Attractor Synthesis?

Anti-attractor synthesis identifies the dominant patterns (attractors) in a TKS scenario and generates an opposing scenario (anti-attractor) that occupies the opposite region of TKS phase space. The algorithm:

1. **Extracts attractor signature**: Analyzes element frequencies, foundation tags, polarity, and dominant patterns
2. **Inverts the signature**: Applies canonical inversions to world, noetic, and foundation dimensions
3. **Synthesizes counter-scenario**: Generates a new TKS expression from the inverted signature

This creates maximum contrast while maintaining structural coherence, producing scenarios that repel from the original pattern.

### Plain-English Example
- Original drift: “He is anxious (emotional negative) and it’s making him sick (physical negative).”
- TKS form: `C3 -> D3` (Emotional Negative leads to Physical Negative).
- Anti-attractor target (flipped): Spiritual/Mental Positive pulling the other way.
- Counter-scenario: `B2 <- A2` — “He practices a calming belief/meditation (spiritual positive), which lifts his outlook (mental positive) and reduces the strain.”

### CLI Usage

```bash
python scripts/run_scenario_inversion.py --anti-attractor --equation "B2 -> C2 +T D2"
```

The `--anti-attractor` flag ignores `--axes` and `--mode` parameters, using its own signature-based inversion algorithm instead.

### Output

When using `--anti-attractor`, the output includes:

- **Original equation**: The input TKS expression
- **Attractor signature**: Element counts, dominant world/noetic, polarity (positive/negative/neutral), and foundation tags
- **Inverted equation**: The counter-scenario generated from the inverted signature
- **Inverted story**: Natural language rendering of the counter-scenario

### Examples

```bash
# Anti-attractor from equation
python scripts/run_scenario_inversion.py --anti-attractor --equation "B2 -> C2 +T D2"

# Anti-attractor from story
python scripts/run_scenario_inversion.py --anti-attractor --story "She loved him"

# JSON output format
python scripts/run_scenario_inversion.py --anti-attractor --equation "C3 -> D3" --format json
```

## Running Tests

```bash
# Run all tests
python -m pytest tests

# Run specific scenario inversion CLI tests
python tests/test_scenario_inversion_cli.py

# Run anti-attractor synthesis tests
python -m pytest tests/test_anti_attractor.py -v

# Print noetic inversion mappings (1-based and 0-based views)
python scripts/print_noetic_mapping.py
```

## Data Augmentation

The augmentation script (`scripts/generate_augmented_data.py`) provides batch data augmentation for training datasets using scenario inversion and anti-attractor synthesis.

### Usage

```bash
python scripts/generate_augmented_data.py \
  --input data/pilot/stories.jsonl \
  --output data/pilot/augmented.jsonl \
  --axes W,N,F \
  --use-anti-attractor \
  --validate
```

### CLI Flags

**Required:**
- `--input`: Input JSONL corpus file (one scenario per line)
- `--output`: Output JSONL file with augmented entries

**Inversion Configuration:**
- `--axes`: Comma-separated inversion axes (default: `W,N`)
  - Available axes: `N` (Noetic), `E` (Element), `W` (World), `F` (Foundation), `S` (SubFoundation), `A` (Acquisition), `P` (Polarity)
- `--mode`: Inversion mode - `soft`, `hard`, or `targeted` (default: `soft`)
  - `soft`: Invert only where canonical dual/opposite exists
  - `hard`: Apply on all selected axes unconditionally
  - `targeted`: Apply TargetProfile remaps; others unchanged

**Anti-Attractor Settings:**
- `--use-anti-attractor`: Enable anti-attractor counter-scenario generation
- `--anti-elements`: Number of elements in anti-attractor scenarios (default: 3)

**Validation Settings:**
- `--validate`: Run canonical validation on augmented data (default: enabled)
  - Strict validation by default - only canonically valid scenarios are kept
- `--lenient`: Allow unknown tokens with warnings (default: strict mode)
  - Default behavior rejects unknown tokens with helpful error messages
  - Use `--lenient` to opt-out of strict validation for experimental data
- `--min-pass-rate`: Minimum validation pass rate (default: 0.90)

**Output Settings:**
- `--save-metrics`: Save augmentation metrics to JSON (default: enabled)
- `--verbose`: Verbose output (default: enabled)

### Default Behavior

**Strict Validation:**
- By default, the script uses strict validation mode
- Only scenarios that parse to valid TKS expressions are included in output
- Unknown tokens are rejected with helpful error messages
- Use `--lenient` flag to allow unknown tokens with warnings

**Default Axes:**
- If `--axes` is not specified, defaults to `W,N` (World + Noetic)
- World axis inverts spiritual/physical domains (A<->D, B<->C)
- Noetic axis inverts mind principles (2<->3, 5<->6, 8<->9)

**Default Mode:**
- If `--mode` is not specified, defaults to `soft`
- Soft mode inverts only where canonical oppositions exist
- Preserves semantic consistency while generating meaningful contrasts

### Examples

```bash
# Basic augmentation with default settings (W,N axes, soft mode, strict validation)
python scripts/generate_augmented_data.py \
  --input data/pilot/stories.jsonl \
  --output data/pilot/augmented.jsonl

# Multi-axis augmentation with anti-attractor
python scripts/generate_augmented_data.py \
  --input data/pilot/stories.jsonl \
  --output data/pilot/augmented.jsonl \
  --axes W,N,F \
  --use-anti-attractor \
  --validate

# Lenient mode for experimental data (allows unknown tokens)
python scripts/generate_augmented_data.py \
  --input data/experimental/raw.jsonl \
  --output data/experimental/augmented.jsonl \
  --axes W,N \
  --lenient

# Production mode with high quality threshold
python scripts/generate_augmented_data.py \
  --input data/training/corpus.jsonl \
  --output data/training/augmented.jsonl \
  --axes W,N \
  --mode soft \
  --min-pass-rate 0.95 \
  --save-metrics
```

### Output Format

The output JSONL file contains augmented scenarios with metadata:

```json
{
  "story": "A physical student effects negative resistance",
  "expr": "D3 -> C3",
  "aug_type": "inverted",
  "source_id": 0,
  "validator_pass": true,
  "expr_elements": ["D3", "C3"],
  "expr_ops": ["->"],
  "axes": ["W", "N"],
  "mode": "soft"
}
```

Field descriptions:
- `aug_type`: One of `"original"`, `"inverted"`, or `"anti_attractor"`
- `source_id`: Index of parent scenario for inverted/anti-attractor entries
- `validator_pass`: Boolean indicating canonical validation result
- `expr_elements`: List of TKS elements extracted from expression
- `expr_ops`: List of operators connecting elements

### Metrics Output

When `--save-metrics` is enabled (default), a metrics JSON file is created alongside the output:

```json
{
  "original_count": 100,
  "inverted_count": 200,
  "anti_attractor_count": 100,
  "validation_failures": 5,
  "augmentation_ratio": 3.0,
  "validator_pass_rate": 0.95,
  "world_validity": 0.98,
  "noetic_validity": 0.97,
  "operator_validity": 0.99,
  "structural_validity": 0.96
}
```

## Canon Constraints

The TKS system enforces canonical constraints:
- **Worlds:** A, B, C, D (fixed)
- **Noetics:** 1-10 (fixed range)
- **Foundations:** 1-7 (fixed range)
- **Involution pairs:** 2<->3, 5<->6, 8<->9 (noetic axis)
- **World mirrors:** A<->D, B<->C (world axis)
- **Foundation mirrors:** 1<->7, 2<->6, 3<->5, 4 (self-dual)

## Extended Syntax Support

The TKS parser supports extended token syntax for richer element representation with sense and foundation suffixes.

### Supported Formats

| Format | Example | Description |
|--------|---------|-------------|
| Basic | `B8` | World + Noetic (standard format) |
| Sense suffix | `B8^5` | Add sense index using caret notation |
| Foundation suffix | `B8_d5` | Add foundation context (foundation 5 in world D) |
| Full extended | `B8^5_d5` | Combine sense and foundation annotations |

### Extended Syntax Examples

**Basic tokens:**
```
B8    # Mental Above (world B, noetic 8)
D5    # Physical Female (world D, noetic 5)
C2    # Emotional Positive (world C, noetic 2)
```

**With sense suffix (^ notation):**
```
B8^5  # Mental Above, sense 5
D5^1  # Physical Female, sense 1
C3^2  # Emotional Negative, sense 2
```

**With foundation suffix (_wN notation):**
```
B8_d5   # Mental Above with foundation 5 in world D (Material Power)
D5_a2   # Physical Female with foundation 2 in world A (Spiritual Wisdom)
C3_b4   # Emotional Negative with foundation 4 in world B (Mental Virtue)
```

**Full extended syntax:**
```
B8^5_d5    # Mental Above, sense 5, foundation 5 in world D
C3^2_a7    # Emotional Negative, sense 2, foundation 7 in world A
D10^1_b3   # Physical Below, sense 1, foundation 3 in world B
```

### Valid Ranges

Extended syntax must conform to canonical constraints:
- **Worlds:** A/B/C/D only
- **Noetics:** 1-10
- **Sense indices:** Any positive integer (typically context-dependent)
- **Foundations:** 1-7
- **Subfoundation worlds:** a/b/c/d (case-insensitive, maps to A/B/C/D)

### Using Extended Syntax in Equations

Extended syntax works seamlessly in TKS equations:

```bash
# Equation with sense suffixes
python scripts/run_scenario_inversion.py --equation "B8^5 +T C3^2 -> D6^1" --axes W,N --mode soft

# Equation with foundation suffixes
python scripts/run_scenario_inversion.py --equation "B8_d5 -> C2_a7" --axes F --mode soft

# Full extended equation
python scripts/run_scenario_inversion.py --equation "B8^5_d5 +T C3^2_a7 -> D6^1_b3" --axes W,N,F --mode soft
```

### Error Messages

The parser provides helpful guidance when invalid syntax is detected:

```
Invalid token 'B8x'.

Extended syntax formats:
  - Basic: B8, D5 (world + noetic)
  - Sense suffix: B8^5 (sense 5) or B8.5 (backward compatible)
  - Foundation suffix: B8_d5 (foundation 5 in world D)
  - Full extended: B8^5_d5 (sense 5, foundation 5 in world D)

Valid ranges:
  - Worlds: A/B/C/D only
  - Noetics: 1-10
  - Foundations: 1-7
  - Foundation worlds: a/b/c/d (case-insensitive)
```

### Backward Compatibility

The extended syntax is fully backward compatible:
- Basic tokens (`B8`, `D5`) continue to work unchanged
- Dot notation for sense (`B8.5`) is supported for backward compatibility
- Caret notation (`B8^5`) is the canonical form for sense suffixes
- All existing equations and scripts work without modification
