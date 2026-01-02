# TKS v7.3 Compilation Instructions

## Overview

This directory contains the TKS Formal Mathematical Manual v7.3 in modular LaTeX format.

## File Structure

### Master Files

| File | Description |
|------|-------------|
| `TKS_v7.3_MASTER.tex` | Original modular master file (uses `\input{}`) |
| `TKS_v7.3_PRINT.tex` | **PDF-ready version** with complete preamble, lstlisting styles, and algorithm packages |

### Track Files (Content Modules)

| File | Description | Agent | Approx. Size |
|------|-------------|-------|--------------|
| `TKS_v7.3_Math.tex` | Transfinite Noetic Fractals | tks-math | ~80k tokens |
| `TKS_v7.3_Compiler.tex` | Extended Language and VM | tks-compiler | ~81k tokens |
| `TKS_v7.3_Quantum.tex` | Quantum Noetics | tks-quantum | ~87k tokens |
| `TKS_v7.3_Meta.tex` | 2-Categories and Topoi | tks-meta | ~96k tokens |

## Current Compilation Status

### IMPORTANT: Known Issues in Track Files

The track files contain LaTeX syntax issues that prevent successful PDF compilation. These issues are WITHIN the track files themselves (not the PRINT preamble):

1. **Algorithm environment syntax errors** (Compiler track, Quantum track):
   - `\Call{}` commands containing math-mode with special characters
   - Located around lines 4200+ in the combined document
   - Example: `\Call{Check}{\mathsf{EX}Z}` causes parsing errors

2. **tikz-cd diagram errors** (Meta track):
   - Some commutative diagrams reference undefined nodes
   - "No shape named 'tikz@f@12-2-1' is known" errors

3. **Undefined control sequences** (various tracks):
   - Some custom macros used but not defined in preamble
   - Most are minor and do not halt compilation in nonstopmode

4. **Figure placement issues** (Meta track):
   - "Not in outer par mode" errors from figures inside tcolorbox

### Required Fixes (in track files)

To achieve successful compilation, the following track files need corrections:

- `TKS_v7.3_Compiler.tex`: Fix algorithm pseudocode with math-mode content
- `TKS_v7.3_Quantum.tex`: Fix algorithm pseudocode with math-mode content
- `TKS_v7.3_Meta.tex`: Fix tikz-cd diagrams and figure placements

These corrections must be made by the respective agents (tks-compiler, tks-quantum, tks-meta) or manually by an editor.

## Compilation

### Prerequisites

You need a LaTeX distribution installed. Recommended:
- **Windows**: MiKTeX (https://miktex.org/) or TeX Live
- **macOS**: MacTeX (https://tug.org/mactex/)
- **Linux**: TeX Live (`sudo apt install texlive-full`)

### Required Packages

The following packages must be available:
- amsmath, amssymb, amsthm, amsfonts
- mathtools, stmaryrd, bm, mathrsfs
- enumitem, booktabs, array, longtable, multirow, multicol
- graphicx, tikz, tikz-cd
- braket (quantum notation)
- bussproofs, proof
- wasysym, extarrows
- algorithm, algpseudocode (for algorithm floats)
- listings
- geometry, fancyhdr, titlesec
- tcolorbox
- xcolor
- hyperref, cleveref
- lmodern

### Compilation Commands

**Using pdflatex (recommended):**

```bash
cd C:\Users\wakil\downloads\everthing-tootra-tks

# First pass - generates aux files
pdflatex TKS_v7.3_PRINT.tex

# Second pass - resolves cross-references and TOC
pdflatex TKS_v7.3_PRINT.tex

# (Optional) Third pass - if references still show "??"
pdflatex TKS_v7.3_PRINT.tex
```

**Using nonstopmode (to generate partial output despite errors):**

```bash
pdflatex -interaction=nonstopmode TKS_v7.3_PRINT.tex
```

Note: This may produce a partial PDF with missing content where errors occurred.

**Using latexmk (automatic):**

```bash
latexmk -pdf TKS_v7.3_PRINT.tex
```

**Using XeLaTeX (for Unicode support):**

```bash
xelatex TKS_v7.3_PRINT.tex
xelatex TKS_v7.3_PRINT.tex
```

### Expected Output (after track files are fixed)

After successful compilation:
- `TKS_v7.3_PRINT.pdf` - The complete PDF document
- `TKS_v7.3_PRINT.aux` - Auxiliary file for cross-references
- `TKS_v7.3_PRINT.toc` - Table of contents data
- `TKS_v7.3_PRINT.log` - Compilation log

### Troubleshooting

#### Missing Package Errors

If you see errors like `! LaTeX Error: File 'stmaryrd.sty' not found.`:

**MiKTeX**: The package manager should prompt to install missing packages automatically. If not:
```
mpm --install=stmaryrd
```

**TeX Live**:
```bash
tlmgr install stmaryrd
```

#### Undefined Control Sequence

If you see errors about undefined commands, ensure you are compiling `TKS_v7.3_PRINT.tex` (not `TKS_v7.3_MASTER.tex`), as PRINT includes the complete preamble with all macro definitions.

#### BNF Style Not Found

The original `TKS_v7.3_MASTER.tex` does not define the `style=bnf` used in the Compiler track. The `TKS_v7.3_PRINT.tex` file includes this definition.

#### Algorithm Environment Errors

If you see errors like `! Argument of \@tempc has an extra }` within algorithm environments, this is a known issue in the track files where math-mode content inside `\Call{}` commands conflicts with the algpseudocode parser. These must be fixed in the source track files.

#### Very Long Compilation Time

The track files are large (80,000+ tokens each). Allow 2-5 minutes for compilation depending on your system.

## Version Differences

### TKS_v7.3_MASTER.tex vs TKS_v7.3_PRINT.tex

| Feature | MASTER | PRINT |
|---------|--------|-------|
| Compiles standalone | Possible issues | Designed for it |
| lstlisting BNF style | Missing | Included |
| algorithm/algpseudocode | Missing | Included |
| lmodern font | No | Yes |
| hyperref config | Basic | Enhanced |
| fancyhdr setup | No | Yes |

## Modular Architecture

The track files (`TKS_v7.3_Math.tex`, etc.) do NOT contain `\documentclass` or `\begin{document}`. They are designed to be included via `\input{}` and start directly with `\chapter{}` commands.

**Do not compile track files individually.** Always compile the MASTER or PRINT file.

## Preamble Features in TKS_v7.3_PRINT.tex

The PRINT file includes:

1. **Complete package set** for all track requirements
2. **Theorem environments**: definition, axiom, theorem, lemma, proposition, corollary, remark, example, notation
3. **TKS custom commands** from v6.1 through v7.3
4. **lstlisting styles**: `bnf` (for grammars), `tks` (for TKS code)
5. **Page layout**: letterpaper, fancy headers, proper margins
6. **Hyperref configuration**: colored links, PDF metadata

## Next Steps for Full Compilation

1. **Agent Review**: Have tks-compiler, tks-quantum, and tks-meta review their track files for LaTeX syntax
2. **Algorithm Fixes**: Replace problematic `\Call{}` with math-safe alternatives
3. **Diagram Fixes**: Ensure all tikz-cd nodes are properly defined
4. **Figure Placement**: Move figures outside tcolorbox environments or use appropriate float settings

## Contact

For questions about the TKS formalization, see the project documentation or contact the TKS Formalization Project team.
