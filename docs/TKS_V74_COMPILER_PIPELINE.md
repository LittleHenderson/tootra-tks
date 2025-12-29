# TKS v7.4 Compiler/VM Pipeline (Rust)

This document defines the implementation pipeline and crate layout for the v7.4 compiler/VM.

## Pipeline Stages
1. Lexing
   - Input: .tks source
   - Output: token stream with spans (file/line/col)
   - Unicode + ASCII supported (Unicode normalized to token kinds; ASCII default output).

2. Parsing
   - Input: tokens
   - Output: AST (v7.3 AST shape) with source locations
   - Parser supports:
     - v7.2 core grammar
     - v7.3 extensions (effects, transfinite, quantum)
     - v6.1 canonical aliases (expr^k, <<...>>)

3. Desugaring
   - Normalize aliases to canonical AST forms:
     - expr^k -> Noetic(k, expr)
     - <<k1:k2:...>>(expr) -> Fractal([k1,k2,...], expr)
     - Unicode fractal tokens -> ASCII fractal node

4. Name Resolution + Module Loading
   - Build module graph (imports, exports, signatures)
   - Load .tksi interfaces for dependencies
   - Resolve identifiers, handler names, effect names

5. Type + Effect Inference
   - Hindley-Milner inference with row-polymorphic effects
   - Validate effect declarations and handler clauses
   - Assign Type + EffectRow to each AST node

6. Lowering to ANF IR
   - Convert to effect-aware ANF IR
   - Make perform/handle explicit and ready for CPS lowering

7. Bytecode Generation
   - Map IR to v7.3 bytecode instruction set
   - Emit .tkso with metadata (module name, imports, exports, constants)

8. Linking / Execution
   - tksvm loads .tkso and resolves imports
   - Optional module linking step for single-file executables
   - Run on the v7.3 VM state model

## Crate Layout (Proposed)
- tks-rs/
  - Cargo.toml (workspace)
  - crates/
    - tkscore/ (lib: lexer, parser, AST, desugar, diagnostics)
    - tkstypes/ (lib: types, effects, unification, inference)
    - tksir/ (lib: IR, ANF transform, effect annotations)
    - tksbytecode/ (lib: bytecode model + encoder/decoder)
    - tksvm/ (lib: VM runtime, heap, handlers, ordinals, quantum)
    - tksc/ (bin: compiler CLI)
    - tks/ (bin: runner + REPL)

## CLI Surface (Draft)
- tksc build <file.tks> [-o out.tkso] [--emit ast|ir|bc]
- tksc check <file.tks> (parse + type + effect)
- tksc fmt <file.tks> (ASCII canonical output; optional)
- tks run <file.tks|file.tkso>
- tks repl

## Error Handling
- Diagnostics carry span + source line context.
- Errors classified: lexer, parse, resolve, type, effect, codegen, VM runtime.

## Test Strategy
- Lexer golden tests (tokens + spans)
- Parser golden tests (AST snapshots)
- Type/effect inference tests (rows, handlers, RPM)
- Bytecode roundtrip tests (encode/decode)
- VM behavioral tests (RPM, noetics, fractals, effects, quantum)

## Open Decisions
- Exact operator precedence table for v7.2 core + v7.3 extensions + ^ suffix.
- Default standard library surface (IO, file, time, math, RPM helpers).
- External FFI ABI subset for initial release.
