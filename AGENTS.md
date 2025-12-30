# TKS v7.4 Compiler/VM Agent Plan

This file defines the agent roster, roles, and work division for the TKS v7.4
compiler/VM effort. Canonical authority is v7.4; changes must remain 90-100%
true to canonical TKS. Rust is the implementation language. ASCII is the
default output unless Unicode is required.

## Roles

### Supervisor (S)
- Owns integration decisions and task assignment.
- Resolves cross-agent conflicts and tracks dependencies.
- Reviews interim results weekly and at each milestone.

### Planner (P)
- Maintains the workflow, milestones, and dependency graph.
- Coordinates handoffs between workers and the supervisor.
- Updates the roadmap based on canonical constraints.

### Quality Inspector (QI)
- Performs intermittent audits and final reviews per milestone.
- Verifies canonical v7.4 alignment and no grammar drift.
- Checks tests, error messaging, and API consistency.

## Workers and Assignments

### Worker A: Front-End Semantics
- Scope: type/effect inference coverage.
- Tasks:
  - Add typing for ACBE, quantum, and transfinite constructs.
  - Extend handler/effect coverage for edge cases (row vars, open rows).
  - Add negative tests for type/effect errors.
- Deliverables: updated `tkstypes` inference + tests.

### Worker B: Name Resolution + Modules
- Scope: module, import/export, and type alias resolution.
- Tasks:
  - Implement TypeDecl support (aliases, params).
  - Implement ModuleDecl resolution and name scoping.
  - Add tests for module boundaries and aliasing.
- Deliverables: resolver layer + tests, wired into `tksc`.

### Worker C: AST -> IR Lowering
- Scope: lowering pipeline and IR validation.
- Tasks:
  - Implement lowering from AST to `tksir` for core + effects/handlers.
  - Add IR validation utilities (well-formedness checks).
  - Wire `tksc build --emit ir`.
- Deliverables: lowering module + tests + CLI integration.

### Worker D: Bytecode + VM
- Scope: bytecode emission and runtime execution.
- Tasks:
  - Define bytecode emission from IR (effects/handlers included).
  - Implement VM runtime semantics (handlers, RPM, ordinals, quantum stubs).
  - Wire `tksc build --emit bc` and `tks run`.
- Deliverables: `tksbytecode` emitter, VM runtime, tests.

### Worker E: Stdlib/FFI + Packaging
- Scope: externs, standard library, packaging.
- Tasks:
  - Formalize extern/FFI type+effect mapping.
  - Define minimal stdlib for canonical TKS primitives.
  - Package Windows `.exe` runtime and CLI artifacts.
- Deliverables: stdlib crate, FFI bridge, packaging scripts.

## Milestones (Supervisor + Planner)

1) Front-end completeness
   - All TKS v7.4 syntax type-checked.
   - Module/type alias resolution implemented.

2) IR pipeline
   - AST -> IR lowering for core/effects/handlers.
   - `tksc build --emit ir` operational.

3) Bytecode + VM
   - IR -> bytecode and VM execution for core constructs.
   - `tksc build --emit bc` and `tks run` operational.

4) Runtime packaging + compliance
   - Windows `.exe` packaging.
   - Canonical compliance checks and regression tests.

## Review Cadence

- Interim reviews at each milestone checkpoint.
- Final review at milestone completion.

## What's Next (Tracking)

Status legend: [ ] todo, [~] in progress, [x] done.

[x] Effects & handlers end-to-end (type/effect rows + IR lowering + bytecode/VM) - Owner: Agent A
[x] Modules + .tksi + resolver + import/export + FFI signatures/links - Owner: Agent B
[x] Runtime packaging (.exe) + CLI polish + build scripts + GPU harness gating - Owner: Agent C
[x] Fractal/foundation/acquisition semantics in VM + tests - Owner: Agent A
[x] Stdlib bootstrap (TKS.Core, TKS.RPM, TKS.Quantum) - Owner: Agent B
[x] Canonical compliance + regression tests (samples, golden files) - Owner: Agent C
[~] FFI runtime binding (extern values + bytecode + VM + CLI registry) - Owner: Agent A/B/C

## Current Sprint (Parallel)

Status legend: [ ] todo, [~] in progress, [x] done.

[x] Agent A: Extern value plumbing (IR + lowering + emitter + bytecode + tests)
[x] Agent B: VM extern call semantics + registry API + tests
[~] Agent C: CLI host registry + builtins + docs/examples
