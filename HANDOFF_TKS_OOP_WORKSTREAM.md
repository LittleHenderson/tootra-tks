# TKS OOP Workstream Handoff

This handoff captures the current status and the next moves for the TKS OOP workstream so another LLM can pick up immediately.

## Repo and Environment

- Repo root: `C:\Users\wakil\downloads\everthing-tootra-tks`
- Main Rust workspace: `tks-rs`
- Default branch: `master`
- Latest pushed commit: `70d1f0e1` ("Align OOP AST with types and lowering")
- Sandbox: read-only by default (commands may need escalation for writes)

## Worktrees and Agents

Fresh worktrees were created for parallel agents:

- Agent A (lowering): `C:\Users\wakil\downloads\tks-agent-a-oop-lower`, branch `agent-a-oop-lower`
- Agent B (bytecode/VM): `C:\Users\wakil\downloads\tks-agent-b-oop-bytecode`, branch `agent-b-oop-bytecode`
- Agent C (docs/examples): `C:\Users\wakil\downloads\tks-agent-c-oop-docs`, branch `agent-c-oop-docs`

These are clean and based on `master` at `70d1f0e1`.

## What Was Just Fixed

OOP AST alignment across parser, type inference, and lowering:

- Class sections are `specifics`, `details`, `actions` and are required.
- Inside sections, the keywords are not used. Examples:
  - `specifics { x: Int; y: Int; }`
  - `details { current: Int = identity.x; }`
  - `actions { inc(self, delta: Int): Int = delta; }`
- Constructors use braces: `repeat ClassName { field: value, ... }`.
- `Expr::Member` uses the field name `field` in the AST.
- `Expr::Constructor` stores fields as `Vec<(Ident, Expr)>`.
- `Type::Class` is now formatted in `.tksi` output.
- Parser returns unknown type names as `Type::Var` (so aliases still resolve).
- `tkstypes` resolves a `Type::Var` to a class if it matches a class name in the environment.
- Member access can now constrain an unknown type to a class when the member name is unambiguous (tkstypes).

Tests run after these changes:

- `cargo test -p tkstypes -p tksir` (all passed)

Files touched by the last fix (now on master):

- `tks-rs/crates/tkscore/src/ast.rs`
- `tks-rs/crates/tkscore/src/parser.rs`
- `tks-rs/crates/tkscore/src/tksi.rs`
- `tks-rs/crates/tksir/src/lower.rs`
- `tks-rs/crates/tksir/tests/lower.rs`
- `tks-rs/crates/tkstypes/src/infer.rs`
- `tks-rs/crates/tkstypes/tests/infer_class.rs`
- `tks-rs/crates/tkstypes/tests/infer_constructor.rs`

## Current Gaps (Open Work)

### 1) OOP lowering to records (Agent A)

Goal: make class instances real runtime values using record ops.

Observations:

- IR already supports `Record`, `RecordGet`, `RecordSet` in `tks-rs/crates/tksir/src/ir.rs`.
- Bytecode emitter and VM already support record ops:
  - emitter: `tks-rs/crates/tksbytecode/src/emit.rs`
  - VM: `tks-rs/crates/tksvm/src/vm.rs`
  - tests: `tks-rs/crates/tksbytecode/tests/emit.rs`, `tks-rs/crates/tksvm/tests/vm.rs`
- Current lowering still creates constructor lambdas that return `Unit` and member access only works for `self`/`identity` inside class scope.

Recommended approach:

- Add a class decl map to `LowerState` (name -> ClassDecl).
- Pre-scan `program.decls` in `lower_program` to populate class map before lowering the entry.
- Update `lower_constructor_term` so `repeat Class { ... }`:
  1. Creates a record with field values.
  2. Binds `self` to that record.
  3. For each property, compute `Class::property self` and `RecordSet` it.
  4. For each method, compute partial application `Class::method self` (closure with self captured) and `RecordSet` it.
  5. Return the record.
- Update `lower_member_term` to lower any member access `obj.field` to `IRTerm::RecordGet`, not just `self`.

Files likely to change:

- `tks-rs/crates/tksir/src/lower.rs`
- `tks-rs/crates/tksir/tests/lower.rs` (add tests for record-based member access)

Tests to run:

- `cargo test -p tksir`

### 2) Bytecode/VM integration checks (Agent B)

Even though bytecode/VM already have record op support, add integration tests that use the new lowering path:

- End-to-end pipeline: class constructor -> record -> member get/set in VM.
- Add `tksbytecode` emitter tests for constructor/member lowering if needed.
- Add `tksvm` tests that execute RecordGet/RecordSet in a class-like program.

Files likely to change:

- `tks-rs/crates/tksbytecode/tests/emit.rs`
- `tks-rs/crates/tksvm/tests/vm.rs`

Tests to run:

- `cargo test -p tksbytecode -p tksvm`

### 3) Docs and examples (Agent C)

Make sure user-facing docs match the exact class syntax and constructor syntax:

- Update `docs/TKS_FOR_DUMMIES.md`
- Update `docs/TKS_OOP_HYBRID.md`
- Add example `tks-rs/examples/oop_counter.tks` with:
  - class `Counter` using specifics/details/actions
  - constructor `repeat Counter { value: 1 }`
  - member access for field, property, and method
- Include a short CLI run snippet in docs:
  - `.\dist\tks-0.1.0-windows\tksc.exe build --emit bc -o .\counter.tkso .\tks-rs\examples\oop_counter.tks`
  - `.\dist\tks-0.1.0-windows\tks.exe run .\counter.tkso`

Tests to run:

- None required (docs/examples only)

## Syntax Reminders (Do Not Drift)

- Class declaration:
  - `class Name { specifics { ... } details { ... } actions { ... } }`
  - `specifics`: field declarations (no `field` keyword)
  - `details`: property declarations (no `property` keyword)
  - `actions`: method declarations (no `method` keyword)
  - Method params require explicit `self` or `identity` as first param.
- Constructor:
  - `repeat Name { field: expr, ... }`
  - braces, not parentheses
- Member access:
  - `obj.field`
  - `obj.method(arg)` parses as member + app.

## Testing Commands

From `tks-rs`:

- `cargo test -p tkstypes -p tksir` (already passing on master)
- `cargo test -p tksbytecode -p tksvm`
- `cargo test --workspace` (optional full sweep)

## Important Caution

The repo root currently has many unrelated modified/untracked files (training, datasets, PDFs). Do not clean or revert them unless explicitly asked. Keep changes scoped to `tks-rs`.

