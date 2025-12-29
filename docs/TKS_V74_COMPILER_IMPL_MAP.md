# TKS v7.4 Compiler/VM Implementation Map (Draft)

Scope:
- Canonical authority: v7.4 track files (CoreCalculus, Semantics).
- Concrete language/VM spec: v7.3 Compiler track (syntax/AST/IR/bytecode/VM).
- v6.1 manual used for notation compatibility (noetic ^, fractal << >>, Unicode fractal).

Goals:
- Implement full v7.4 compiler/VM in Rust.
- Static types + effect rows/handlers (no dynamic typing).
- ASCII output defaults; accept ASCII and Unicode tokens.

## Canonical Sources
- TKS_v7.4_CoreCalculus.tex: core calculus, RPM monad, fractals, noetics, foundations.
- TKS_v7.4_Semantics.tex: domain-theoretic semantics (Scott domains, fixpoints).
- TKS_v7.3_Compiler.tex: surface syntax, AST, type system, IR, bytecode, VM, modules, FFI.
- TKS_FORMAL_MANUAL_v6.1_CLEAN_DEFINITIONS.md: notation alignment (noetic superscripts, fractal Unicode/ASCII).

## Lexical Tokens (v7.2 + v7.3)
- v7.2 core tokens: ELEMENT, NOETIC, FOUNDATION, INT, BOOL, IDENT, LAMBDA, LET, IN, IF, THEN, ELSE,
  RETURN, BIND, CHECK, ACQUIRE, ACBE, FRAC_OPEN/CLOSE, COLON, LPAREN/RPAREN, ARROW, EQUALS,
  PLUS, MINUS, TIMES, DIVIDE, EOF.
- v7.3 additions: OMEGA, ORD, LIMIT, SUCC, TRANSFINITE, EFFECT, HANDLE, WITH, RESUME, PERFORM,
  HANDLER, OP, MEASURE, SUPERPOSE, ENTANGLE, QSTATE, AMPLITUDE, ORDINAL_LIT(alpha),
  LBRACE, RBRACE, PIPE, BANG, UNDERSCORE_OMEGA.
- Ordinal literals: omega, omega + n, omega * n, omega ^ n, epsilon_n, aleph_n (per compiler track).

## Concrete Syntax
- v7.2 core grammar summary (compiler track).
- v7.3 extensions: effect declarations, handler declarations, handle/perform expressions,
  transfinite fractals, ordinal expressions, transfinite loops, quantum operations.
- Compatibility extensions (canonical notation from v6.1):
  - Support noetic application suffix: expr^k (alias for nu k (expr)).
  - Accept fractal ASCII: <<k1:k2:...>> (alias for <k1:k2:...>).
  - Accept fractal Unicode: \u{0192}Y"1:4:7\u{0192}Yc (alias).
  - ASCII output defaults to ^ and << >>.

## AST (v7.3)
- TopDecl: LetDecl, TypeDecl, EffectDecl, HandlerDecl.
- Expr: v7.2 forms + Handle, Perform, Ordinal forms, Quantum forms.
- Handler: named handler ref or inline handler definition.
- Type: v7.2 types + effectful types, handler type, ordinal type, QState type.
- EffectRow: empty, cons, or row variable (row polymorphism).
- All nodes carry source locations.

## Type System
- Hindley-Milner inference + unification (v7.3 compiler track).
- Types: Int, Bool, Unit, Void, Element[W], Foundation, Domain, Frac[k], RPM[t],
  function, product, sum.
- Effectful types: tau ! epsilon, tau1 -(epsilon)-> tau2, Handler[E, tau_in, tau_out].
- Effect rows: {} | {E | r}, with row unification (duplicate effects invalid).
- Type-level ordinals: 0.., omega, omega_1, succ, +, *, ^.
- Quantum types: QState[t], ket/bra are values (per compiler track).

## Core Semantics (v7.4)
- Noetics: 0-9 operators (monoid; 0 is identity).
- Foundations: 7 foundations, 28 subfoundations.
- RPM monad: return/bind/check/acquire semantics.
- Fractals: sequence of noetics, iteration, fixpoint (Kleene).
- Map semantics to VM instructions for noetic application, fractals, RPM effects.

## IR (effect-aware ANF)
- IRVal: var, literal, lambda, element, noetic, ordinal, ket.
- IRTerm: return, let, app, if, perform, handle, rpm ops, ordinal ops, quantum ops.
- ANF transformation per compiler track.

## Bytecode + VM
- Instruction set groups: core stack ops, TKS domain ops, RPM ops, effect ops,
  transfinite ops, quantum ops.
- VM state: (code, pc, stack, env, frames, handlers, heap, acquisition alpha,
  ordinal register, qreg).
- Heap objects: closures, ideas, fractals, handlers, continuations, qstate.

## Module System + Artifacts
- Source: .tks
- Interface: .tksi
- Object: .tkso
- Module syntax: module, export, import (qualified/aliased/selective/wildcard).

## FFI
- external "C"/"stdcall"/"fastcall"/"system" fn ...
- safe/unsafe annotations; effect annotations on externs.

## Open Questions / Spec Gaps
- Precedence/associativity for ^ suffix vs nu form (define in parser).
- Ordinal literal surface syntax beyond omega (epsilon/aleph) in v7.4 core.
- Mapping of v7.4 fixpoint/fractal semantics to VM execution model.
- Default standard library and effect set (IO, RPM, quantum, filesystem).
