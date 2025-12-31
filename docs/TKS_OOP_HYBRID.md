# Hybrid OOP for TKS (Draft)

This document captures the agreed hybrid OOP layer for TKS. It is a surface
syntax that lowers into the existing core (functions, values, effects).

## Keyword Map
- `blueprint` / `plan` = class
- `specifics` / `description` = fields (stored data)
- `details` = properties (computed accessors)
- `actions` = methods
- `identity` = self
- `repeat` = new

Optional aliases (if you want standard keywords too):
- `class`, `field`, `method`, `new`, `self`

## Example (Conceptual)
```tks
blueprint Counter {
  specifics {
    value: Ordinal;
  }

  details {
    current = identity.value;
  }

  actions {
    inc(identity): Counter = repeat Counter(value: succ(identity.value));
  }
}
```

## Semantics (Lowering Sketch)
- `blueprint` defines a record type + constructor function.
- `specifics` become stored fields in the record.
- `details` become computed accessors (zero-arg methods) with implicit
  `identity`.
- `actions` become functions where `identity` is the first parameter.
- `repeat T(...)` becomes a constructor call.
- `identity` is an alias for `self`.

Conceptual lowering:
```tks
type Counter = { value: Ordinal };
let Counter = \value -> { value = value };
let Counter.current = \self -> self.value;
let Counter.inc = \self -> Counter(succ(self.value));
```

## Mutability
Default: immutable fields. Updates return a new instance.
If mutability is needed later, introduce `mut` fields explicitly.

## Inheritance
Default: none (composition + interfaces/traits if added later).
If inheritance is required, define the dispatch rules explicitly.

## Implementation Steps (Planned)
1) Lexer: add new keywords (`blueprint`, `plan`, `specifics`, `details`,
   `actions`, `identity`, `repeat`, plus optional standard aliases).
2) Parser: parse class blocks into new AST nodes.
3) Type checker: build record types + method signatures.
4) Lowering: desugar into existing core (records + functions).
5) Bytecode/VM: no new ops required if lowering is correct.

## Status
Draft. Syntax is not implemented yet.
