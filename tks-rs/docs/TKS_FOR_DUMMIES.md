# TKS for Dummies

A beginner-friendly guide to the TKS (Tootra Kabbalistic System) programming language.

## What is TKS?

TKS is a programming language that combines functional programming with noetic (consciousness-related) operators, ordinal arithmetic, quantum semantics, and object-oriented constructs. It is designed to express computations that span multiple "worlds" and incorporate effects and handlers.

## Getting Started

### Running a Program

```bash
# Run a TKS source file
tks run program.tks

# Run with FFI support (print_int, print_bool, etc.)
tks run --ffi program.tks
```

### Compiling a Program

```bash
# Type-check a program
tksc check program.tks

# Build and output bytecode
tksc build --emit bc -o program.tkso program.tks

# Run the compiled bytecode
tks run program.tkso
```

## Basic Syntax

### Variables and Let Bindings

```tks
let x = 42;
let y: Int = x;
```

### Functions (Lambdas)

```tks
let id = \a -> a;
let result = id 42;  -- result is 42
```

### Conditionals

```tks
if true then 1 else 0
```

> **Note:** Arithmetic operators (`+`, `-`, `*`, `/`) are currently supported only for ordinal arithmetic. Integer arithmetic via FFI or future parser extensions is planned.

## Object-Oriented Programming in TKS

TKS provides OOP constructs with domain-specific terminology.

### Class Declaration

A class is declared using `class`, `blueprint`, or `plan` (all equivalent). The class body has three required sections:

1. **`specifics`** - Instance fields (data stored per object)
2. **`details`** - Computed properties (derived from fields)
3. **`actions`** - Methods (behaviors)

```tks
class Counter {
  specifics {
    value: Int;
  }
  details {
    current: Int = identity.value;
  }
  actions {
    inc(self, delta: Int): Int = delta;
    reset(self): Int = 0;
  }
}
```

> **Note:** Full arithmetic in property/method bodies requires integer arithmetic operator support (planned). Currently, property expressions can reference fields but not perform arithmetic.

### Self Reference

Inside a class, use `identity` or `self` to refer to the current instance:

- In `details`, use `identity.fieldname` to access fields
- In `actions`, the first parameter must be `self` or `identity`

### Creating Instances (Constructors)

Use `repeat` (or `new`) followed by the class name and field initializers in braces:

```tks
let c = repeat Counter { value: 1 };
```

### Accessing Members

```tks
-- Access a field
c.value

-- Access a computed property
c.doubled

-- Call a method
c.inc(5)
```

## Noetic Operators

Noetic operators (indices 0-21) represent consciousness transformations:

```tks
-- Apply noetic operator 5 to value x
noetic(5, x)
```

## Ordinals

TKS supports transfinite ordinals:

```tks
omega           -- first infinite ordinal
omega + 1       -- successor of omega
epsilon_0       -- fixed point of omega^x = x
aleph_0         -- first infinite cardinal
```

## Quantum Forms

Quantum constructs for superposition and measurement:

```tks
-- Create a ket
|0>
|1>

-- Superposition
superpose(|0>, |1>)

-- Measurement
measure(superposed_state)
```

## Effects and Handlers

Algebraic effects for managing side effects:

```tks
effect State[S] {
  op get(): S;
  op put(s: S): Unit;
}

handler stateHandler: State[Int] {
  return x -> x
  get() k -> \s -> k(s)(s)
  put(s) k -> \_ -> k(())(s)
}
```

## RPM (Result/Promise Monad)

For success/failure computation:

```tks
-- Success value
rpm_win(42)

-- Failure value
rpm_fail("error")

-- Bind/chain computations
rpm_bind(computation, \x -> rpm_win(x + 1))
```

## Example: Complete Counter Program

```tks
class Counter {
  specifics {
    value: Int;
  }
  details {
    doubled: Int = identity.value;
  }
  actions {
    inc(self, delta: Int): Int = delta;
    reset(self): Int = 0;
  }
}

let c = repeat Counter { value: 1 };
c.doubled
```

To run this example:

```bash
tks run examples/oop_counter.tks
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `tks run FILE` | Run a TKS source file or bytecode |
| `tks run --ffi FILE` | Run with FFI support |
| `tksc check FILE` | Type-check a program |
| `tksc build FILE --emit ast` | Output AST |
| `tksc build FILE --emit ir` | Output IR |
| `tksc build FILE --emit bc` | Output bytecode |
| `tksc build FILE -o OUT` | Compile to binary |

## Next Steps

- See `TKS_OOP_HYBRID.md` for advanced OOP patterns
- Explore `examples/` directory for more sample programs
- Run `cargo test --workspace` to see the test suite
