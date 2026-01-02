# TKS - The Tootra Kabbalistic System

A Rust implementation of the TKS language, featuring noetics, ordinals, quantum forms, effects/handlers, RPM, foundations, fractals, and OOP constructs.

## Building

```bash
cd tks-rs
cargo build --release
```

This produces two binaries:
- **`tks`** - Runtime for executing TKS programs
- **`tksc`** - Compiler for checking and building TKS programs

## Usage

### Running Programs

```bash
# Run a TKS source file
tks run program.tks

# Run with FFI support (print_int, print_bool)
tks run --ffi program.tks

# Run precompiled bytecode
tks run program.tkso
```

### Compiling Programs

```bash
# Type-check a program
tksc check program.tks

# Build and output AST
tksc build program.tks --emit ast

# Build and output IR
tksc build program.tks --emit ir

# Build and output bytecode
tksc build program.tks --emit bc

# Compile to binary format
tksc build program.tks -o program.tkso
```

## Language Features

### Core DSL

| Feature | Description |
|---------|-------------|
| Noetics | Consciousness operators (indices 0-21) |
| Ordinals | Transfinite numbers (omega, epsilon, aleph) |
| Quantum | Kets, superposition, measurement, entanglement |
| Effects | Algebraic effects with handlers |
| RPM | Result/Promise monad for success/failure |
| Foundations | Hierarchical type levels |
| Fractals | Self-similar structures |

### OOP Constructs

TKS supports object-oriented programming with these keywords:

| TKS Keyword | Traditional | Description |
|-------------|-------------|-------------|
| `blueprint` / `plan` / `class` | class | Define a type |
| `specifics` / `description` | fields | Instance data |
| `details` | properties | Computed values |
| `actions` | methods | Behaviors |
| `identity` / `self` | this/self | Self reference |
| `repeat` / `new` | new | Instantiation |

**Example:**

```tks
blueprint Counter {
  specifics { value: Int; }
  details { doubled: Int = identity.value; }
  actions { }
}

repeat Counter { value: 42 }
```

## Examples

Run the included examples:

```bash
# Quantum ordinals
tks run examples/quantum_ordinal.tks

# OOP blueprint
tks run examples/oop_counter.tks

# Stdlib demos
tks run examples/stdlib_quantum.tks
tks run examples/stdlib_rpm.tks
tks run examples/stdlib_noetics.tks
tks run examples/stdlib_foundations.tks
tks run examples/stdlib_fractals.tks

# FFI example
tks run --ffi examples/ffi_print_int.tks
```

## Testing

```bash
cargo test --workspace
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `tkscore` | Parser, AST, lexer |
| `tkstypes` | Type inference |
| `tksir` | Intermediate representation |
| `tksbytecode` | Bytecode emission |
| `tksvm` | Virtual machine |
| `tksc` | Compiler CLI |
| `tks` | Runtime CLI |
| `tksgpu` | GPU acceleration (optional) |

## GPU Support

Build with GPU support:

```bash
cargo build --release --features gpu
tks gpu info
```
