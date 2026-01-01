# TKS OOP Hybrid Model

This document describes the object-oriented programming model in TKS, which combines functional programming with class-based constructs using domain-specific terminology.

## Design Philosophy

TKS OOP is a hybrid model:
- Classes define structure (fields, properties, methods)
- Instances are runtime records with computed properties and method closures
- Functional style is preserved (immutable by default, expression-oriented)

## Class Declaration Syntax

A class is declared using one of three equivalent keywords: `class`, `blueprint`, or `plan`.

### Grammar

```bnf
ClassDecl ::= ("class" | "blueprint" | "plan") Ident "{"
              SpecificsSection
              DetailsSection
              ActionsSection
            "}"

SpecificsSection ::= "specifics" "{" FieldDecl* "}"
DetailsSection   ::= "details" "{" PropertyDecl* "}"
ActionsSection   ::= "actions" "{" MethodDecl* "}"

FieldDecl    ::= Ident ":" Type ";"
PropertyDecl ::= Ident ":" Type "=" Expr ";"
MethodDecl   ::= Ident "(" SelfParam ("," Param)* ")" ":" Type "=" Expr ";"

SelfParam ::= "self" | "identity"
Param     ::= Ident ":" Type
```

### Section Semantics

| Section | Keyword | Contains | Description |
|---------|---------|----------|-------------|
| Fields | `specifics` | `name: Type;` | Instance data, provided at construction |
| Properties | `details` | `name: Type = expr;` | Computed from fields, uses `identity` |
| Methods | `actions` | `name(self, ...): Type = expr;` | Behaviors, first param is `self` |

## Complete Example

```tks
class Counter {
  specifics {
    value: Int;
    step: Int;
  }
  details {
    current: Int = identity.value;
    current_step: Int = identity.step;
  }
  actions {
    inc(self, delta: Int): Int = delta;
    dec(self, delta: Int): Int = delta;
    reset(self): Int = 0;
    get_step(self): Int = self.step;
  }
}
```

> **Note:** Integer arithmetic operators (`+`, `-`, `*`, `/`) are not yet implemented for expressions. The examples above show field access patterns. Full arithmetic support is planned.

## Constructor Syntax

Instances are created using `repeat` (or `new`) with brace-enclosed field initializers:

```tks
let counter = repeat Counter { value: 10, step: 5 };
```

### Grammar

```bnf
Constructor ::= ("repeat" | "new") ClassName "{" FieldInit ("," FieldInit)* "}"
FieldInit   ::= Ident ":" Expr
```

### Constructor Semantics

When `repeat Counter { value: 10, step: 5 }` executes:

1. A record is created with the specified field values
2. Properties are computed and added to the record
3. Methods are bound with `self` captured as closures
4. The resulting record is returned as the instance

## Member Access

### Fields

Direct access to stored data:

```tks
counter.value    -- returns 10
counter.step     -- returns 5
```

### Properties

Access to computed values (evaluated at construction time):

```tks
counter.current       -- returns 10
counter.current_step  -- returns 5
```

### Methods

Method calls pass the instance as the first argument:

```tks
counter.inc(3)      -- returns 3 (the delta argument)
counter.dec(2)      -- returns 2 (the delta argument)
counter.reset()     -- returns 0
counter.get_step()  -- returns 5 (the step field)
```

## Self References

Inside class definitions, use `identity` or `self` to refer to the instance:

| Context | Keyword | Usage |
|---------|---------|-------|
| Properties (`details`) | `identity` | `identity.field` |
| Methods (`actions`) | `self` | First parameter, then `self.field` |

Both `identity` and `self` are interchangeable in the parser, but by convention:
- `identity` is used in property expressions
- `self` is used as the method receiver parameter

## Runtime Representation

At runtime, class instances are represented as records:

```
Record {
  "value" -> 10,
  "step" -> 5,
  "current" -> 10,
  "current_step" -> 5,
  "inc" -> Closure(self, delta -> delta),
  "dec" -> Closure(self, delta -> delta),
  "reset" -> Closure(self -> 0),
  "get_step" -> Closure(self -> self.step)
}
```

Member access is lowered to `RecordGet` operations in the IR.

## Type Inference

The type system infers `Type::Class(name)` for class instances:

```tks
let c = repeat Counter { value: 1, step: 1 };
-- c has type: Counter
```

Member access on an unknown type can constrain it to a class if the member name is unambiguous in the class registry.

## Comparison with Traditional OOP

| Traditional | TKS | Notes |
|-------------|-----|-------|
| `class Foo` | `class Foo`, `blueprint Foo`, `plan Foo` | Three synonyms |
| `private int x;` | `specifics { x: Int; }` | Fields section |
| `get doubled() { return x*2; }` | `details { doubled: Int = identity.x; }` | Computed property (arithmetic planned) |
| `void inc(int d) { ... }` | `actions { inc(self, d: Int): Int = ...; }` | Method section |
| `new Foo(1)` | `repeat Foo { x: 1 }` | Brace syntax |
| `this.x` | `self.x`, `identity.x` | Two synonyms |

## Example: CLI Execution

Save the following as `examples/oop_counter.tks`:

```tks
class Counter {
  specifics { value: Int; }
  details { doubled: Int = identity.value; }
  actions {
    inc(self, delta: Int): Int = delta;
    reset(self): Int = 0;
  }
}

let c = repeat Counter { value: 1 };
c.doubled
```

Build and run:

```bash
# From repo root
.\dist\tks-0.1.0-windows\tksc.exe build --emit bc -o counter.tkso tks-rs\examples\oop_counter.tks
.\dist\tks-0.1.0-windows\tks.exe run counter.tkso

# Or directly
.\dist\tks-0.1.0-windows\tks.exe run tks-rs\examples\oop_counter.tks
```

Expected output: `1` (the value field)

## Integration with Effects

Classes can work with effect handlers (conceptual example - string concat `++` and comparison operators are parser extensions):

```tks
effect Log {
  op log(msg: String): Unit;
}

class Logger {
  specifics { prefix: String; }
  details { }
  actions {
    log_prefix(self): String = self.prefix;
  }
}
```

## Integration with RPM

Methods can return RPM values (conceptual example - division operator is a planned extension):

```tks
class Wrapper {
  specifics { value: Int; }
  details { }
  actions {
    wrap(self): RPM[Int] = rpm_win(self.value);
    fail_msg(self): RPM[Int] = rpm_fail("error");
  }
}
```

## Future Extensions

Planned enhancements to the OOP model:

- **Integer arithmetic**: `+`, `-`, `*`, `/` operators for Int expressions
- **Comparison operators**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **String concatenation**: `++` operator
- **Inheritance**: `class Child extends Parent { ... }`
- **Traits**: `trait Comparable { ... }`
- **Visibility**: `private`, `public` modifiers
- **Mutable fields**: `var` keyword for mutable specifics

## Summary

TKS OOP provides:

1. Class declarations with `class`/`blueprint`/`plan`
2. Three sections: `specifics` (fields), `details` (properties), `actions` (methods)
3. Constructor syntax: `repeat ClassName { field: value, ... }`
4. Member access: `obj.field`, `obj.property`, `obj.method(args)`
5. Self reference: `identity` in properties, `self` in methods
6. Runtime record-based representation
7. Integration with effects, RPM, and other TKS features
