# TKS Language Manual (For First-Time Users)

This is a practical "how to use it" guide for the current Rust implementation of
the TKS v7.4 language. It focuses on running code and the real syntax that is
implemented in this repo.

## What This Is (Purpose + Origin)
- TKS is a general-purpose programming language grounded in the Tootra
  Kabbalistic System (TKS) canonical material (v7.4).
- It treats TKS constructs (Noetics, Elements, Foundations, RPM, Fractals,
  Quantum forms) as first-class values.
- Canonical references in this repo:
  - `TKS_v7.4_MASTER.pdf`
  - `TKS_FORMAL_MANUAL_v6.1_CLEAN_DEFINITIONS.pdf`

This guide does not redefine metaphysics; it shows how to run code.

## One-Page Concept Map (Metaphor + Use + Professions)
- Elements: stickers for ideas; tagging and grouping; therapy, coaching, education, business ops, data labeling.
- Foundations: the floor you stand on; stable context; therapy, coaching, education, business ops, systems design.
- Noetics: a lens that reframes; perspective shifts; therapy, coaching, education, cognitive modeling, UX research.
- RPM: a quest log; staged progress; coaching, education, business ops, workflow automation, process ops.
- Ordinals: steps past all finite steps; long-horizon ranking; formal methods, theorem proving, theoretical CS, education research, business ops.
- Limit: end-of-process summary; convergence; therapy, coaching, education, business ops, systems modeling.
- Fractals: repeating stamp patterns; self-similar structure; therapy, coaching, education, business ops, generative design.
- Transfinite loops: all steps + a limit step; infinite process rules; therapy, coaching, education, business ops, formal verification.
- Quantum forms: many possibilities at once; scenario exploration; decision science, quantum computing, therapy, coaching, business ops.
- Effects + handlers: ring a bell, decide response; structured interactions; backend engineering, platform tooling, business ops, education, therapy/coaching.
- Externs (FFI): phone call outside; native integration; systems programming, DevOps, business ops, education, therapy/coaching.
- Modules: folders for ideas; organization; software engineering, education, business ops, therapy/coaching.
- ACBE: goal-checker; scoring alignment; evaluation, coaching, education, business ops, therapy.

## Tools You Use (What Each Component Does)
- `tks.exe`: runner/VM. Runs `.tks` source or `.tkso` bytecode.
- `tksc.exe`: compiler. `check` and `build` with `--emit ast|ir|bc|tksi`.
- `.tks`: source file.
- `.tkso`: bytecode output (runnable by `tks.exe`).
- `.tksi`: interface signature output (for modules).
- Stdlib location: `tks-rs/stdlib` (loaded by `tksc`).

## Step-by-Step: Run Your Own Code (CPU Build)
1) Go to the repo:
```powershell
cd C:\Users\wakil\downloads\everthing-tootra-tks
```

2) Build and stage the binaries (CPU):
```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release
```

3) Create a file to play with:
```powershell
@'
let idea = A1;
idea^2
'@ | Set-Content -NoNewline .\play.tks
```

4) Run it:
```powershell
.\dist\tks-0.1.0-windows\tks.exe run .\play.tks
```

## Step-by-Step: GPU Build (Optional)
```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release -GpuOnly
.\dist\tks-0.1.0-windows-gpu\tks.exe gpu info
```

## Why GPU (Short Explanation)
TKS work can be highly parallel (fractals, quantum state sets, large ordinal
iterations). GPUs accelerate these workloads by running many small operations
at once. Today the GPU build exposes basic GPU commands (`tks gpu info`,
`tks gpu add`). As fractal/quantum kernels land, the same build will be used
to accelerate those transforms.

## Step-by-Step: Compile to Bytecode
```powershell
.\dist\tks-0.1.0-windows\tksc.exe build --emit bc -o .\play.tkso .\play.tks
.\dist\tks-0.1.0-windows\tks.exe run .\play.tkso
```

## Step-by-Step: Printing (FFI)
```powershell
@'
extern c safe fn print_int(x: Int): Unit !{IO};
print_int(42)
'@ | Set-Content -NoNewline .\play.tks

.\dist\tks-0.1.0-windows\tks.exe run --ffi .\play.tks
```

Built-in externs: `print_int`, `print_bool`.

## First Program (Empowerment Style, Working Today)
This example runs on the current runtime. We treat an Element as a symbolic
stand-in for an idea and apply a noetic operator to reframe it.

```tks
-- idea: "I am strong" (symbolic label)
let idea = A1;
idea^2
```

Run it:
```powershell
@'
-- idea: "I am strong" (symbolic label)
let idea = A1;
idea^2
'@ | Set-Content -NoNewline .\affirm.tks

.\dist\tks-0.1.0-windows\tks.exe run .\affirm.tks
```

Why hook (interpretive): noetic operators (nu0-nu9) transform a value. In a
personal-growth frame, that is like re-framing the same idea in a new light.

### Conceptual (Book-Inspired, Not Implemented Yet)
The book discusses inversion axes and affirmations. The current language does
not include strings or an `invert(...)` builtin, so this is pseudocode only:

```tks
-- pseudocode only (not valid in current runtime)
invert("I am strong", axes: N,E)
```

Conceptual output: "You are weak" (noetic inversion flips polarity; the axis
shift moves the subject perspective). This is illustrative only.

If you want this today, implement an extern and map text to symbols (Elements or
Noetics) on the host side.

## Language Outline (Syntax You Can Use)

### Comments
Line comments start with `--`:
```tks
-- this is a comment
```

### Core Expressions
```tks
let x = 1;          -- top-level binding
let x = A1 in x^1   -- let expression
\x -> x             -- lambda
f x                 -- application (left-assoc)
if cond then a else b
```

### Literals
```tks
42        -- Int
true      -- Bool
()        -- Unit
3.14      -- Float (Domain)
2.0i      -- Complex (Domain)
(1.0, 2.0) -- Complex (Domain)
```

### Elements (40)
Elements are `A1`..`D10`:
```tks
A1
C7
```
Why hook: Elements are symbolic anchors. In a personal-growth frame, they let
you tag an idea with a stable archetype before transforming it.

### Foundations (7 x 4)
Foundations are `1a`..`7d` (prefix `F` also works):
```tks
1a
4d
F3b
```
Why hook: Foundations are stable contexts. They represent the "ground" an idea
stands on before you apply noetic change.

### Noetics (0-9)
Suffix and prefix forms:
```tks
3^1
nu2(7)
nu3 10
```
Why hook: Noetics (nu0-nu9) reframe a value. They are the core "transform
ideas" operators in TKS.

### Fractals
ASCII forms:
```tks
<1>(x)
<1:2>(x)
<<1:2:...>>_omega(x)
```

Unicode fractal delimiters:
```tks
ƒY"1:4:7ƒYc(x)
```

Notes:
- Digits are 0-9.
- `...` marks ellipsis.
- `_omega` (or any ordinal expression) is optional.

Why hook: Fractals repeat a transform across scales, which mirrors how patterns
recur in beliefs and behaviors.

### Ordinals
```tks
omega + 2
epsilon_0
aleph_1
succ(omega)
limit(k < omega . k)
```

### Quantum Forms
```tks
|10>                   -- ket
<10|                   -- bra
<A1|A2>                -- bra-ket
superpose { 1: |10>, 2: |20> }
superpose[(0.5, |A1>), (0.5, |A2>)]
measure(superpose { 1: |10>, 2: |20> })
entangle(|1>, |2>)
```
Why hook: Quantum forms let you hold multiple possibilities, then measure to
choose a realized path. This mirrors exploring alternate perspectives.

### RPM (Rule of Progressive Manifestation)
```tks
return 5
check 3
acquire 7
(return 1) >>= (\x -> return x)
```
Why hook: RPM encodes progress and acquisition steps, which makes growth
stages explicit and testable.

### Effects + Handlers
```tks
effect Log {
  op log(msg: Int): Int;
}

handle let x = perform log(2) in x with {
  return v -> v;
  log(msg) k -> resume(msg);
}
```

You can also call the continuation directly (`k(value)`).

Why hook: Effects separate "what happened" from "how you respond," which is a
useful mental model for conscious action.

### Externs (FFI)
```tks
extern c safe fn print_int(x: Int): Unit !{IO};
print_int(7)
```

Run with `--ffi` to enable built-in externs.

### Modules + Stdlib
```tks
module A {
  export { value }
  let value = 10;
}

module B {
  from A import { value };
  let doubled = value;
}
```

Stdlib modules (loaded by `tksc`):
- `TKS.Core`, `TKS.RPM`, `TKS.Quantum`, `TKS.Noetics`,
  `TKS.Fractals`, `TKS.Foundations`

Set `TKS_STDLIB_DIR` to use a custom stdlib location.

### Types (Common Ones)
```tks
Int, Bool, Unit, Domain, Foundation, Ordinal
Element[A]          -- world A/B/C/D
Noetic[T], Fractal[T], RPM[T], QState[T]
Handler[{Effect}, In, Out]
```

Effect row types:
```tks
Int !{IO}
Int !{IO|r}
```

Function types:
```tks
Int -> Int
```

## Raw TKS Snippets (1:1 With Explanations)
Each snippet below shows real, runnable syntax. The explanation directly
maps to the code so readers can see how the language expresses different
ideas.
If a snippet is currently check-only (not lowered to bytecode), it is
explicitly labeled.

### 1) Ordinal Math + Binding
```tks
let x = omega + 2;
x
```
Explanation: bind an ordinal value, then reuse it.

### 2) Functions + Application
```tks
let id = \x -> x;
id A1
```
Explanation: define a lambda and apply it to a value.

### 3) Elements + Noetics
```tks
let idea = A1;
idea^2
```
Explanation: treat an Element as a symbolic anchor and apply a noetic
operator to reframe it.

### 4) Foundations (Context Grounding)
```tks
let ground = F3b;
ground
```
Explanation: foundations are first-class values that represent a stable
context you can reference or pass around. Note: currently check-only
(not lowered to bytecode).

### 5) Ordinals
```tks
omega + 2
```
Explanation: ordinal literals and arithmetic are supported as values.

### 6) Quantum Forms
```tks
let q = superpose { 1: |10>, 2: |20> };
measure(q)
```
Explanation: build a superposition, then measure to choose a value.

### 7) RPM Flow
```tks
return A1 >>= (\x -> return (x^1))
```
Explanation: RPM chaining (`>>=`) sequences progressive steps.

### 8) Effects + Handlers
```tks
effect Log { op log(msg: Int): Int; }

handle let x = perform log(2) in x with {
  return v -> v;
  log(msg) k -> resume(msg);
}
```
Explanation: define an effect, perform it, and handle it by resuming the
continuation.

### 9) Extern Calls (FFI)
```tks
extern c safe fn print_int(x: Int): Unit !{IO};
print_int(7)
```
Explanation: call a host function via FFI (run with `tks run --ffi`).

### 10) Modules + Imports
```tks
module A {
  export { value }
  let value = omega + 2;
}

module B {
  from A import { value };
  let echoed = value;
}
```
Explanation: define modules and import values across them.
Note: module bodies are currently check-only; use `tksc check`/`tksc build`.

### 11) Ordinals + RPM Combo
```tks
return omega >>= (\x -> return x)
```
Explanation: RPM can carry ordinal values; this wraps an ordinal and binds it.

### 12) Multi-Module Import Pattern
```tks
module A {
  export { a }
  let a = omega;
}

module B {
  export { b }
  let b = omega;
}

module C {
  from A import { a };
  from B import { b };
  let sum = ord(a + b);
}
```
Explanation: combine multiple modules by importing from more than one source.
Note: module bodies are currently check-only; use `tksc check`/`tksc build`.

### 13) Suggested Project Layout (Canon/Engine/CLI)
```tks
module Canon {
  export { seed }
  let seed = omega;
}

module Engine {
  export { step }
  let step = \x -> succ(x);
}

module CLI {
  from Canon import { seed };
  from Engine import { step };
  let result = step seed;
}
```
Explanation: split the project into canonical data (`Canon`), reusable
transformations (`Engine`), and a command/entry layer (`CLI`) that wires them
together.
Note: module bodies are currently check-only; use `tksc check`/`tksc build`.

### GUI-Ready Snippets (Paste Into GUI)
These are safe to paste into the GUI Run button (no FFI or modules).

```tks
let x = omega + 2;
x
```
Explanation: ordinal math and binding.

```tks
let idea = A2;
idea^3
```
Explanation: noetic apply on an Element value.

```tks
let q = superpose { 1: |1>, 2: |2> };
measure(q)
```
Explanation: quantum superposition and measurement.

## Real-World Uses by Concept (Metaphor + Professions + Code)
Each entry includes a kid-level metaphor, the exact technical term, and a
practical use case with professions that use similar ideas.

### Elements
Metaphor: stickers on boxes so you remember what each box is.
Tech mapping: Element literals like `A1`..`D10`.
Practical use: tagging ideas or categories so they can be transformed later.
Professions: therapy, coaching, education, business ops, taxonomy design, data labeling.
```tks
let idea = A1;
idea
```

### Foundations
Metaphor: the floor you stand on before you build anything.
Tech mapping: Foundation literals like `F3b` (`1a`..`7d`).
Practical use: a stable context or baseline for reasoning.
Professions: therapy, coaching, education, business ops, systems design.
```tks
let base = F3b;
base
```
Note: currently check-only (not lowered to bytecode).

### Noetics
Metaphor: a lens that changes how you see the same idea.
Tech mapping: noetic apply `expr^2` (digits 0-9).
Practical use: transforming a labeled idea into a new view.
Professions: therapy, coaching, education, cognitive modeling, UX research.
```tks
let idea = A1;
idea^2
```

### RPM (Rule of Progressive Manifestation)
Metaphor: a quest log where each step unlocks the next step.
Tech mapping: `return`, `>>=`, `check`, `acquire`.
Practical use: staged workflows, gated progress, step-by-step reasoning.
Professions: coaching, education, business ops, workflow automation, process ops.
```tks
return A1 >>= (\x -> return (x^1))
```

### Ordinals
Metaphor: counting steps past all normal counting steps.
Tech mapping: `omega`, `succ(...)`, `ord(...)`, `+`, `*`, `^` in ordinal space.
Practical use: ranking infinite processes and proving termination.
Professions: formal methods, theorem proving, theoretical CS, education research, business ops.
```tks
omega + 2
```

### Limit (Ordinal)
Metaphor: “what you get after you consider every step before infinity.”
Tech mapping: `limit(k < omega . k)`.
Practical use: convergence and end behavior of infinite steps.
Professions: therapy, coaching, education, business ops, systems modeling.
```tks
limit(k < omega . k)
```
Note: currently check-only (not lowered to bytecode).

### Fractals
Metaphor: a stamp pattern that repeats at many sizes.
Tech mapping: `<<1:2:...>>_omega(expr)` or Unicode delimiters.
Practical use: self-similar patterns, procedural generation.
Professions: therapy, coaching, education, business ops, generative design.
```tks
<<1:2:...>>_omega(A1)
```
Note: currently check-only (not lowered to bytecode).

### Transfinite Loops
Metaphor: a loop that runs through every normal step, then a special limit step.
Tech mapping: `transfinite loop i < omega from ... step (...) limit (...)`.
Practical use: define processes that must include a limit rule.
Professions: therapy, coaching, education, business ops, formal verification.
```tks
transfinite loop i < omega from A1 step (x -> x^1) limit (l -> l)
```
Note: currently check-only (not lowered to bytecode).

### Quantum Forms
Metaphor: a box of possibilities; measuring picks one.
Tech mapping: `|10>`, `superpose { ... }`, `measure(...)`, `entangle(...)`.
Practical use: probabilistic modeling and quantum-style simulations.
Professions: decision science, quantum computing, therapy, coaching, business ops.
```tks
let q = superpose { 1: |10>, 2: |20> };
measure(q)
```

### Effects + Handlers
Metaphor: ring a bell, and a helper decides how to respond.
Tech mapping: `effect`, `perform`, `handle`, `resume`.
Practical use: logging, errors, structured external interactions.
Professions: backend engineering, platform tooling, business ops, education, therapy/coaching.
```tks
effect Log { op log(msg: Int): Int; }

handle let x = perform log(2) in x with {
  return v -> v;
  log(msg) k -> resume(msg);
}
```

### Externs (FFI)
Metaphor: a phone call to the outside world.
Tech mapping: `extern c safe fn ...`.
Practical use: access OS, hardware, or native libraries.
Professions: systems programming, DevOps, business ops, education, therapy/coaching.
```tks
extern c safe fn print_int(x: Int): Unit !{IO};
print_int(7)
```

### Modules
Metaphor: folders that keep related ideas together.
Tech mapping: `module`, `export`, `from ... import ...`.
Practical use: organize larger projects and share values.
Professions: software engineering, education, business ops, therapy/coaching.
```tks
module Canon {
  export { seed }
  let seed = omega;
}

module Engine {
  export { step }
  let step = \x -> succ(x);
}

module CLI {
  from Canon import { seed };
  from Engine import { step };
  let result = step seed;
}
```
Note: module bodies are currently check-only; use `tksc check`/`tksc build`.

### ACBE
Metaphor: a goal-checker that compares a target to a result.
Tech mapping: `acbe(goal, expr)`.
Practical use: alignment checks and scoring pipelines.
Professions: evaluation, coaching, education, business ops, therapy.
```tks
acbe(A1, A1^1)
```
Note: currently check-only (not lowered to bytecode).

## What You Can Build (Practical Uses)
- TKS equation calculators and canonical validators.
- Symbolic or numeric pipelines with TKS constructs as data.
- Small interpreters or DSLs with effect handlers.
- Tools that integrate with native code via FFI.
- GPU experiments using `tks gpu ...` (GPU build only).

## Current Implementation Status (Important)
End-to-end supported (parse -> type -> lower -> bytecode -> VM):
- Int/Bool/Unit, `let`, lambdas, application, arithmetic.
- Noetic apply (`x^n`), ordinals, ket/superpose/measure/entangle.
- Effects/handlers/resume.
- RPM `return/check/acquire/>>=`.
- Extern calls with `tks run --ffi`.

Parsed + typed but not yet lowered to bytecode:
- Fractals, Foundations, bra/bra-ket, ACBE, transfinite loops.
- Module bodies during codegen (use `tksc check` for now).

Notes:
- `tks run` does not resolve modules; use `tksc` to check/build.
- REPL is not implemented yet.


## CLI Calculator (Optional)
If you want a terminal-only workflow (no GUI), you can run code directly
from stdin:

1) Build the binaries:
```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release
```

2) Run a quick expression:
```powershell
@'
let idea = A1;
idea^2
'@ | .\dist\tks-0.1.0-windows\tks.exe run -
```

3) Validate without running:
```powershell
@'
let idea = A2;
idea^3
'@ | .\dist\tks-0.1.0-windows\tksc.exe check -
```

4) Compile to bytecode and run:
```powershell
@'
omega + 2
'@ | .\dist\tks-0.1.0-windows\tksc.exe build --emit bc -o .\calc.tkso -

.\dist\tks-0.1.0-windows\tks.exe run .\calc.tkso
```

### One-Liner Cheat Sheet
```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release
@'let idea = A1; idea^2'@ | .\dist\tks-0.1.0-windows\tks.exe run -
@'let idea = A1; idea^2'@ | .\dist\tks-0.1.0-windows\tksc.exe check -
@'omega + 2'@ | .\dist\tks-0.1.0-windows\tksc.exe build --emit bc -o .\calc.tkso -
.\dist\tks-0.1.0-windows\tks.exe run .\calc.tkso
```

## GUI Equation Lab (Optional)
You can use the local GUI to validate and run code:

```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release
python .\tks-gui\server.py
```

Then open:
```
http://127.0.0.1:8747
```

See `tks-gui/README.md` for details.
The GUI includes save/load for snippets and projects, plus a Run Bytecode button.

## Video Demos + Online Playground (Optional)
If you want a low-friction "try it now" path, add links here:
- Video demo: [ADD_LINK_HERE]
- Online playground: [ADD_LINK_HERE]

These are placeholders until you publish the content.
