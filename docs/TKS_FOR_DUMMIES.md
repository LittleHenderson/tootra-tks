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
let x = 2 + 3;
x * 4
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
let x = 1 in x + 2  -- let expression
\x -> x + 1         -- lambda
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
