# Design Notes and Load Limits

## Status

This artifact is a **reference semantics experiment**. The source material specifies CCDL as a declarative netlist and validation language; this project adds an executable scheduler and transfer model so the proposed processor analogy can be tested rather than asserted.

## Implemented CCDL subset

The parser supports:

```text
circuit
input / output
node
cjt CJT< cause | bias => effect > [parameters]
feedback
probe
```

It checks identifiers, state literals, references, parameter ranges, `cutoff < sat`, bias-domain precedence, feedback gain below one, feed-forward cycles, and cascade depth. The complete canonical CAS admissibility relation is not available in this artifact, so `is_admissible(operator, source, target)` remains a pluggable external rule set rather than being silently invented.

## CJT transfer semantics

v0.1 uses this explicit transfer rule:

```text
bias < cutoff        -> effect = 0, region=cutoff
otherwise raw        -> cause_activation * beta * supervisor_derate
raw < sat            -> effect = raw, region=active
sat <= raw < limit   -> effect = sat, region=saturation
raw >= limit         -> effect state flips to its involution, region=breakdown
```

The bias gates the transition. Continuous `gm` and `r_pi` modulation are not modeled, preserving the source statement that active-region gain is approximately `beta` at constant bias.

## Scheduler

CCDL declaration order does not define execution order. The runtime therefore:

1. parses the whole graph;
2. removes declared feedback edges from the combinational schedule;
3. topologically sorts the CJTs;
4. executes one component per trace tick;
5. applies low-gain feedback only as a governed state-update concept, not as an instantaneous combinational loop.

## CPSS supervisor

The supervisor implements the topology in the CCB-6 reading guide:

```text
P -> C -> E -> O startup
brownout comparator and beta derate
three-tier thermal response
three diverse channels -> 2-of-3 vote
oscillator/watchdog liveness
emergency OR -> permanent master fuse
vote-controlled answer/refusal MUX
```

The guide's explicit 2.5 V brownout reference and 25%/50% derates are retained. The normalized thermal thresholds and severe-brownout cutoff are testbench design choices and are labeled as such in code.

## Seam metric

The seam layer compares two independent executions:

```text
structural interpreter = declared CCDL state types and beta values
behavioral interpreter = synthetic task contract in semantic_contracts.json
```

At each requested recursive depth:

```text
delta = absolute numeric disagreement
      + categorical penalty for each accumulated Noetic mismatch
```

The baseline deliberately confuses `^3 Negative` with `^6 Male`, mirroring a documented semantic drift hazard. The accepted patch brings the full depth-1..3 curve to zero against the declared fixture. That is evidence that the patch satisfies this fixture, not evidence of universal semantic compositionality.

## Reflective governance

The reflective layer is proposal-based, never direct self-write:

```text
observe -> diagnose -> propose -> DRC -> simulate -> regression/fault suite
        -> three review votes -> commit or rollback
```

Immutable during execution:

```text
constitution node
TMR safety channel components
voter and supervisor kernel
master-cutoff logic
```

A low-gain `O -> P` feedback edge may exist in the design, but any proposed round-trip gain at or above one is rejected with `E-CJT-005 GainDivergence`.

## What v0.1 does not claim

- It does not parse raw English into CAS states.
- It does not execute a real language model.
- It does not implement the full 40-dimensional open-simplex state vector at every node.
- It does not implement canonical operator admissibility beyond the local structural checks.
- It does not model `gm`, `r_pi`, bandwidth, phase, or analog noise.
- It does not establish that CAS is universal for cognition.
- It does not establish production safety.

These omissions are surfaced in the report rather than hidden behind a successful demo.
