# Source Map

The implementation was grounded in the user-provided engineering materials.

## CAS PCBA Engineer

File: `CAS_PCBA_Engineer (1).pdf`  
SHA-256: `9252b76a892d04ebe0ec3464729f1229f8bdee47d25068ffa5f5ca48049de983`

Key sections used:

- pp. 29-56: CJT terminals, cutoff/active/saturation/breakdown, involution behavior, and first test point.
- pp. 133-149: CJT datasheet, biasing, gain, operating regions, and runaway behavior.
- pp. 319-339: Cognitive Safety Board topology, three-stage decision cascade, saturation latch, MUX, fuse, and slow `O -> P` feedback.
- pp. 387-405: 40-cell Cognitive Memory Board, structural storage, leakage, refresh, and Fisher-Rao addressing. The prototype implements an eight-cell register subset.
- pp. 435-450: unit/integration/regression/system/fault testing, golden traces, fail-closed injection, and coverage.
- pp. 521-530: component datasheets and CJT parameter ranges.
- pp. 547-572: CCDL grammar, declarations, parameter defaults, DRC passes, error codes, and annotated example.

## CPSS CCB-6 Schematic Reading Guide

File: `CPSS_CCB6_Schematic_Reading_Guide.pdf`  
SHA-256: `9b5e05931208e73a8de385a527cfc1dd43a51ca635d2506dccbb2748017307a2`

Key pages used:

- pp. 2-6: supervisor role, drive rails, per-domain regulators, power-good signals, and strict `P -> C -> E -> O` sequencing.
- pp. 7-8: 2.5 V Vitality brownout reference, 50% derate, three-tier thermal response, and status bus.
- pp. 8-9: three diverse safety channels and 2-of-3 majority voter.
- pp. 9-11: oscillator-clocked watchdog, emergency OR, permanent master fuse, output MUX, and TP1-TP6 journey.
- p. 12: fault table and fail-safe summary.

## CPSS schematic image

File: `cpss_schematic.png`  
SHA-256: `cfc15cf03fee270a2205fad94f8ac4f2022a01efbdc44053c847ae987a3b2501`

Copied to `docs/CPSS_CCB6_schematic.png` for the generated HTML report.
