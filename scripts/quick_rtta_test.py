"""Quick RTTA test."""
import sys
sys.path.insert(0, '.')

from tks_features.rtta_scaffolding import StandaloneRTTA

rtta = StandaloneRTTA()

questions = [
    "A store sells pens for $2 each and notebooks for $5 each. A customer buys 4 items in total and spends $14. How many pens?",
    "A store sells shirts for $3 each and pants for $8 each. A customer buys 6 items in total and spends $33. How many shirts?",
]

print("="*60)
print("RTTA STANDALONE TEST")
print("="*60)

for q in questions:
    print(f"\nQ: {q}")
    result = rtta.reason(q)
    print(f"A: {result['answer']}")
    print(f"Method: {result['method']}")

print("\n" + "="*60)
print("RTTA with structure shown:")
print("="*60)

# Show the RTTA structure
q = questions[0]
print(f"\nQuestion: {q}")
print("\nGoal: Resolve wisdom question in mental/emotional domain")
print("\nRPM Prerequisites:")
for pid, name, deps in rtta.rpm_templates["math"]:
    dep_str = f" <- {deps}" if deps else ""
    print(f"  {pid}: {name}{dep_str}")

print("\nReasoning loop simulation:")
print("  Step 1: P1 (Variable ID) - Define p=pens, n=notebooks")
print("  Step 2: P2 (Equations) - p+n=4, 2p+5n=14")
print("  Step 3: P3 (Solve) - p=2, n=2")
print("  Step 4: P4 (Check) - Valid (positive integers)")
print("  Step 5: P5 (Answer) - 2 pens")

result = rtta.reason(q)
print(f"\nFinal Answer: {result['answer']}")
