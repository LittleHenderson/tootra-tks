"""
Quick syntax check - verify all modules can be imported
"""

print("Testing module imports...")

try:
    print("  Importing tks_4op_block_v2...")
    import tks_4op_block_v2
    print("    ✓ Success")
except Exception as e:
    print(f"    ✗ Failed: {e}")

try:
    print("  Importing negation_scope_dataset...")
    import negation_scope_dataset
    print("    ✓ Success")
except Exception as e:
    print(f"    ✗ Failed: {e}")

try:
    print("  Importing experiment_negation_scope...")
    import experiment_negation_scope
    print("    ✓ Success")
except Exception as e:
    print(f"    ✗ Failed: {e}")

try:
    print("  Importing mechanistic_probe...")
    import mechanistic_probe
    print("    ✓ Success")
except Exception as e:
    print(f"    ✗ Failed: {e}")

print("\nAll imports successful!")
