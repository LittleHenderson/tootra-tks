"""
TKS CI Replay Gate (Agent Eval)

Runs the Core Replay Set and enforces strict non-regression.
Exit Code 0: All Pass
Exit Code 1: Regression Detected
"""

import sys
import os
import json
import random

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.tks.core.tks_agent import TKSStrategicAgent

def main():
    print("--- TKS CI Replay Gate ---")
    
    # 1. Load Replay Set
    replay_path = "replay_sets/rs_core_v1.json"
    if not os.path.exists(replay_path):
        print(f"FATAL: Replay set not found at {replay_path}")
        sys.exit(1)
        
    with open(replay_path, "r") as f:
        cases = json.load(f)
        
    # 2. Initialize Agent
    # Deterministic Seed for Logic
    random.seed(42)
    try:
        agent = TKSStrategicAgent()
    except Exception as e:
        print(f"FATAL: Agent init failed: {e}")
        sys.exit(1)
        
    failures = []
    
    for case in cases:
        print(f"Testing {case['episode_id']}...", end=" ")
        
        # Run Episode
        try:
            result = agent.run_episode(case['goal'])
        except Exception as e:
            print("ERROR")
            failures.append(f"{case['episode_id']} crashed: {e}")
            continue
            
        # Assert Status
        if result['status'] != case['expected_status']:
            print("FAIL (Status)")
            failures.append(f"{case['episode_id']} Status Mismatch: Expected {case['expected_status']}, Got {result['status']}")
            continue
            
        # Assert Reason (Partial Match)
        if 'expected_reason' in case:
            if case['expected_reason'] not in result.get('reason', ''):
                print("FAIL (Reason)")
                failures.append(f"{case['episode_id']} Reason Mismatch: Expected '{case['expected_reason']}' in '{result.get('reason')}'")
                continue
                
        # Assert Packet (if Success)
        if result['status'] == 'SUCCESS' and 'expected_packet' in case:
            if result['tks_packet'] != case['expected_packet']:
                print(f"FAIL (Packet: Got {result['tks_packet']})")
                failures.append(f"{case['episode_id']} Packet Mismatch: Expected {case['expected_packet']}, Got {result['tks_packet']}")
                continue
                
        # Assert Regulators
        if 'expected_regulators' in case:
            trace_str = str(result.get('plan_trace', []))
            for reg in case['expected_regulators']:
                if reg not in trace_str:
                    print(f"FAIL (Missing Regulator {reg})")
                    failures.append(f"{case['episode_id']} Missing Regulator: {reg}")
                    continue

        print("PASS")

    print("-" * 30)
    if failures:
        print(f"FAILED: {len(failures)} regressions detected.")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SUCCESS: All replay cases passed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
