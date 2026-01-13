"""
TKS CLI Entrypoint - Run Episode

Runs either the EpisodeRunner runtime loop or the Strategic Agent.
Usage: python -m src.cli.run_episode --goal "..." --stack "1a:5b:2b"
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

# Import the Agent/Runner from the new src structure
try:
    from src.tks.core.tks_agent import TKSStrategicAgent
    from src.tks.runner import EpisodeRunner, EpisodeInput
    from src.tks.config import TKSConfig
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run a TKS Cognitive Episode")
    parser.add_argument("--goal", type=str, required=True, help="The goal or query for the agent")
    parser.add_argument("--stack", type=str, default="1a:5b:2b", help="Foundation Stack (Mission:Method:Constraint)")
    parser.add_argument("--mode", type=str, default="NORMAL", help="Operational Mode override")
    parser.add_argument("--engine", choices=["runner", "agent"], default="runner", help="Execution engine")
    parser.add_argument("--show-trace", action="store_true", help="Print full episode trace")
    
    args = parser.parse_args()
    
    print(f"--- TKS Episode Started ---")
    print(f"Goal: {args.goal}")
    print(f"Stack: {args.stack}")
    
    if args.engine == "agent":
        try:
            agent = TKSStrategicAgent()
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize agent. {e}")
            return
        result = agent.run_episode(args.goal, foundation_stack=args.stack)
    else:
        try:
            runner = EpisodeRunner(TKSConfig(), enable_rumination=False)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize runner. {e}")
            return
        input_obj = EpisodeInput(goal=args.goal, foundation_stack=args.stack)
        result = runner.run_episode(input_obj)
    
    # Output Result
    print("\n--- Episode Result ---")
    if hasattr(result, "__dict__"):
        for k, v in result.__dict__.items():
            if k == "trace" and not args.show_trace:
                continue
            print(f"{k}: {v}")
    elif isinstance(result, dict):
        for k, v in result.items():
            if k == "trace" and not args.show_trace:
                continue
            print(f"{k}: {v}")
    else:
        print(result)
        
    print("\n--- End of Episode ---")

if __name__ == "__main__":
    main()
