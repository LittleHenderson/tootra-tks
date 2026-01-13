import os
import shutil
import json

# --- Configuration ---
ROOT_DIR = os.getcwd()
STRG_DIR = os.path.join(ROOT_DIR, "strg-llm-stk")

# Directory Structure from Runbook Section 2
DIRS_TO_CREATE = [
    "specs",
    "specs/context",
    "specs/module_1_executor_router",
    "specs/module_2_verifier",
    "specs/module_3_capability",
    "specs/module_4_scoring",
    "specs/module_5_eval_replay",
    "specs/module_6_observability",
    "specs/module_7_governance",
    "canon/raw_pdfs",
    "canon/normalized",
    "canon/gold",
    "datasets/jsonl",
    "datasets/schemas",
    "datasets/registry",
    "src/tks/core",
    "src/tks/governance",
    "src/tks/planning",
    "src/tks/regulators",
    "src/cli",
    "configs",
    "replay_sets",
    "logs",
    "scripts/canon",
    "scripts/train",
    "scripts/eval",
    "AGENT_MISSIONS" # Folder for the team assignments
]

# Agent Roles & Missions (Runbook Section 6)
AGENT_MISSIONS = {
    "agent_data": {
        "role": "Data & Canon Steward",
        "objective": "Build and validate the Source of Truth (Canon).",
        "tasks": [
            "Scan 'canon/raw_pdfs' and extract normalized definitions into 'canon/normalized/*.json'.",
            "Implement 'scripts/canon/build_canon.py' and 'validate_canon.py'.",
            "Maintain 'canon/gold' with undeniable examples (JSONL).",
            "Ensure 'canon_index.json' contains hashes to prevent drift."
        ],
        "acceptance_criteria": "Canon validation script passes with 0 errors. Hashes match."
    },
    "agent_be": {
        "role": "Backend & Runtime Engineer",
        "objective": "Implement the core runtime modules and CLI.",
        "tasks": [
            "Move 'tks_llm_core_v4.py' to 'src/tks/core/' and refactor imports.",
            "Implement 'src/tks/governance' (High-Stakes gates, Clearance Tokens).",
            "Implement 'src/tks/planning' (RPM nesting logic, Regulator triggers).",
            "Create CLI entrypoint 'src/cli/run_episode.py'."
        ],
        "acceptance_criteria": "Can run 'python -m tks.cli.run_episode' end-to-end with a stub tool."
    },
    "agent_arch": {
        "role": "System Architect",
        "objective": "Enforce strict interfaces and policy consistency.",
        "tasks": [
            "Define strict interfaces for Module 1-7 interaction.",
            "Review 'specs/' and ensure implementation matches the PDF/MD specs.",
            "Implement 'anti-recursion' hard caps in the Router/Planner.",
            "Define the 'Foundation Stack' data structure (F1:F5:F2)."
        ],
        "acceptance_criteria": "Architecture diagram matches code structure. Policy caps are enforceable."
    },
    "agent_ml": {
        "role": "Machine Learning Engineer",
        "objective": "Train the model to understand TKS operators and Noetics.",
        "tasks": [
            "Generate the JSONL training packs using the schemas in 'datasets/schemas'.",
            "Implement 'scripts/train/mix_datasets.py' and 'run_train.py'.",
            "Fine-tune v4 on Operator Semantics (substitution tests).",
            "Run 'operator suite' evaluations."
        ],
        "acceptance_criteria": "Model passes 95% of Operator Substitution Tests."
    },
    "agent_eval": {
        "role": "Evaluation & Replay Lead",
        "objective": "Ensure the system improves and doesn't regress.",
        "tasks": [
            "Create 'replay_sets/rs_core.json' with golden episodes.",
            "Implement 'scripts/eval/run_replay.py'.",
            "Build the 'High-Stakes compliance suite' (inputs that MUST trigger PAUSE).",
            "Compare baseline vs candidate policies."
        ],
        "acceptance_criteria": "Replay regression suite exists and runs."
    },
    "agent_devops": {
        "role": "DevOps & Release",
        "objective": "Packaging, Environments, and Reproducibility.",
        "tasks": [
            "Create 'requirements.txt' and setup instructions.",
            "Create '.env.example'.",
            "Write the master 'Makefile' or 'run_all.bat' scripts.",
            "Ensure logs are structured and queryable."
        ],
        "acceptance_criteria": "New developer can 'git clone' and 'run_episode' in < 10 mins."
    }
}

def setup_directories():
    print("Creating directory structure...")
    for directory in DIRS_TO_CREATE:
        path = os.path.join(ROOT_DIR, directory)
        os.makedirs(path, exist_ok=True)
        # Create .gitkeep to ensure git tracks folders
        with open(os.path.join(path, ".gitkeep"), "w") as f:
            pass
    print("Directories created.")

def migrate_files():
    print("Migrating specs and schemas from strg-llm-stk...")
    if not os.path.exists(STRG_DIR):
        print(f"Warning: {STRG_DIR} not found. Skipping migration.")
        return

    # Map file patterns to destinations
    migration_map = {
        "SCHEMA.md": "datasets/schemas",
        "Spec": "specs",
        "Module": "specs",
        "Runbook": "specs", # The Master Runbook
        "Governance": "specs/module_7_governance"
    }

    for filename in os.listdir(STRG_DIR):
        src_path = os.path.join(STRG_DIR, filename)
        if os.path.isfile(src_path):
            moved = False
            # Check for Governance specifically first
            if "Governance" in filename:
                shutil.copy2(src_path, os.path.join(ROOT_DIR, "specs/module_7_governance", filename))
                print(f"Copied {filename} -> specs/module_7_governance/")
                continue

            for pattern, dest_sub in migration_map.items():
                if pattern in filename:
                    dest_path = os.path.join(ROOT_DIR, dest_sub, filename)
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied {filename} -> {dest_sub}/")
                    moved = True
                    break
            
            # Special case: The convo log is useful context for everyone
            if "convo" in filename:
                shutil.copy2(src_path, os.path.join(ROOT_DIR, "specs", "context", filename))
                print(f"Copied {filename} -> specs/context/")
                
    # Move v4 core if it exists
    v4_core = os.path.join(ROOT_DIR, "tks_llm_core_v4.py")
    if os.path.exists(v4_core):
        shutil.copy2(v4_core, os.path.join(ROOT_DIR, "src/tks/core/tks_llm_core_v4.py"))
        print("Copied tks_llm_core_v4.py -> src/tks/core/")

    # Move new implementation files
    new_files = {
        "governance.py": "src/tks/governance/governance.py",
        "regulators.py": "src/tks/regulators/regulators.py", 
        "abstraction.py": "src/tks/core/abstraction.py",
        "tks_agent.py": "src/tks/core/tks_agent.py"
    }
    
    for src, dest in new_files.items():
        src_path = os.path.join(ROOT_DIR, src)
        dest_path = os.path.join(ROOT_DIR, dest)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src} -> {dest}")

def create_mission_files():
    print("Generating AGENT_MISSIONS...")
    for agent_id, mission in AGENT_MISSIONS.items():
        filename = f"{agent_id}_mission.md"
        path = os.path.join(ROOT_DIR, "AGENT_MISSIONS", filename)
        
        content = f"# Mission: {mission['role']} ({agent_id})\n\n"
        content += f"**Objective:** {mission['objective']}\n\n"
        content += "## Critical Tasks\n"
        for task in mission['tasks']:
            content += f"- [ ] {task}\n"
        content += "\n## Acceptance Criteria\n"
        content += f"> {mission['acceptance_criteria']}\n"
        content += "\n## Context\n"
        content += "Refer to `specs/TKS_Master_README_CLI_Runbook.md` for full operational details."
        
        with open(path, "w") as f:
            f.write(content)
    print("Mission files created.")

if __name__ == "__main__":
    setup_directories()
    migrate_files()
    create_mission_files()
    print("\nTeam setup complete. Check 'AGENT_MISSIONS/' for work assignments.")
