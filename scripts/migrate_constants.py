"""
TKS Canonical Constants Migration Script

This script automates the migration of hardcoded TKS constants to a centralized
`tks_rules` package. It is designed to be robust and idempotent.

It performs the following steps for each target file:
1.  Parses the file's Abstract Syntax Tree (AST).
2.  Identifies which canonical constants are defined locally.
3.  Determines the necessary imports from `tks_rules`.
4.  Uses an AST transformer to:
    a. Remove the old, local constant assignment nodes.
    b. Inject the new `from tks_rules... import ...` nodes at the top of the file.
5.  Unparses the modified AST back into source code, preserving comments and
    most formatting.
6.  Overwrites the original file with the migrated content.

This approach is safer than simple regex or line-based replacement as it
understands the Python code structure.

Author: Gemini-CLI
Date: 2025-12-16
"""
import ast
import os
from collections import defaultdict

# =============================================================================
# MIGRATION CONFIGURATION
# =============================================================================

# Map of constant names to the `tks_rules` submodule they belong to.
# This is the single source of truth for the migration.
MIGRATION_MAP = {
    "NOETICS": "noetics",
    "NOETIC_NAMES": "noetics",
    "INVOLUTION_PAIRS": "noetics",
    "WORLDS": "worlds",
    "FOUNDATIONS": "foundations",
    "RPM_MAPPINGS": "rpm",
    "ALLOWED_OPS": "operators",
    "OPERATORS": "operators",
    "RPM_DESIRE": "rpm",
    "RPM_WISDOM": "rpm",
    "RPM_POWER": "rpm",
}

# The list of all 23 files requiring migration, as provided by the user.
# Grouped by tier for clarity.
FILES_TO_MIGRATE = {
    # Tier 1 - Core Files
    "teacher/validator.py",
    "training/datasets.py",
    "training/losses.py",
    "tks_llm_core.py",
    "inversion/engine.py",

    # Tier 2 - TKS Features
    "tks_features/scenario_generator.py",

    # Tier 3 - Scripts
    "scripts/generate_pilot_data.py",
    "scripts/generate_new_canonical_equations.py",
    "scripts/generate_expanded_pairs.py",
    "scripts/generate_pairs_direct.py",
    "scripts/generate_rich_equations.py",
    "scripts/generate_diverse_equations.py",
    "scripts/generate_long_train_equations.py",
    "scripts/generate_lacunary_benchmark.py",
    "scripts/generate_ci_style_pairs.py",
    "scripts/create_mixed_finetune.py",
    "scripts/create_mixed_holdout.py",
    "scripts/attractor_checks.py",
    "scripts/evaluate_pilot.py",
    "scripts/infer.py",
    "scripts/serve_api.py",
    "scripts/phase6_eval.py",
    "scripts/train_with_augmented.py",
}

# =============================================================================
# AST MIGRATION LOGIC
# =============================================================================

class ConstantMigrationTransformer(ast.NodeTransformer):
    """
    An AST transformer that removes local constant definitions and adds
    `tks_rules` imports.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.constants_to_migrate = self._find_local_constants()
        self.imports_to_add = self._prepare_imports()
        self.imports_injected = False

    def _find_local_constants(self):
        """Parse the file to find which constants from MIGRATION_MAP it defines."""
        to_migrate = set()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read(), filename=self.file_path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in MIGRATION_MAP:
                                to_migrate.add(target.id)
            except Exception as e:
                print(f"    - WARNING: Could not parse {self.file_path}. Skipping. Error: {e}")
        return to_migrate

    def _prepare_imports(self):
        """Group constants by the module they need to be imported from."""
        imports = defaultdict(set)
        for const in self.constants_to_migrate:
            module = MIGRATION_MAP[const]
            imports[module].add(const)
        return imports

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Overrides the top-level module visit to inject imports.
        """
        # First, visit all children to remove assignments
        self.generic_visit(node)

        if not self.imports_to_add:
            return node

        new_import_nodes = []
        for module, constants in sorted(self.imports_to_add.items()):
            import_node = ast.ImportFrom(
                module=f"tks_rules.{module}",
                names=[ast.alias(name=c, asname=None) for c in sorted(list(constants))],
                level=0
            )
            new_import_nodes.append(import_node)

        # Find the correct insertion point (after docstring and other imports)
        insert_pos = 0
        for i, body_node in enumerate(node.body):
            if isinstance(body_node, ast.Expr) and isinstance(body_node.value, ast.Constant) and isinstance(body_node.value.value, str):
                # This is likely the module docstring
                insert_pos = i + 1
            elif isinstance(body_node, (ast.Import, ast.ImportFrom)):
                insert_pos = i + 1

        # Insert new imports at the determined position
        node.body[insert_pos:insert_pos] = new_import_nodes

        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """
        Removes a top-level assignment if its target is a constant we need
        to migrate.
        """
        targets_to_keep = []
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.constants_to_migrate:
                # This is a constant we're migrating, so we don't keep its assignment
                print(f"    - Removing local definition of '{target.id}'")
                continue
            targets_to_keep.append(target)
        
        if not targets_to_keep:
            # If all targets of this assignment are removed, remove the whole node
            return None
        
        # If it was a multi-target assignment and only some were removed
        node.targets = targets_to_keep
        return node


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def migrate_file(file_path: str):
    """
    Applies the migration logic to a single file.
    """
    print(f"\nProcessing file: {file_path}")
    if not os.path.exists(file_path):
        print(f"  - ERROR: File not found. Skipping.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            original_tree = ast.parse(source_code, filename=file_path)

        transformer = ConstantMigrationTransformer(file_path)
        
        if not transformer.constants_to_migrate:
            print("  - No canonical constants found to migrate. Skipping.")
            return

        print(f"  - Found constants to migrate: {', '.join(sorted(list(transformer.constants_to_migrate)))}")

        modified_tree = transformer.visit(original_tree)
        ast.fix_missing_locations(modified_tree)

        # Generate the new source code
        new_source_code = ast.unparse(modified_tree)

        # Write the changes back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_source_code)

        print(f"  - Migration successful.")

    except Exception as e:
        print(f"  - ERROR: Failed to migrate {file_path}. Error: {e}")

def main():
    """
    Main function to run the migration on all specified files.
    """
    print("=" * 60)
    print("TKS Canonical Constants Migration Tool")
    print("=" * 60)
    print(f"Found {len(FILES_TO_MIGRATE)} files to process.")

    for file_path in sorted(list(FILES_TO_MIGRATE)):
        migrate_file(file_path)

    print("\n" + "=" * 60)
    print("Migration complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
