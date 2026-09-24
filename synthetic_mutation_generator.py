"""
synthetic_mutation_generator.py

Walks .py files in a directory of already-cloned repos, finds real Pydantic
models and FastAPI routes, and applies a controlled mutation to each one
(field deletion, param rename, type change, etc). Each mutation produces a
(source_before, source_after) pair that is guaranteed to be a true positive
for is_breaking_commit -- no keyword guessing involved, the label is correct
by construction.

Reuses the same AST feature-extraction logic as the mining script so the
output schema matches your organic dataset exactly.
"""

import os
import ast
import copy
import random
import pandas as pd

REPOS_DIR = "cloned_repos"          # directory containing already-cloned repos (subfolders)
OUTPUT_CSV = "synthetic_breaking_dataset.csv"
MUTATIONS_PER_FILE = 3              # cap mutations sampled per eligible file
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Same detection logic as the mining script (kept in sync intentionally)
# ---------------------------------------------------------------------------

def get_pydantic_models(tree):
    """Return list of (class_node, field_names) for Pydantic models in the file."""
    models = []
    if not tree:
        return models
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_pydantic = any(
                (isinstance(b, ast.Name) and b.id == "BaseModel")
                or (isinstance(b, ast.Attribute) and b.attr == "BaseModel")
                for b in node.bases
            )
            if is_pydantic:
                field_items = [
                    item for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                ]
                if field_items:
                    models.append((node, field_items))
    return models


def get_route_functions(tree):
    """Return list of (func_node, decorator_node) for FastAPI route handlers with resolvable params."""
    routes = []
    if not tree:
        return routes
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                verb = decorator.func.attr
                if verb in ("get", "post", "put", "delete", "patch"):
                    real_args = [a for a in node.args.args if a.arg not in ("self", "cls")]
                    if real_args:
                        routes.append((node, decorator))
    return routes


# ---------------------------------------------------------------------------
# Mutation operators. Each takes a parsed tree (already deep-copied) and
# mutates it in place, returning a short description of what it did.
# ---------------------------------------------------------------------------

def mutate_delete_field(tree):
    models = get_pydantic_models(tree)
    if not models:
        return None
    class_node, field_items = random.choice(models)
    target = random.choice(field_items)
    class_node.body.remove(target)
    return f"deleted field '{target.target.id}' from model '{class_node.name}'"


def mutate_change_field_type(tree):
    models = get_pydantic_models(tree)
    if not models:
        return None
    class_node, field_items = random.choice(models)
    target = random.choice(field_items)
    old_type = ast.dump(target.annotation) if target.annotation else "Any"
    # Swap to a deliberately incompatible type
    target.annotation = ast.Name(id="int" if "str" in old_type else "str", ctx=ast.Load())
    ast.fix_missing_locations(target)
    return f"changed type of field '{target.target.id}' in model '{class_node.name}'"


def mutate_rename_route_param(tree):
    routes = get_route_functions(tree)
    if not routes:
        return None
    func_node, _ = random.choice(routes)
    real_args = [a for a in func_node.args.args if a.arg not in ("self", "cls")]
    target = random.choice(real_args)
    old_name = target.arg
    target.arg = old_name + "_renamed"
    return f"renamed route param '{old_name}' in function '{func_node.name}'"


def mutate_remove_route_param(tree):
    routes = get_route_functions(tree)
    eligible = [(f, d) for f, d in routes if len([a for a in f.args.args if a.arg not in ("self", "cls")]) > 1]
    if not eligible:
        return None
    func_node, _ = random.choice(eligible)
    real_args = [a for a in func_node.args.args if a.arg not in ("self", "cls")]
    target = random.choice(real_args)
    idx = func_node.args.args.index(target)
    n_defaults = len(func_node.args.defaults)
    default_start = len(func_node.args.args) - n_defaults
    func_node.args.args.remove(target)
    if idx >= default_start:
        del func_node.args.defaults[idx - default_start]
    return f"removed route param '{target.arg}' from function '{func_node.name}'"


MUTATION_OPERATORS = [
    mutate_delete_field,
    mutate_change_field_type,
    mutate_rename_route_param,
    mutate_remove_route_param,
]


# ---------------------------------------------------------------------------
# Import the mining script's feature extractor so synthetic rows use the
# exact same computation as your organic dataset. Adjust the module name
# below to match your actual mining script filename.
# ---------------------------------------------------------------------------

try:
    from fastapi_ast_delta_dataset import compute_ast_deltas
except ImportError:
    raise SystemExit(
        "Could not import compute_ast_deltas. Put this script in the same "
        "folder as your mining script (fastapi_ast_delta_dataset.py) so the "
        "synthetic features are computed identically to the organic dataset."
    )


def generate_synthetic_rows():
    dataset = []

    for repo_name in os.listdir(REPOS_DIR):
        repo_path = os.path.join(REPOS_DIR, repo_name)
        if not os.path.isdir(repo_path):
            continue

        for root, _, files in os.walk(repo_path):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                filepath = os.path.join(root, fname)
                rel_path = os.path.relpath(filepath, repo_path)

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        source_before = f.read()
                    tree_check = ast.parse(source_before)
                except (SyntaxError, UnicodeDecodeError):
                    continue

                # Only bother with files that actually have something to mutate
                if not get_pydantic_models(tree_check) and not get_route_functions(tree_check):
                    continue

                applied = 0
                attempts = 0
                while applied < MUTATIONS_PER_FILE and attempts < MUTATIONS_PER_FILE * 4:
                    attempts += 1
                    operator = random.choice(MUTATION_OPERATORS)
                    tree_mutated = ast.parse(source_before)  # fresh parse per attempt
                    description = operator(tree_mutated)
                    if description is None:
                        continue

                    try:
                        source_after = ast.unparse(tree_mutated)
                    except Exception:
                        continue  # some trees may fail to unparse cleanly, skip

                    ast_deltas = compute_ast_deltas(source_before, source_after)

                    dataset.append({
                        "repo_name": repo_name,
                        "commit_hash": f"synthetic-{repo_name}-{rel_path}-{applied}",
                        "filepath": rel_path,
                        "commit_date": None,
                        "is_breaking_commit": 1,   # true by construction
                        "mutation_description": description,
                        "lines_added": 0,
                        "lines_deleted": 0,
                        "net_churn": 0,
                        "code_complexity": 0,
                        **ast_deltas,
                    })
                    applied += 1

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated {len(df)} synthetic breaking examples -> {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    generate_synthetic_rows()