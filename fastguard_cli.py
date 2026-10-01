#!/usr/bin/env python3
"""
fastguard_cli.py — FastGuard CI/CD Breaking-Change Gate
=======================================================

Production-grade CLI designed to run inside GitHub Actions (or any CI
environment) to evaluate whether a Pull Request introduces breaking
API-contract changes to a FastAPI codebase.

Pipeline:
  1. Run `git diff --name-only <base>..<head>` to discover modified `.py` files.
  2. Retrieve the "before" and "after" source for each file via `git show`.
  3. Parse both versions with the Python `ast` module and compute the same
     10 structural AST-delta features used during model training.
  4. Load `fastguard_model.pkl` (serialised via joblib) and call
     `predict_proba()` on the aggregated feature vector.
  5. Apply the decision threshold (0.50) to produce a
     binary PASS / FAIL decision.

Exit codes:
  0 — PR is safe (probability < 0.50)
  1 — PR is potentially breaking (probability >= 0.50)

Usage (local):
    python fastguard_cli.py --base origin/main --head HEAD

Usage (GitHub Actions):
    - name: FastGuard Breaking-Change Gate
      run: python fastguard_cli.py --base origin/${{ github.base_ref }} --head ${{ github.sha }}
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exactly mirrors train_model.py::AST_FEATURE_COLS — order matters for the
# model's feature vector.
AST_FEATURE_COLS: list[str] = [
    "fields_removed_count",
    "fields_added_count",
    "fields_type_changed_count",
    "fields_default_changed_count",
    "models_renamed_count",
    "routes_removed_count",
    "routes_added_count",
    "routes_param_changed_count",
    "routes_param_type_changed_count",
    "routes_param_default_changed_count",
]

# Decision threshold — flag as breaking when P(breaking) >= 0.50.
DECISION_THRESHOLD: float = 0.50

# Default model artefact location (same directory as this script).
DEFAULT_MODEL_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fastguard_model.pkl",
)

# FastAPI HTTP verbs recognised as route decorators.
_HTTP_VERBS: frozenset[str] = frozenset({"get", "post", "put", "delete", "patch"})

# Framework-injected dependencies that are NOT part of the public contract.
_INTERNAL_DEPENDENCY_CALLS: frozenset[str] = frozenset({"Depends", "Security"})
_INTERNAL_ANNOTATION_NAMES: frozenset[str] = frozenset({
    "Request", "Response", "BackgroundTasks",
    "Session", "AsyncSession", "Header", "Cookie",
})

# ANSI colour codes (safe for GitHub Actions log rendering).
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ============================================================================
# Git Helpers
# ============================================================================

def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    """Execute a git sub-process and return the CompletedProcess result.

    Raises ``SystemExit`` with a descriptive message on any git failure so
    the CI pipeline fails fast with a useful diagnostic.
    """
    cmd = ["git"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print(
            f"{_RED}{_BOLD}FATAL:{_RESET} `git` executable not found on PATH. "
            "Ensure git is installed in this CI environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    return result


def get_changed_py_files(base: str, head: str) -> list[str]:
    """Return a list of `.py` file paths modified between *base* and *head*.

    Uses ``git diff --name-only --diff-filter=ACMR`` to capture files that
    were Added, Copied, Modified, or Renamed (excludes pure deletions since
    there is no "after" source to analyse).
    """
    result = _run_git(
        "diff", "--name-only", "--diff-filter=ACMR",
        f"{base}...{head}", "--", "*.py",
    )

    if result.returncode != 0:
        # Common failure: shallow clone without sufficient history.
        stderr = result.stderr.strip()
        print(
            f"{_RED}{_BOLD}ERROR:{_RESET} `git diff` failed (rc={result.returncode}).\n"
            f"  stderr: {stderr}\n\n"
            f"  Hint: In GitHub Actions, ensure you checkout with sufficient history:\n"
            f"    - uses: actions/checkout@v4\n"
            f"      with:\n"
            f"        fetch-depth: 0",
            file=sys.stderr,
        )
        sys.exit(1)

    files = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(".py")
    ]
    return files


def get_file_source(ref: str, filepath: str) -> str | None:
    """Retrieve the contents of *filepath* at git ref *ref*.

    Returns ``None`` when the file does not exist at that ref (e.g. newly
    added files have no "before" version).
    """
    result = _run_git("show", f"{ref}:{filepath}")
    if result.returncode != 0:
        return None
    return result.stdout


# ============================================================================
# AST Feature Extraction  (mirrors fastapi_ast_delta_dataset.py exactly)
# ============================================================================

def _extract_annotation_info(
    node: ast.AST | None,
) -> tuple[set[str], bool]:
    """Recursively collect annotation names and detect dependency-injection
    calls embedded inside ``Annotated[...]`` type hints."""
    names: set[str] = set()
    has_dep_call = False

    if node is None:
        return names, has_dep_call

    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Attribute):
        names.add(node.attr)
    elif isinstance(node, ast.Subscript):
        n_val, d_val = _extract_annotation_info(node.value)
        n_slice, d_slice = _extract_annotation_info(node.slice)
        names.update(n_val | n_slice)
        has_dep_call = has_dep_call or d_val or d_slice
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            n_elt, d_elt = _extract_annotation_info(elt)
            names.update(n_elt)
            has_dep_call = has_dep_call or d_elt
    elif isinstance(node, ast.Call):
        func = node.func
        call_name = (
            func.id if isinstance(func, ast.Name)
            else getattr(func, "attr", None)
        )
        if call_name in _INTERNAL_DEPENDENCY_CALLS:
            has_dep_call = True
        for arg in node.args:
            _, d_arg = _extract_annotation_info(arg)
            has_dep_call = has_dep_call or d_arg

    return names, has_dep_call


def _is_internal_dependency(
    annotation_node: ast.AST | None,
    default_node: ast.AST | None,
) -> bool:
    """Determine whether a function parameter is framework-injected rather
    than part of the public API contract."""
    if isinstance(default_node, ast.Call):
        func = default_node.func
        call_name = (
            func.id if isinstance(func, ast.Name)
            else getattr(func, "attr", None)
        )
        if call_name in _INTERNAL_DEPENDENCY_CALLS:
            return True

    ann_names, has_embedded_dep = _extract_annotation_info(annotation_node)
    if has_embedded_dep or any(
        name in _INTERNAL_ANNOTATION_NAMES for name in ann_names
    ):
        return True

    return False


def _get_pydantic_fields(tree: ast.Module | None) -> dict[str, dict[str, tuple[str, str]]]:
    """Extract Pydantic ``BaseModel`` subclasses and their typed fields."""
    models: dict[str, dict[str, tuple[str, str]]] = {}
    if tree is None:
        return models

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_pydantic = any(
            (isinstance(b, ast.Name) and b.id == "BaseModel")
            or (isinstance(b, ast.Attribute) and b.attr == "BaseModel")
            for b in node.bases
        )
        if not is_pydantic:
            continue

        fields: dict[str, tuple[str, str]] = {}
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                type_str = ast.dump(item.annotation) if item.annotation else "Any"
                default_str = ast.dump(item.value) if item.value else "NONE"
                fields[item.target.id] = (type_str, default_str)
        models[node.name] = fields

    return models


def _get_fastapi_routes(
    tree: ast.Module | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Map ``(verb, path)`` → ``{func_name, params}`` for FastAPI routes."""
    routes: dict[tuple[str, str], dict[str, Any]] = {}
    if tree is None:
        return routes

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
            ):
                continue
            verb = decorator.func.attr
            if verb not in _HTTP_VERBS:
                continue

            path: str | None = None
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                path = str(decorator.args[0].value)
            if path is None:
                continue  # skip dynamic/variable route paths

            args = node.args.args
            defaults = node.args.defaults
            first_default_idx = len(args) - len(defaults)

            params: dict[str, tuple[str, str]] = {}
            for i, arg in enumerate(args):
                if arg.arg in ("self", "cls"):
                    continue
                default_node = (
                    defaults[i - first_default_idx]
                    if i >= first_default_idx
                    else None
                )
                if _is_internal_dependency(arg.annotation, default_node):
                    continue
                type_str = ast.dump(arg.annotation) if arg.annotation else "Any"
                default_str = ast.dump(default_node) if default_node else "NONE"
                params[arg.arg] = (type_str, default_str)

            routes[(verb, path)] = {"func_name": node.name, "params": params}

    return routes


def compute_ast_deltas(
    source_before: str | None,
    source_after: str | None,
) -> dict[str, int]:
    """Compute the 10 structural AST-delta features between two versions of a
    Python source file.

    This function is an exact mirror of
    ``fastapi_ast_delta_dataset.compute_ast_deltas`` to guarantee feature
    parity between training and inference.
    """
    deltas: dict[str, int] = {col: 0 for col in AST_FEATURE_COLS}

    try:
        tree_before = ast.parse(source_before) if source_before else None
        tree_after = ast.parse(source_after) if source_after else None
    except SyntaxError:
        return deltas

    # --- Pydantic model diff ---
    models_before = _get_pydantic_fields(tree_before)
    models_after = _get_pydantic_fields(tree_after)

    matched_b: set[str] = set()
    matched_a: set[str] = set()

    for name, fields_bef in models_before.items():
        if name in models_after:
            fields_aft = models_after[name]
            removed = set(fields_bef) - set(fields_aft)
            added = set(fields_aft) - set(fields_bef)
            common = set(fields_bef) & set(fields_aft)

            deltas["fields_removed_count"] += len(removed)
            deltas["fields_added_count"] += len(added)
            deltas["fields_type_changed_count"] += sum(
                1 for f in common if fields_bef[f][0] != fields_aft[f][0]
            )
            deltas["fields_default_changed_count"] += sum(
                1 for f in common if fields_bef[f][1] != fields_aft[f][1]
            )
            matched_b.add(name)
            matched_a.add(name)

    # Fuzzy rename matching for unmatched models (Jaccard ≥ 0.75, ≥ 4 fields).
    unmatched_b = {n: f for n, f in models_before.items() if n not in matched_b}
    unmatched_a = {n: f for n, f in models_after.items() if n not in matched_a}

    for name_b, fields_b in list(unmatched_b.items()):
        best_match: str | None = None
        highest_sim = 0.0
        for name_a, fields_a in list(unmatched_a.items()):
            set_b, set_a = set(fields_b.keys()), set(fields_a.keys())
            union = len(set_b | set_a)
            jaccard = len(set_b & set_a) / union if union > 0 else 0.0
            if union >= 4 and jaccard >= 0.75 and jaccard > highest_sim:
                highest_sim = jaccard
                best_match = name_a

        if best_match:
            deltas["models_renamed_count"] += 1
            fields_aft = unmatched_a.pop(best_match)
            common = set(fields_b) & set(fields_aft)
            deltas["fields_removed_count"] += len(set(fields_b) - set(fields_aft))
            deltas["fields_added_count"] += len(set(fields_aft) - set(fields_b))
            deltas["fields_type_changed_count"] += sum(
                1 for f in common if fields_b[f][0] != fields_aft[f][0]
            )
            deltas["fields_default_changed_count"] += sum(
                1 for f in common if fields_b[f][1] != fields_aft[f][1]
            )
        else:
            deltas["fields_removed_count"] += len(fields_b)

    # --- Route diff ---
    routes_before = _get_fastapi_routes(tree_before)
    routes_after = _get_fastapi_routes(tree_after)

    matched_r_b: set[tuple[str, str]] = set()
    matched_r_a: set[tuple[str, str]] = set()

    for r_key, r_info_b in routes_before.items():
        if r_key in routes_after:
            r_info_a = routes_after[r_key]
            p_b, p_a = r_info_b["params"], r_info_a["params"]

            p_removed = set(p_b) - set(p_a)
            p_added = set(p_a) - set(p_b)
            p_common = set(p_b) & set(p_a)

            deltas["routes_param_changed_count"] += len(p_removed) + len(p_added)
            deltas["routes_param_type_changed_count"] += sum(
                1 for p in p_common if p_b[p][0] != p_a[p][0]
            )
            deltas["routes_param_default_changed_count"] += sum(
                1 for p in p_common if p_b[p][1] != p_a[p][1]
            )
            matched_r_b.add(r_key)
            matched_r_a.add(r_key)

    deltas["routes_removed_count"] = len(routes_before) - len(matched_r_b)
    deltas["routes_added_count"] = len(routes_after) - len(matched_r_a)

    return deltas


# ============================================================================
# Model Loading & Inference
# ============================================================================

def load_model(model_path: str) -> Any:
    """Load the serialised FastGuard model from disk.

    Falls back gracefully if joblib is unavailable or the pickle is missing.
    """
    if not os.path.isfile(model_path):
        print(
            f"{_RED}{_BOLD}FATAL:{_RESET} Model file not found: {model_path}\n"
            "  Ensure `fastguard_model.pkl` is committed to the repository or\n"
            "  available as a CI artefact.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import joblib  # type: ignore[import-untyped]
    except ImportError:
        print(
            f"{_RED}{_BOLD}FATAL:{_RESET} `joblib` is not installed.\n"
            "  Install it with: pip install joblib",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        print(
            f"{_RED}{_BOLD}FATAL:{_RESET} Failed to deserialise model: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sanity-check that the loaded object has the expected interface.
    if not hasattr(model, "predict_proba"):
        print(
            f"{_RED}{_BOLD}FATAL:{_RESET} Loaded object from {model_path} does "
            "not expose `predict_proba()`. Expected a scikit-learn classifier.",
            file=sys.stderr,
        )
        sys.exit(1)

    return model


# ============================================================================
# Report Formatting
# ============================================================================

def _format_feature_table(deltas: dict[str, int]) -> str:
    """Build a Markdown-style table of non-zero AST features for CI logs."""
    non_zero = {k: v for k, v in deltas.items() if v != 0}
    if not non_zero:
        return "  (all AST feature counts are zero)"

    max_name_len = max(len(k) for k in non_zero)
    lines = [
        f"  {'Feature':<{max_name_len}}  Count",
        f"  {'─' * max_name_len}  ─────",
    ]
    for feat, count in non_zero.items():
        lines.append(f"  {feat:<{max_name_len}}  {count:>5}")
    return "\n".join(lines)


def print_report(
    *,
    changed_files: list[str],
    aggregated_deltas: dict[str, int],
    probability: float,
    threshold: float,
    is_breaking: bool,
) -> None:
    """Print a formatted CI/CD report to stdout."""

    print()
    print(f"{_BOLD}{'═' * 62}{_RESET}")
    print(f"{_BOLD}  ⚡  FastGuard — Breaking-Change Analysis Report{_RESET}")
    print(f"{_BOLD}{'═' * 62}{_RESET}")
    print()

    # Files analysed
    print(f"{_CYAN}Modified Python files ({len(changed_files)}):{_RESET}")
    for f in changed_files:
        print(f"  • {f}")
    print()

    # AST features
    print(f"{_CYAN}Extracted AST-delta features:{_RESET}")
    print(_format_feature_table(aggregated_deltas))
    print()

    # ----- Structured AST mutations report -----
    non_zero = {k: v for k, v in aggregated_deltas.items() if v != 0}
    if non_zero:
        print(f"{_CYAN}Detected AST mutations:{_RESET}")
        # Type changes
        if aggregated_deltas.get("fields_type_changed_count", 0) > 0:
            print(f"  • Type changes: {aggregated_deltas['fields_type_changed_count']} field(s) had their type annotation modified")
        if aggregated_deltas.get("routes_param_type_changed_count", 0) > 0:
            print(f"  • Route param type changes: {aggregated_deltas['routes_param_type_changed_count']} route parameter(s) changed type")
        # Field removals
        if aggregated_deltas.get("fields_removed_count", 0) > 0:
            print(f"  • Field removals: {aggregated_deltas['fields_removed_count']} Pydantic model field(s) removed")
        # Field additions
        if aggregated_deltas.get("fields_added_count", 0) > 0:
            print(f"  • Field additions: {aggregated_deltas['fields_added_count']} Pydantic model field(s) added")
        # Default changes
        if aggregated_deltas.get("fields_default_changed_count", 0) > 0:
            print(f"  • Default changes: {aggregated_deltas['fields_default_changed_count']} field default(s) modified")
        if aggregated_deltas.get("routes_param_default_changed_count", 0) > 0:
            print(f"  • Route param default changes: {aggregated_deltas['routes_param_default_changed_count']} route parameter default(s) modified")
        # Model renames
        if aggregated_deltas.get("models_renamed_count", 0) > 0:
            print(f"  • Model renames: {aggregated_deltas['models_renamed_count']} Pydantic model(s) renamed")
        # Endpoint route updates
        if aggregated_deltas.get("routes_removed_count", 0) > 0:
            print(f"  • Endpoint removals: {aggregated_deltas['routes_removed_count']} route(s) removed")
        if aggregated_deltas.get("routes_added_count", 0) > 0:
            print(f"  • Endpoint additions: {aggregated_deltas['routes_added_count']} route(s) added")
        if aggregated_deltas.get("routes_param_changed_count", 0) > 0:
            print(f"  • Route param changes: {aggregated_deltas['routes_param_changed_count']} route parameter(s) added/removed")
        print()

    # Inference result
    print(f"{_CYAN}Model inference:{_RESET}")
    print(f"  Breaking probability : {probability:.4f}")
    print(f"  Decision threshold   : {threshold:.4f}")
    # Machine-parseable probability line for programmatic consumption
    print(f"Probability: {probability:.4f}")
    print()

    if is_breaking:
        print(f"{_RED}{_BOLD}{'─' * 62}{_RESET}")
        print(f"{_RED}{_BOLD}  ✖  FAIL — Potential breaking change detected!{_RESET}")
        print(f"{_RED}{_BOLD}{'─' * 62}{_RESET}")
        print()
        print(f"{_YELLOW}  The following structural changes contributed to this signal:{_RESET}")
        if non_zero:
            for feat, count in non_zero.items():
                print(f"    ⚠  {feat} = {count}")
        else:
            print(f"    (model flagged based on overall feature distribution)")
        print()
        print(f"{_YELLOW}  Recommended actions:{_RESET}")
        print(f"    1. Review the changed Pydantic models and FastAPI routes above.")
        print(f"    2. Ensure backward-compatible defaults or a versioned endpoint.")
        print(f"    3. Add a deprecation period if removing fields or routes.")
        print(f"    4. If this is intentional, document it as a breaking change.")
        print()
    else:
        print(f"{_GREEN}{_BOLD}{'─' * 62}{_RESET}")
        print(f"{_GREEN}{_BOLD}  ✔  PASS — No breaking changes detected.{_RESET}")
        print(f"{_GREEN}{_BOLD}{'─' * 62}{_RESET}")
        print()


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="fastguard",
        description=(
            "FastGuard — Evaluate a Pull Request for breaking API-contract "
            "changes in FastAPI applications using structural AST analysis "
            "and machine learning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fastguard_cli.py\n"
            "  python fastguard_cli.py --base origin/main --head HEAD\n"
            "  python fastguard_cli.py --base abc1234 --head def5678 --model ./model.pkl\n"
        ),
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base git ref for the diff (default: origin/main).",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head git ref for the diff (default: HEAD).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the serialised FastGuard model (.pkl). "
            f"Default: {DEFAULT_MODEL_PATH}"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DECISION_THRESHOLD,
        help=f"Decision threshold for breaking probability (default: {DECISION_THRESHOLD}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Execute the FastGuard CI/CD gate."""
    args = parse_args(argv)

    base: str = args.base
    head: str = args.head
    model_path: str = args.model
    threshold: float = args.threshold

    # ------------------------------------------------------------------
    # 1. Discover modified .py files
    # ------------------------------------------------------------------
    print(f"{_CYAN}FastGuard: Diffing {base}...{head}{_RESET}")
    changed_files = get_changed_py_files(base, head)

    if not changed_files:
        print()
        print(f"{_GREEN}{_BOLD}{'─' * 62}{_RESET}")
        print(f"{_GREEN}{_BOLD}  ✔  PASS — No Python files changed in this PR.{_RESET}")
        print(f"{_GREEN}{_BOLD}{'─' * 62}{_RESET}")
        print()
        sys.exit(0)

    print(f"  Found {len(changed_files)} modified Python file(s).")

    # ------------------------------------------------------------------
    # 2. Extract AST deltas for each file and aggregate
    # ------------------------------------------------------------------
    aggregated: dict[str, int] = {col: 0 for col in AST_FEATURE_COLS}
    files_analysed = 0

    for filepath in changed_files:
        source_before = get_file_source(base, filepath)
        source_after = get_file_source(head, filepath)

        # Both versions are None → nothing to analyse (shouldn't happen
        # given diff-filter, but guard defensively).
        if source_before is None and source_after is None:
            continue

        deltas = compute_ast_deltas(source_before, source_after)
        for col in AST_FEATURE_COLS:
            aggregated[col] += deltas[col]
        files_analysed += 1

    if files_analysed == 0:
        print(
            f"\n{_YELLOW}WARNING:{_RESET} Could not retrieve source for any "
            "changed file. Treating as safe.",
            file=sys.stderr,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # 3. Model inference
    # ------------------------------------------------------------------
    model = load_model(model_path)

    # Build the feature vector in the exact column order expected by the model.
    feature_vector = [[aggregated[col] for col in AST_FEATURE_COLS]]

    try:
        probabilities = model.predict_proba(feature_vector)
        breaking_prob: float = float(probabilities[0][1])
    except Exception as exc:
        print(
            f"{_RED}{_BOLD}FATAL:{_RESET} Model inference failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Decision & report
    # ------------------------------------------------------------------
    is_breaking = breaking_prob >= threshold

    print_report(
        changed_files=changed_files,
        aggregated_deltas=aggregated,
        probability=breaking_prob,
        threshold=threshold,
        is_breaking=is_breaking,
    )

    if is_breaking:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
