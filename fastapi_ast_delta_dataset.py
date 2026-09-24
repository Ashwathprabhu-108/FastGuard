import os
import re
import ast
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from pydriller import Repository

# --- CONFIGURATION ---
GITHUB_TOKEN = ""  # Optional PAT to bypass GitHub API rate limits
MIN_STARS = 50
MAX_REPOS = 80
MAX_COMMITS_PER_REPO = 2000
SINCE_DATE = datetime.now(timezone.utc) - timedelta(days=365 * 3)
OUTPUT_CSV = "fastapi_ast_delta_dataset.csv"

# Label source ONLY. Never combine with AST delta features when computing is_breaking_commit.
BREAKING_KEYWORDS = [
    "breaking change", "breaking", "deprecate", "removed field",
    "schema change", "api change", "contract change", "incompatible"
]

INTERNAL_DEPENDENCY_CALLS = {"Depends", "Security"}
INTERNAL_ANNOTATION_NAMES = {
    "Request", "Response", "BackgroundTasks", "Session", "AsyncSession", "Header", "Cookie"
}

FALLBACK_REPOS = [
    "https://github.com/fastapi/full-stack-fastapi-template",
    "https://github.com/tiangolo/sqlmodel",
    "https://github.com/zhanymkanov/fastapi-best-practices",
    "https://github.com/long2ice/fastapi-cache",
    "https://github.com/dmontagu/fastapi-utils",
    "https://github.com/pydantic/pydantic",
    "https://github.com/fastapi/fastapi",
]

DEPENDENCY_MANIFEST_CANDIDATES = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]
DEPENDENCY_PATTERN = re.compile(r"\bfastapi\b|\bpydantic\b", re.IGNORECASE)


def _extract_annotation_info(node):
    """
    Recursively walk annotation AST nodes to collect:
    1. Base type/annotation names (Name, Attribute)
    2. Dependency injection calls (ast.Call to Depends/Security) inside Annotated[...]
    """
    names = set()
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
        call_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if call_name in INTERNAL_DEPENDENCY_CALLS:
            has_dep_call = True
        for arg in node.args:
            _, d_arg = _extract_annotation_info(arg)
            has_dep_call = has_dep_call or d_arg

    return names, has_dep_call


def is_internal_dependency(annotation_node, default_node):
    """Check if a parameter is framework-injected or part of the public API contract."""
    if isinstance(default_node, ast.Call):
        func = default_node.func
        call_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if call_name in INTERNAL_DEPENDENCY_CALLS:
            return True

    ann_names, has_embedded_dep = _extract_annotation_info(annotation_node)
    if has_embedded_dep or any(name in INTERNAL_ANNOTATION_NAMES for name in ann_names):
        return True

    return False


def get_pydantic_fields(tree):
    """Extract Pydantic model names and their field annotations + defaults.
    Matches BaseModel inheritance by base-name only (not substring/ast.dump match,
    which would false-positive on unrelated classes)."""
    models = {}
    if not tree:
        return models
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_pydantic = any(
                (isinstance(b, ast.Name) and b.id == "BaseModel")
                or (isinstance(b, ast.Attribute) and b.attr == "BaseModel")
                for b in node.bases
            )
            if not is_pydantic:
                continue
            fields = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    type_str = ast.dump(item.annotation) if item.annotation else "Any"
                    default_str = ast.dump(item.value) if item.value else "NONE"
                    fields[item.target.id] = (type_str, default_str)
            models[node.name] = fields
    return models


def get_fastapi_routes(tree):
    """Map (verb, path) -> {func_name, params}. Skip unresolved dynamic route paths."""
    routes = {}
    if not tree:
        return routes
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
                continue
            verb = decorator.func.attr
            if verb not in ("get", "post", "put", "delete", "patch"):
                continue

            path = None
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                path = str(decorator.args[0].value)
            if path is None:
                continue  # Skip dynamic route path expressions

            args = node.args.args
            defaults = node.args.defaults
            first_default_idx = len(args) - len(defaults)

            params = {}
            for i, arg in enumerate(args):
                if arg.arg in ("self", "cls"):
                    continue
                default_node = defaults[i - first_default_idx] if i >= first_default_idx else None
                if is_internal_dependency(arg.annotation, default_node):
                    continue
                type_str = ast.dump(arg.annotation) if arg.annotation else "Any"
                default_str = ast.dump(default_node) if default_node else "NONE"
                params[arg.arg] = (type_str, default_str)

            routes[(verb, path)] = {"func_name": node.name, "params": params}
    return routes


def compute_ast_deltas(source_before, source_after):
    """Compute AST structural diff features between pre-commit and post-commit code."""
    deltas = {
        "fields_removed_count": 0,
        "fields_added_count": 0,
        "fields_type_changed_count": 0,
        "fields_default_changed_count": 0,
        "models_renamed_count": 0,
        "routes_removed_count": 0,
        "routes_added_count": 0,
        "routes_param_changed_count": 0,
        "routes_param_type_changed_count": 0,
        "routes_param_default_changed_count": 0,
    }

    try:
        tree_before = ast.parse(source_before) if source_before else None
        tree_after = ast.parse(source_after) if source_after else None
    except SyntaxError:
        return deltas

    # --- Pydantic model diff ---
    models_before = get_pydantic_fields(tree_before)
    models_after = get_pydantic_fields(tree_after)

    matched_b, matched_a = set(), set()

    for name, fields_bef in models_before.items():
        if name in models_after:
            fields_aft = models_after[name]
            removed = set(fields_bef) - set(fields_aft)
            added = set(fields_aft) - set(fields_bef)
            common = set(fields_bef) & set(fields_aft)

            deltas["fields_removed_count"] += len(removed)
            deltas["fields_added_count"] += len(added)
            deltas["fields_type_changed_count"] += sum(1 for f in common if fields_bef[f][0] != fields_aft[f][0])
            deltas["fields_default_changed_count"] += sum(1 for f in common if fields_bef[f][1] != fields_aft[f][1])

            matched_b.add(name)
            matched_a.add(name)

    unmatched_b = {n: f for n, f in models_before.items() if n not in matched_b}
    unmatched_a = {n: f for n, f in models_after.items() if n not in matched_a}

    for name_b, fields_b in list(unmatched_b.items()):
        best_match, highest_sim = None, 0.0
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
            deltas["fields_type_changed_count"] += sum(1 for f in common if fields_b[f][0] != fields_aft[f][0])
            deltas["fields_default_changed_count"] += sum(1 for f in common if fields_b[f][1] != fields_aft[f][1])
        else:
            deltas["fields_removed_count"] += len(fields_b)

    # --- Route diff ---
    routes_before = get_fastapi_routes(tree_before)
    routes_after = get_fastapi_routes(tree_after)

    matched_r_b, matched_r_a = set(), set()
    for r_key, r_info_b in routes_before.items():
        if r_key in routes_after:
            r_info_a = routes_after[r_key]
            p_b, p_a = r_info_b["params"], r_info_a["params"]

            p_removed = set(p_b) - set(p_a)
            p_added = set(p_a) - set(p_b)
            p_common = set(p_b) & set(p_a)

            deltas["routes_param_changed_count"] += len(p_removed) + len(p_added)
            deltas["routes_param_type_changed_count"] += sum(1 for p in p_common if p_b[p][0] != p_a[p][0])
            deltas["routes_param_default_changed_count"] += sum(1 for p in p_common if p_b[p][1] != p_a[p][1])

            matched_r_b.add(r_key)
            matched_r_a.add(r_key)

    deltas["routes_removed_count"] = len(routes_before) - len(matched_r_b)
    deltas["routes_added_count"] = len(routes_after) - len(matched_r_a)

    return deltas


def _repo_slug_from_url(clone_url):
    """https://github.com/owner/repo.git -> 'owner/repo'"""
    slug = clone_url.rstrip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    parts = slug.split("/")
    return "/".join(parts[-2:])


def repo_depends_on_fastapi(clone_url, headers, default_branch_candidates=("main", "master")):
    slug = _repo_slug_from_url(clone_url)
    for branch in default_branch_candidates:
        for manifest in DEPENDENCY_MANIFEST_CANDIDATES:
            raw_url = f"https://raw.githubusercontent.com/{slug}/{branch}/{manifest}"
            try:
                res = requests.get(raw_url, headers=headers, timeout=8)
                if res.status_code == 200:
                    if DEPENDENCY_PATTERN.search(res.text):
                        return True
                    return False
            except Exception:
                continue
    return True


def get_fastapi_repositories():
    url = f"https://api.github.com/search/repositories?q=fastapi+language:python+stars:>{MIN_STARS}&sort=stars&order=desc"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        items = res.json().get("items", [])
        candidate_urls = [item["clone_url"] for item in items[: MAX_REPOS * 3]]
    except Exception as e:
        print(f"GitHub API query failed ({e}). Using fallback repos.")
        return FALLBACK_REPOS

    filtered = []
    for url_ in candidate_urls:
        if repo_depends_on_fastapi(url_, headers):
            filtered.append(url_)
        else:
            print(f"Skipping (no fastapi/pydantic dependency found): {url_}")
        if len(filtered) >= MAX_REPOS:
            break

    return filtered if filtered else FALLBACK_REPOS


def _file_path(file):
    """Full relative path, not just basename. new_path is None on deletions,
    so fall back to old_path in that case."""
    return file.new_path or file.old_path


def mine_ast_delta_repositories():
    repos = get_fastapi_repositories()
    print(f"\nMining {len(repos)} repositories after dependency filtering.\n")
    dataset = []

    for repo_url in repos:
        print(f"Mining repository: {repo_url}...")
        try:
            miner = Repository(
                repo_url,
                only_no_merge=True,
                since=SINCE_DATE,
                order="reverse",
            )
            commit_count = 0

            for commit in miner.traverse_commits():
                if commit_count >= MAX_COMMITS_PER_REPO:
                    print(f"Reached cap of {MAX_COMMITS_PER_REPO} commits for {repo_url}.")
                    break
                commit_count += 1

                msg_lower = commit.msg.lower()
                # LABEL: derived from commit message text only. Never mix with AST features.
                is_breaking_commit = 1 if any(kw in msg_lower for kw in BREAKING_KEYWORDS) else 0

                for file in commit.modified_files:
                    path = _file_path(file)
                    if not path or not path.endswith(".py"):
                        continue

                    ast_deltas = compute_ast_deltas(file.source_code_before, file.source_code)

                    dataset.append({
                        "repo_name": commit.project_name,
                        "commit_hash": commit.hash,
                        "filepath": path,  # full relative path, not just basename
                        "commit_date": commit.committer_date.isoformat(),

                        # TARGET (y) -- independent of AST features
                        "is_breaking_commit": is_breaking_commit,

                        # FEATURES (X) - Churn Metrics
                        "lines_added": file.added_lines,
                        "lines_deleted": file.deleted_lines,
                        "net_churn": file.added_lines - file.deleted_lines,
                        "code_complexity": file.complexity if file.complexity else 0,

                        # FEATURES (X) - Structural AST Deltas
                        "fields_removed_count": ast_deltas["fields_removed_count"],
                        "fields_added_count": ast_deltas["fields_added_count"],
                        "fields_type_changed_count": ast_deltas["fields_type_changed_count"],
                        "fields_default_changed_count": ast_deltas["fields_default_changed_count"],
                        "models_renamed_count": ast_deltas["models_renamed_count"],
                        "routes_removed_count": ast_deltas["routes_removed_count"],
                        "routes_added_count": ast_deltas["routes_added_count"],
                        "routes_param_changed_count": ast_deltas["routes_param_changed_count"],
                        "routes_param_type_changed_count": ast_deltas["routes_param_type_changed_count"],
                        "routes_param_default_changed_count": ast_deltas["routes_param_default_changed_count"],
                    })

        except Exception as e:
            print(f"Error processing repository {repo_url}: {e}")

    df = pd.DataFrame(dataset)

    before = len(df)
    df = df.drop_duplicates(subset=["commit_hash", "filepath"])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} true duplicate rows (same commit_hash + full filepath).")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} samples to {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    mine_ast_delta_repositories()