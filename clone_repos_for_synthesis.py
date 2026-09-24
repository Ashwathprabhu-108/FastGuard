"""
clone_repos_for_synthesis.py

Your mining script (fastapi_ast_delta_dataset.py) clones repos temporarily
via pydriller and discards them after mining. The synthetic mutation
generator needs actual persistent .py files on disk to mutate, so this
script clones the same repo list into cloned_repos/ and keeps them.

Run this BEFORE synthetic_mutation_generator.py.
"""

import os
import subprocess

CLONE_DIR = "cloned_repos"

# Keep this list in sync with FALLBACK_REPOS / your mined repo list.
# You can also just paste in the exact list your mining run actually used
# (printed to console when you ran fastapi_ast_delta_dataset.py) for an
# exact match.
REPOS = [
    "https://github.com/fastapi/full-stack-fastapi-template",
    "https://github.com/tiangolo/sqlmodel",
    "https://github.com/zhanymkanov/fastapi-best-practices",
    "https://github.com/long2ice/fastapi-cache",
    "https://github.com/dmontagu/fastapi-utils",
    "https://github.com/pydantic/pydantic",
    "https://github.com/fastapi/fastapi",
]


def clone_repos():
    os.makedirs(CLONE_DIR, exist_ok=True)

    for repo_url in REPOS:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        dest = os.path.join(CLONE_DIR, repo_name)

        if os.path.exists(dest):
            print(f"Already cloned, skipping: {repo_name}")
            continue

        print(f"Cloning {repo_url} -> {dest}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, dest],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo_url}: {e.stderr}")


if __name__ == "__main__":
    clone_repos()