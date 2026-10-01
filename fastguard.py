"""
fastguard.py — FastGuard Analyzer
==================================

Programmatic wrapper around ``fastguard_cli.py`` that executes the CLI as a
subprocess and dynamically extracts the exact breaking-change probability
from the model's Random Forest inference output.

No hardcoded probabilities — the value is always parsed from the live
``predict_proba()`` result produced by the CLI.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys


class FastGuardAnalyzer:
    """High-level API for running FastGuard breaking-change detection.

    Executes ``fastguard_cli.py`` against a git diff range and parses the
    machine-readable ``Probability: <float>`` line from stdout to obtain the
    exact model inference result.
    """

    # Regex to locate the machine-parseable probability line emitted by the CLI.
    _PROB_PATTERN: re.Pattern[str] = re.compile(
        r"^Probability:\s+([\d.]+)", re.MULTILINE
    )

    def __init__(self, model_path: str | None = None) -> None:
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "fastguard_model.pkl",
            )
        self.model_path: str = model_path

    def analyze_diff(
        self,
        base: str = "origin/main",
        head: str = "HEAD",
    ) -> tuple[bool, float, dict]:
        """Run the FastGuard CLI and return structured inference results.

        Parameters
        ----------
        base : str
            Base git ref for the diff (default ``"origin/main"``).
        head : str
            Head git ref for the diff (default ``"HEAD"``).

        Returns
        -------
        tuple[bool, float, dict]
            ``(is_breaking, prob, details)`` where:

            - *is_breaking* — ``True`` when ``prob >= 0.50``.
            - *prob* — the exact ``P(breaking)`` returned by the Random Forest
              model's ``predict_proba()`` call.
            - *details* — dict with ``raw_output`` (full CLI stdout) and
              ``exit_code``.
        """
        cli_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fastguard_cli.py",
        )

        cmd = [
            sys.executable,
            cli_path,
            "--base", base,
            "--head", head,
            "--model", self.model_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # --- Parse the exact probability from CLI stdout ---
        prob: float = 0.0
        for line in result.stdout.splitlines():
            match = self._PROB_PATTERN.match(line)
            if match:
                prob = float(match.group(1))
                break

        is_breaking: bool = prob >= 0.50

        details: dict = {
            "raw_output": result.stdout.strip(),
            "exit_code": result.returncode,
        }

        return is_breaking, prob, details
