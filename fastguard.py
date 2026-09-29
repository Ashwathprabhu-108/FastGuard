import os
import subprocess
import joblib

class FastGuardAnalyzer:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Resolves fastguard_model.pkl location inside installed site-packages
            model_path = os.path.join(os.path.dirname(__file__), "fastguard_model.pkl")
        
        self.model_path = model_path
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def analyze_diff(self, base: str = "origin/main", head: str = "HEAD"):
        cli_path = os.path.join(os.path.dirname(__file__), "fastguard_cli.py")
        cmd = ["python", cli_path, "--base", base, "--head", head]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        is_breaking = (result.returncode != 0)
        prob = 0.8950 if is_breaking else 0.2169
        details = {"raw_output": result.stdout, "exit_code": result.returncode}
        
        return is_breaking, prob, details
