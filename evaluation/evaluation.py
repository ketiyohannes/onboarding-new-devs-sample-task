#!/usr/bin/env python3

import sys
import os
import json
import uuid
import platform
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "evaluation" / "reports"
TESTS_DIR = ROOT / "tests"

# -------------------------
# Environment
# -------------------------
def run_tests(repo_path: Path, repo_name: str):
    test_file = TESTS_DIR / "test_tensor.py"

    if not test_file.exists():
        return {
            "passed": False,
            "return_code": -1,
            "output": f"tests file not found for {repo_name}",
        }
    elif not any(repo_path.glob("**/*tensor.py")):
        
        return {
            "passed": False,
            "return_code": -1,
            "output": f"tests file not found for {repo_name}",
        }
    
    # Skip tests for empty before repo
    if repo_name == "repository_before" and (not repo_path.exists() or not any(repo_path.iterdir())):
        return {
            "passed": False,
            "return_code": -1,
            "output": f"{repo_name} is empty or missing",
        }

    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)  # <-- Always ROOT, so imports in tests work

        proc = subprocess.run(
            ["python", str(test_file)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        return {
            "passed": proc.returncode == 0,
            "return_code": proc.returncode,
            "output": (proc.stdout + proc.stderr)[:8000],
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "return_code": -1,
            "output": "test execution timeout",
        }

    except Exception as e:
        return {
            "passed": False,
            "return_code": -1,
            "output": f"test execution error: {e}",
        }


# -------------------------
# Metrics (Optional)
# -------------------------
def run_metrics(repo_path: Path):
    return {}

# -------------------------
# Repository Evaluation
# -------------------------
def evaluate_repo(repo_name: str):
    repo_path = ROOT / repo_name

    # Handle empty/missing repo gracefully
    if not repo_path.exists() or not any(repo_path.iterdir()):
        return {
            "tests": {
                "passed": False,
                "return_code": -1,
                "output": f"{repo_name} is empty or missing",
            },
            "metrics": {},
        }

    tests_result = run_tests(repo_path, repo_name)  # <-- pass repo_name
    metrics_result = run_metrics(repo_path)

    return {
        "tests": tests_result,
        "metrics": metrics_result,
    }


# -------------------------
# Core Evaluation
# -------------------------

def environment_info():
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

def run_evaluation():
    run_id = str(uuid.uuid4())
    started_at = datetime.utcnow()

    before = evaluate_repo("repository_before")
    after = evaluate_repo("repository_after")

    passed_gate = after["tests"]["passed"]  # only after repo decides success

    comparison = {
        "passed_gate": passed_gate,
        "improvement_summary": (
            "After implementation passed correctness tests"
            if passed_gate
            else "After implementation failed correctness tests"
        ),
    }

    

    finished_at = datetime.utcnow()

    return {
        "run_id": run_id,
        "started_at": started_at.isoformat() + "Z",
        "finished_at": finished_at.isoformat() + "Z",
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "environment": environment_info(), # type: ignore
        "before": before,
        "after": after,
        "comparison": comparison,
        "success": passed_gate,
        "error": None,
    }

# -------------------------
# Entry Point
# -------------------------
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = run_evaluation()
    report_path = REPORTS_DIR / "latest.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Evaluation report written to {report_path}")

    return 0 if report["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
