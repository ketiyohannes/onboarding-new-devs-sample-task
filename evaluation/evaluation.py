#!/usr/bin/env python3

import sys
import json
import time
import uuid
import platform
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "evaluation" / "reports"

def environment_info():
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

def run_tests():
    try:
        proc = subprocess.run(
            ["pytest", "-s"], cwd=ROOT, capture_output=True, text=True, timeout=120
        )
        output = (proc.stdout + proc.stderr)[:8000]
        return {
            "passed": proc.returncode == 0,
            "return_code": proc.returncode,
            "output": output,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "return_code": -1, "output": "pytest timeout"}

def run_metrics(repo_path: Path):
    return {}

def evaluate(repo_name: str):
    repo_path = ROOT / repo_name
    if not repo_path.exists():
        return {
            "tests": {
                "passed": False,
                "return_code": -1,
                "output": f"{repo_name} not found",
            },
            "metrics": {},
        }
    elif not any(repo_path.glob("**/*.py")):
        return {
            "tests": {
                "passed": False,
                "return_code": -1,
                "output": "No Python files to test",
            }
        }
    tests = run_tests()
    metrics = run_metrics(repo_path)
    return {"tests": tests, "metrics": metrics}

def run_evaluation():
    run_id = str(uuid.uuid4())
    start = datetime.utcnow()
    before = evaluate("repository_before")
    after = evaluate("repository_after")
    comparison = {
        "passed_gate": after["tests"]["passed"],
        "improvement_summary": (
            "After implementation passed all requirement-based tests"
            if after["tests"]["passed"]
            else "After implementation failed one or more requirement-based tests"
        ),
    }
    end = datetime.utcnow()
    return {
        "run_id": run_id,
        "started_at": start.isoformat() + "Z",
        "finished_at": end.isoformat() + "Z",
        "duration_seconds": (end - start).total_seconds(),
        "environment": environment_info(),
        "before": before,
        "after": after,
        "comparison": comparison,
        "success": comparison["passed_gate"],
        "error": None,
    }

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        report = run_evaluation()
    except Exception as e:
        report = {
            "run_id": str(uuid.uuid4()),
            "started_at": None,
            "finished_at": None,
            "duration_seconds": 0.0,
            "environment": environment_info(),
            "before": None,
            "after": None,
            "comparison": None,
            "success": False,
            "error": str(e),
        }
    path = REPORTS / "latest.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {path}")
    print("\n--- repository_before output ---")
    print(report["before"]["tests"]["output"])
    print("\n--- repository_after output ---")
    print(report["after"]["tests"]["output"])
    return 0 if report["success"] else 1

if __name__ == "__main__":
    sys.exit(main())