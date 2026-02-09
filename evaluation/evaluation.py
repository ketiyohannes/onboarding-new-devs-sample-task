import os
import sys
import json
import time
import uuid
import platform
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"

def environment_info():
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform()
    }

def run_tests(repo_name: str):
    """Runs tests against a specific repository folder using PYTHONPATH."""
    repo_path = ROOT / repo_name
    env = os.environ.copy()
    # Ensure the repo and tests directory are in the path
    env["PYTHONPATH"] = str(repo_path) + os.pathsep + env.get("PYTHONPATH", "")
    
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "tests"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )
        duration = time.perf_counter() - start
        passed = proc.returncode == 0
        icon = "✅" if passed else "❌"
        print(f"[{repo_name}] Tests {icon} (Duration: {duration:.2f}s)")
        
        return {
            "passed": passed,
            "return_code": proc.returncode,
            "output": (proc.stdout + proc.stderr)[:8000]
        }
    except subprocess.TimeoutExpired:
        print(f"[{repo_name}] Tests ❌ (Timeout)")
        return {
            "passed": False,
            "return_code": -1,
            "output": "Timeout expired after 120s"
        }

def run_metrics(repo_name: str):
    repo_path = ROOT / repo_name
    target_file = repo_path / "scalar_tensor_ad.py"
    if target_file.exists():
        return {
            "file_size_bytes": target_file.stat().st_size
        }
    return {}

def evaluate(repo_name: str):
    tests = run_tests(repo_name)
    metrics = run_metrics(repo_name)
    return {
        "tests": tests,
        "metrics": metrics
    }

def run_evaluation():
    run_id = str(uuid.uuid4())
    start = datetime.utcnow()
    
    print("\n" + "="*60)
    print("AUTOMATIC DIFFERENTIATION ENGINE - EVALUATION REPORT")
    print("="*60 + "\n")
    

    before = evaluate("repository_before")
    after = evaluate("repository_after")
    
    passed_gate = after["tests"]["passed"]
    improvement = "Refactored code passed all correctness gates." if passed_gate else "Refactored code failed correctness gate."
    
    comparison = {
        "passed_gate": passed_gate,
        "improvement_summary": improvement
    }
    
    end = datetime.utcnow()
    duration_seconds = (end - start).total_seconds()
    
    report = {
        "run_id": run_id,
        "started_at": start.isoformat() + "Z",
        "finished_at": end.isoformat() + "Z",
        "duration_seconds": duration_seconds,
        "environment": environment_info(),
        "before": before,
        "after": after,
        "comparison": comparison,
        "success": passed_gate,
        "error": None
    }
    
    return report

def main():
    # Folder name format: Year-Day-Month 
    now = datetime.now()
    folder_name = now.strftime("%Y-%d-%m")
    
    # Directory to save the report: evaluation/YYYY-DD-MM
    target_date_dir = EVAL_DIR / folder_name
    target_date_dir.mkdir(parents=True, exist_ok=True)
    
    report = run_evaluation()
    
    # Save as report.json in the folder
    report_file_path = target_date_dir / "report.json"
    report_file_path.write_text(json.dumps(report, indent=2))
    
    print("\n" + "="*60)
    print(f"🏁 EVALUATION COMPLETE - Success: {'✅' if report['success'] else '❌'}")
    print(f"📁 Report saved to: {report_file_path}")
    print("="*60 + "\n")
    
    return 0 if report["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
