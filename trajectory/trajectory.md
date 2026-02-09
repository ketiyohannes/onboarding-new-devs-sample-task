# Autodiff From-Scratch Engineering Trajectory

## Project Overview

**Objective**: Build a minimal automatic differentiation engine in Python from scratch.
The system should model **scalar-valued tensors** that can participate in mathematical operations while automatically building a **computational graph**. Using this graph, the system supports **reverse-mode backpropagation** to compute gradients.

**Final Result**: ✅ **Functional Minimal Autodiff Engine**

* Tensor class representing scalar value and gradient
* Fully functional computational graph for forward and backward passes
* Operations supported: `+`, `-`, `*`, `/`, `**`, `tanh`, `relu`, `sigmoid`
* Gradient accumulation for multiple paths in the graph
* Single-file Python implementation using **only built-in features**

---

## Step-by-Step Development Journey

### Step 1: Understanding the Problem

* Build a minimal, educational **autodiff engine**.
* Engine allows scalar computations and automatically computes gradients.
* Challenge: **start from scratch**, no external libraries.
* `repository_before` was empty; everything implemented from zero.

---

### Step 2: Designing the Tensor Class

* Created `Tensor` class in `repository_after/tensor.py`.
* Each tensor stores:

  * `.value` → scalar value
  * `.grad` → gradient
  * References to parent tensors → computational graph
  * `_backward()` function → computes local gradients
* Implemented **operator overloading** for: `+`, `-`, `*`, `/`, `**`
* Implemented **activation functions**: `tanh()`, `relu()`, `sigmoid()`

---

### Step 3: Implementing Backpropagation

* Added `backward()` method:

  * Topological traversal of computational graph
  * Chain rule applied for gradients
  * Supports gradient accumulation for tensors in multiple paths
* Tested small chains to verify forward and backward passes.

---

### Step 4: Writing Tests

* Created `tests/test_tensor.py` to validate:

  * Forward operations
  * Backward gradient computation
  * Activation functions and chaining
* Initial tests: scalar operation checks.

---

### Step 5: Evaluation Metrics Challenge

* Used `evaluation/evaluation.py` for automated evaluation.

**Problems faced**:

1. `repository_before` was empty → `ModuleNotFoundError`
2. Evaluation expected both repos to exist and pass tests
3. `run_tests()` missing correct `repo_path` argument → `TypeError`
4. `environment_info()` missing → `NameError`
5. `tensor.py` file not detected properly → failed test detection

**Solutions**:

* Modified `evaluate_repo()` to handle empty/missing repo:

```python
if not repo_path.exists() or not any(repo_path.glob("**/*tensor.py")):
    return {
        "tests": {
            "passed": False,
            "return_code": -1,
            "output": f"tests file not found for {repo_name}",
        },
        "metrics": {},
    }
```

* Only **after repository** determines evaluation success.
* Defined `environment_info()` consistently.
* Handled subprocess errors, timeouts, stdout/stderr capture.

---

### Step 6: Creating Patch

* Created a diff patch for reproducibility:

```bash
git diff > patch/diff.patch
```

* Ensures anyone can reproduce the changes from the empty repo.

---

### Step 7: Running Evaluation

* Ran:

```bash
docker-compose run --rm autodiff python evaluation/evaluation.py
```

* Evaluation generated:

  * JSON report at `evaluation/reports/latest.json`
  * Passed/failed status for before/after repos
  * Comparison summary: `After implementation passed correctness tests`

---

### Step 8: Lessons Learned

1. **Starting from scratch** is challenging but teaches the core principles of autodiff.
2. **Evaluation metrics must be robust**: handle missing or empty repositories gracefully.
3. Python’s built-in features are enough for minimal autodiff.
4. **Patch creation** ensures reproducibility and traceability.

---

## Folder Structure Recap

```
.
├── docker-compose.yml
├── Dockerfile
├── evaluation
│   ├── evaluation.py
│   └── reports
│       └── latest.json
├── evaluation.sh
├── patch
│   └── diff.patch
├── README.md
├── repository_after
│   └── tensor.py
├── repository_before
├── requirements.txt
├── tests
│   └── test_tensor.py
└── trajectory
    └── trajectory.md
```

---

## Final Outcome

* ✅ Minimal autodiff engine fully functional
* ✅ Forward and backward computations correct
* ✅ Gradient accumulation verified
* ✅ Evaluation framework robust to empty `repository_before`
* ✅ JSON evaluation report generated successfully
* ✅ Patch created for reproducibility

**Conclusion**: This project demonstrates the **full workflow from empty repo → tensor implementation → tests → evaluation → patch**, keeping the system minimal and educational.
