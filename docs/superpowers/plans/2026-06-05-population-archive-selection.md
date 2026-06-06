# Population Archive and Exploration-Aware Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every evaluated policy and use exploration-aware archived parents for crossover.

**Architecture:** Add a focused `core.population` module for structured persistence, descriptors, obstacle clustering, and P-UCB selection. Integrate it into candidate generation and the evaluation loop while retaining the legacy policy pool as a fallback.

**Tech Stack:** Python 3.8, standard-library JSON/hash/path utilities, `unittest`.

---

### Task 1: Define Population Archive Behavior

**Files:**
- Create: `tests/test_population.py`
- Create: `core/population.py`

- [ ] Write failing tests for obstacle keys, duplicate-policy updates, specialist retention, and exploration-aware selection.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest tests.test_population -v` and confirm failures are caused by the missing module.
- [ ] Implement the smallest archive and selection API that satisfies the tests.
- [ ] Re-run `.\venv38\Scripts\python.exe -m unittest tests.test_population -v`.

### Task 2: Add Policy Preflight Validation

**Files:**
- Create: `core/policy_validator.py`
- Create: `tests/test_policy_validator.py`
- Modify: `main.py`

- [ ] Write failing tests for missing contracts, restricted imports, and dangerous builtins.
- [ ] Add AST validation before `load_policy` imports generated code.
- [ ] Verify all current champion, working, and pooled policies pass validation.

### Task 3: Integrate Parent Selection

**Files:**
- Modify: `main.py`
- Modify: `tests/test_generate_candidates.py`

- [ ] Write failing tests proving archived parent selection is preferred and legacy pool fallback remains available.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest tests.test_generate_candidates -v`.
- [ ] Add an optional parent-selector callback to `generate_candidates`.
- [ ] Re-run the focused tests.

### Task 4: Record Every Evaluation

**Files:**
- Modify: `main.py`
- Modify: `tests/test_candidate_handling.py`

- [ ] Write a failing integration-focused test for recording non-promoted candidate metadata.
- [ ] Add a small candidate-recording helper and call it after every evaluation.
- [ ] Ensure archive failures are logged but never stop training.
- [ ] Re-run focused tests.

### Task 5: Verify

**Files:**
- Modify: `README.md`

- [ ] Document the persistent population archive and exploration-aware selection.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest discover -s tests`.
- [ ] Run `.\venv38\Scripts\ruff.exe check .`.
- [ ] Inspect `git diff --check` and the final diff.
