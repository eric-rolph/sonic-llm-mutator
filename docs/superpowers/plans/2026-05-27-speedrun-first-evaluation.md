# Speedrun-First Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Green Hill Act 1 speedrun performance the primary objective while adding benchmark checks across multiple Sonic states.

**Architecture:** Keep policy execution in `main.py`, isolate scoring in `core/evaluator.py`, and add a focused benchmark CLI that reuses the existing evaluator with a no-vision mutator stub. Mutation memory cleanup remains in `llm/mutator.py`.

**Tech Stack:** Python 3.8 for emulator execution, Python stdlib tests with `unittest`, `gym-retro`, `opencv-python`, and OpenAI-compatible clients.

---

### Task 1: Evaluator Failure Reasons

**Files:**
- Modify: `main.py`
- Test: `tests/test_evaluate_policy.py`

- [ ] **Step 1: Write failing tests for stuck and timeout reasons**

Create `tests/test_evaluate_policy.py` with fake env and policy objects that reproduce stuck termination and max-frame timeout.

- [ ] **Step 2: Run tests and verify failure**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_evaluate_policy -v`

Expected: at least one failure showing the evaluator overwrites a stuck reason as a fatal obstacle.

- [ ] **Step 3: Preserve explicit termination reasons**

Initialize `failure_reason = None` before the loop in `evaluate_policy`. Set it inside stuck and timeout branches. After the loop, only infer fatal or unknown reasons when `failure_reason is None`.

- [ ] **Step 4: Re-run tests**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_evaluate_policy -v`

Expected: all tests in the file pass.

### Task 2: Speedrun-Oriented Fitness

**Files:**
- Modify: `core/evaluator.py`
- Test: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests for scoring priorities**

Create tests proving faster runs with equal distance score higher, distance still dominates early progress, and rings/score are small tie-breakers.

- [ ] **Step 2: Run tests and verify failure**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_evaluator -v`

Expected: failure under the existing low-weight speed formula.

- [ ] **Step 3: Update scoring**

Use distance as the base, add a stronger speed bonus, add a completion bonus near Act 1 goal distance, and reduce rings/score weight.

- [ ] **Step 4: Re-run tests**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_evaluator -v`

Expected: all tests in the file pass.

### Task 3: Benchmark CLI

**Files:**
- Create: `benchmark_policies.py`
- Test: `tests/test_benchmark_policies.py`
- Modify: `README.md`

- [ ] **Step 1: Write formatting and argument tests**

Test benchmark row formatting and default state/policy selection without launching the emulator.

- [ ] **Step 2: Run tests and verify failure**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_benchmark_policies -v`

Expected: import or function-not-found failure.

- [ ] **Step 3: Implement benchmark helpers and CLI**

Add `DEFAULT_STATES`, `DEFAULT_POLICIES`, `load_policy`, `run_benchmark`, `format_results_table`, and a `main()` parser. Use `NoVisionMutator` to avoid model calls.

- [ ] **Step 4: Re-run tests**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_benchmark_policies -v`

Expected: all tests in the file pass.

### Task 4: Semantic Memory Hygiene

**Files:**
- Modify: `llm/mutator.py`
- Test: `tests/test_mutator_memory.py`

- [ ] **Step 1: Write failing tests for memory dedupe and validation**

Test that malformed entries are ignored, duplicate nearby lessons collapse, and relevant lessons are selected by coordinate.

- [ ] **Step 2: Run tests and verify failure**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_mutator_memory -v`

Expected: missing helper failure.

- [ ] **Step 3: Implement small memory helpers**

Add helpers for JSON extraction, lesson normalization, dedupe, and relevant lesson selection. Keep prompt construction compact.

- [ ] **Step 4: Re-run tests**

Run: `.\venv38\Scripts\python.exe -m unittest tests.test_mutator_memory -v`

Expected: all tests in the file pass.

### Task 5: Verification And Publishing

**Files:**
- Modify: `.github/workflows/evaluate_policy.yml`
- Modify: `README.md`

- [ ] **Step 1: Run full local tests**

Run: `.\venv38\Scripts\python.exe -m unittest discover -v`

Expected: all tests pass.

- [ ] **Step 2: Run compile check**

Run: `.\venv38\Scripts\python.exe -m compileall main.py core emulator llm benchmark_policies.py local_ci.py dashboard.py render_video.py`

Expected: exit code 0.

- [ ] **Step 3: Run emulator smoke benchmark**

Run: `.\venv38\Scripts\python.exe benchmark_policies.py --max-frames 900 --states GreenHillZone.Act1 --policies policies/champion_policy.py`

Expected: one formatted benchmark row and exit code 0.

- [ ] **Step 4: Commit, push, and merge**

Stage intended files, commit, push branch to GitHub, open a PR if needed, and merge once checks are satisfactory.
