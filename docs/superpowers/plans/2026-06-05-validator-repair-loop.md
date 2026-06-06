# Bounded Validator Repair Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair invalid generated policies once using exact validator feedback before emulator evaluation.

**Architecture:** Add a local-only repair API to the mutator and a focused candidate-preparation helper in `main.py`. The helper archives invalid source, makes one repair attempt, validates the repaired result, and returns a normalized result to the existing evaluation loop.

**Tech Stack:** Python 3.8, standard-library file handling, existing OpenAI-compatible micro model client, `unittest`.

---

### Task 1: Local Validator-Feedback Repair

**Files:**
- Modify: `llm/mutator.py`
- Modify: `tests/test_mutator_memory.py`

- [ ] Write a failing test proving repair uses the micro model and exact validator feedback.
- [ ] Implement `repair_policy`.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest tests.test_mutator_memory -v`.

### Task 2: One-Attempt Candidate Preparation

**Files:**
- Modify: `main.py`
- Modify: `tests/test_candidate_handling.py`

- [ ] Write failing tests for valid passthrough, successful repair, invalid-attempt archiving, and failed-repair termination.
- [ ] Implement `prepare_candidate_policy`.
- [ ] Integrate it into the candidate evaluation loop.
- [ ] Run focused candidate-handling tests.

### Task 3: Document and Verify

**Files:**
- Modify: `README.md`

- [ ] Document the bounded local repair loop.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest discover -s tests`.
- [ ] Run `.\venv38\Scripts\ruff.exe check .`.
- [ ] Run `git diff --check` and inspect the final diff.
