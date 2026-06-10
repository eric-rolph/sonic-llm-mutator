# Agentic Failure Diagnosis Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Work task-by-task; run the listed tests after each task before moving on.

**Goal:** On a visual failure (Sonic stuck against geometry or killed by a hazard), let the vision model *interactively* interrogate the failure — seek to moments before death, read authoritative RAM state, and run counterfactual inputs ("would RIGHT,B from 2 seconds earlier clear it?") — then feed that diagnosis into the mutation prompt. Today the model judges one static montage; this upgrade makes failure analysis experimental instead of one-shot.

**Architecture:**

1. **Savestate ring (`core/diagnosis.py`)** — during evaluation, capture `env.em.get_state()` every 60 frames into a 10-slot ring (~10 s window before failure). When a run becomes the working frontier, persist the ring to `artifacts/diagnosis/window/` as raw `.state` blobs plus a `window.json` manifest (frame, x, y, zone, act, rings per snapshot, and the failure frame/x). Disk, not memory: debuggable, and shareable with the MCP sidecar.
2. **Savestate primitives (`emulator/sonic_env.py`)** — `save_emulator_state()` / `load_emulator_state(bytes)` on the wrapper via `unwrapped.em`, refreshing obs/info from `em.get_screen()` + `data.lookup_all()` without stepping. Velocity tracking re-baselines on load.
3. **DiagnosisSession (`core/diagnosis.py`)** — wraps a *dedicated non-recording* env (never the training env, so bk2 recordings are never polluted) plus a persisted window. Operations: `describe_window()`, `view_frame(frames_before_failure)` → state + screenshot file, `try_actions(frames_before_failure, actions, hold_frames)` → counterfactual rollout summary (start/end x-y, max x, rings/lives delta, whether the failure x was passed) + screenshot. Hard caps: ≤300 frames per try, every operation exception-safe.
4. **Tool-driven diagnosis loop (`llm/mutator.py`)** — `diagnose_failure(session, failure_reason, trace)` drives the macro client through OpenAI function calling (max 6 tool rounds): tools `view_frame`, `try_actions`, `finish_diagnosis(report)`. Tool results are text; screenshots attach as follow-up user `image_url` messages (universally supported, unlike images in tool results). Budget exhaustion forces a final no-tools report request. Returns `(report, evidence_screenshot)`; any error returns `None` and the existing montage-only path runs unchanged.
5. **Loop integration (`main.py`)** — diagnosis runs **once per generation** (both candidates mutate the same frontier), cached while the frontier is unchanged so stagnant generations pay nothing. `generate_candidates` / `mutate_policy` accept `diagnosis_report`; the report is embedded in the mutation prompt and the diagnosis evidence screenshot is preferred over the montage.
6. **MCP sidecar parity (`emulator/mcp_server.py`)** — new tools `list_failure_window`, `view_failure_frame`, `try_failure_actions` backed by the same `core/diagnosis.py` and the same persisted window, so a human (or Claude) can replay exactly what the mutator saw.
7. **Safety/config** — `SONIC_AGENTIC_DIAGNOSIS=1` (default on); runs only when a macro client, a persisted window, and an emulator all exist. Every layer degrades to today's behavior on any failure. Diagnosis never touches the training env.

**Tech Stack:** Python 3.8, gym-retro/stable-retro savestates (`em.get_state`/`em.set_state`), OpenAI SDK function calling, `unittest` with stubbed emulator/clients, real-backend smoke coverage in CI.

---

### Task 1: Emulator savestate primitives

**Files:**
- Modify: `emulator/sonic_env.py`
- Modify: `tests/test_sonic_env_backends.py`
- Modify: `tests/test_emulator_smoke.py`

- [x] Add `save_emulator_state()` / `load_emulator_state(state_bytes)` to `SonicEnvWrapper` (via `unwrapped.em`; refresh obs from `em.get_screen()`, info from `data.lookup_all()`, re-baseline velocity).
- [x] Stubbed tests: round-trip through a fake `em`, obs/info refreshed, velocity re-baselined, missing `em` raises a clear error.
- [x] Extend the real-backend smoke test with a savestate round-trip (capture → step 30 → restore → state matches) so CI proves the API on stable-retro and local runs prove gym-retro.
- [x] Run `.\venv38\Scripts\python.exe -m unittest tests.test_sonic_env_backends tests.test_emulator_smoke -v`.

### Task 2: Snapshot ring + persisted failure window

**Files:**
- Create: `core/diagnosis.py`
- Create: `tests/test_diagnosis.py`

- [x] `FailureSnapshotRing(interval=60, capacity=10)` with `record(env, frame, state)` (exception-safe, cadence-gated) and `persist(directory, failure_reason, final_state)` writing `<frame>.state` blobs + `window.json` manifest.
- [x] `load_failure_window(directory)` returning the manifest with verified blob paths; tolerate missing/corrupt files by returning `None`.
- [x] Tests: cadence and capacity, exception-safety when `save_emulator_state` raises, persist/load round-trip, corrupt manifest handling.
- [x] Run `.\venv38\Scripts\python.exe -m unittest tests.test_diagnosis -v`.

### Task 3: DiagnosisSession (seek, view, counterfactual rollouts)

**Files:**
- Modify: `core/diagnosis.py`
- Modify: `tests/test_diagnosis.py`

- [x] `DiagnosisSession(env_factory, window)`: lazy env creation (non-recording), `describe_window()`, `view_frame(frames_before_failure)` (seek nearest snapshot, return state dict + screenshot path under `artifacts/diagnosis/`), `try_actions(frames_before_failure, actions, hold_frames)` (seek, hold action ≤300 frames, summarize movement + whether failure x was passed, screenshot at end), `close()`.
- [x] All operations return error strings instead of raising; a failed step sequence recovers by reseeking.
- [x] Tests with a fake env: nearest-snapshot seeking, rollout summary fields, frame cap enforcement, error recovery, screenshots written.
- [x] Run `.\venv38\Scripts\python.exe -m unittest tests.test_diagnosis -v`.

### Task 4: Snapshot collection during evaluation + frontier window persistence

**Files:**
- Modify: `core/evaluation.py`
- Modify: `main.py`
- Modify: `tests/test_evaluate_policy.py`
- Modify: `tests/test_run_resume.py`

- [x] `evaluate_policy(..., snapshot_sink=None)`: cadence-record into the sink inside the loop (no behavior change when `None`).
- [x] Baseline and candidate evaluations in `main.py` each use a fresh ring; when a run becomes the working frontier (baseline or promotion), persist its ring and carry the window directory in the frontier context.
- [x] Tests: sink receives cadenced snapshots; frontier carries the window dir on promotion and baseline.
- [x] Run `.\venv38\Scripts\python.exe -m unittest tests.test_evaluate_policy tests.test_run_resume -v`.

### Task 5: Tool-calling diagnosis loop in the mutator

**Files:**
- Modify: `llm/mutator.py`
- Create: `tests/test_diagnoser.py`

- [ ] Tool schemas (`view_frame`, `try_actions`, `finish_diagnosis`) and `diagnose_failure(session, failure_reason, trace)` driving `macro_client` chat completions with `tools=`; text tool results + screenshots as follow-up user image messages; max 6 tool rounds then a forced no-tools final report; returns `(report, evidence_screenshot)` or `None` on any error.
- [ ] Tests with a scripted fake OpenAI client: multi-round tool dispatch, message assembly (tool result + image follow-up), forced finish on budget exhaustion, `None` on client error, no macro client → `None` without calls.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest tests.test_diagnoser -v`.

### Task 6: Wire diagnosis into mutation

**Files:**
- Modify: `llm/mutator.py`
- Modify: `main.py`
- Modify: `tests/test_mutator_memory.py`
- Modify: `tests/test_generate_candidates.py`

- [ ] `mutate_policy(..., diagnosis_report=None)` embeds the report in the prompt under an "Agentic Failure Diagnosis" section.
- [ ] `generate_candidates(..., diagnosis_report=None)` forwards it to mutations (not crossovers).
- [ ] `maybe_diagnose_frontier(mutator, session_cache, frontier)` in `main.py`: gates on `SONIC_AGENTIC_DIAGNOSIS`, visual failure, persisted window, macro client; caches `(window key → report)` so an unchanged frontier is diagnosed once; evidence screenshot preferred over the montage for the mutation call.
- [ ] Tests: report lands in the prompt; gating and caching logic; diagnosis errors fall back silently.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest tests.test_mutator_memory tests.test_generate_candidates tests.test_run_resume -v`.

### Task 7: MCP sidecar parity

**Files:**
- Modify: `emulator/mcp_server.py`

- [ ] Add `list_failure_window`, `view_failure_frame`, `try_failure_actions` tools backed by `core/diagnosis.py` and the persisted window, reusing the server's singleton env.
- [ ] Run `.\venv38\Scripts\python.exe -m compileall emulator/mcp_server.py` (mcp package optional locally; keep imports lazy-safe).

### Task 8: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/superpowers/plans/2026-06-09-agentic-failure-diagnosis.md` (check boxes)

- [ ] README: replace "planned next step" wording with the shipped behavior (diagnosis flow, config flag, MCP parity); add `SONIC_AGENTIC_DIAGNOSIS` to `.env.example`.
- [ ] Run `.\venv38\Scripts\python.exe -m unittest discover -s tests`.
- [ ] Run `.\venv38\Scripts\python.exe -m ruff check .`.
- [ ] Push branch, open PR, watch CI (including the real-emulator savestate smoke), merge when green.
