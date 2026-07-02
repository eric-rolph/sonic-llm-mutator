# Sonic LLM Mutator

![Dashboard Screenshot](docs/dashboard.png)
This project uses a Large Language Model (LLM) as a genetic algorithm mutator to learn how to play Sonic the Hedgehog. The system drives Sonic through a retro emulator backend and uses a local CI/CD pipeline script to iteratively test and evolve the Python script that controls Sonic. An optional MCP server exposes the emulator for interactive debugging.

## Watch it in Action

<div align="center">
  <a href="https://www.youtube.com/watch?v=5gRK_w-sXaI">
    <img src="https://img.youtube.com/vi/5gRK_w-sXaI/0.jpg" alt="Sonic Run 227" width="600">
  </a>
  <br>
  <em>A sample run showcasing the AI's learned policy after 227 generations of automated LLM mutation.</em>
</div>

## Architecture

1.  **Emulator Wrapper (`emulator/sonic_env.py`)**: Wraps a retro emulator backend (stable-retro or legacy gym-retro) and exposes Sonic's game state (velocity, coordinates, zone/act, rings) as a plain state dict.
2.  **LLM Mutator (`llm/`)**: Acts as the genetic mutation engine. When a run fails it rewrites the control policy from the failure context (trace, screenshot, semantic memory). It routes heavy visual debugging to a cloud vision model (like OpenRouter or Gemini) and minor tweaks to any local LLM (like Ollama or LM Studio).
3.  **Core Orchestrator (`core/` & `main.py`)**: Runs the current policy, calculates fitness (penalizing stagnation), and manages the automated evolutionary pipeline.
4.  **Policies (`policies/`)**: Contains the generated Python scripts that decide the button presses for each frame. The best evolved policy (`champion_policy.py`) and its extracted skill library (`skills.py`) are committed so a fresh clone can inspect and benchmark the headline result.
5.  **Web Dashboard (`dashboard.py`)**: A live Streamlit interface that tracks fitness progression and visually plays `.mp4` recordings of both the Champion and Latest mutation attempts.
6.  **Agentic Failure Diagnosis (`core/diagnosis.py`)**: On a visual failure, the vision model gets *tools* — it rewinds emulator savestates captured before the failure, inspects the authoritative state, and runs counterfactual input experiments ("would `RIGHT,B` from 2 seconds earlier clear this?"). Its verified findings feed the mutation prompt. See [Agentic Failure Diagnosis](#agentic-failure-diagnosis) below.
7.  **MCP Debugging Sidecar (`emulator/mcp_server.py`)**: A small FastMCP server for poking at the emulator interactively from an MCP client such as Claude — manual play (`reset_game`, `get_game_state`, `get_screenshot`, `step_frames`) plus the *same* failure-window tools the mutator's diagnosis uses (`list_failure_window`, `view_failure_frame`, `try_failure_actions`), so you can replay exactly what the model saw.

## Intelligent Universal LLM Routing

To maximize efficiency and minimize API costs, the mutator (`llm/mutator.py`) intelligently routes its requests based on exactly *how* Sonic failed the previous run. Using the standard OpenAI Python SDK, the pipeline universally supports almost every modern AI engine:

*   **Local LLMs (Micro-Mutations):** A *pure code fault* — an infinite loop caught as a timeout, or any failure where no screenshot is available — is routed to a **local, free LLM** with the failure coordinate trace, to debug the code without visual context.
    *   *Supported:* **LM Studio, Ollama, llama.cpp, vLLM, SGLang, and Apple MLX** (any engine exposing a `/v1` endpoint).
*   **Vision LLMs (Macro-Mutations):** A *visual* failure — Sonic physically **stuck** against level geometry, or **killed** by a hazard (enemy/spike pit) — captures the failure frame and sends it to a **vision model** that acts as the "eyes" to analyze the obstacle and rewrite the policy. This can be a cloud model or a local vision model (e.g. a Gemma/Qwen-VL in LM Studio).
    *   *Supported:* **Google Gemini, Anthropic Claude, OpenAI ChatGPT, Kimi (Moonshot), OpenRouter, or any local vision model.**

    > A 30-generation run surfaced this: originally "stuck" was treated as a blind code bug and sent to the code model, but with **zero** vision calls the agent could never get past Act 2 geometry it couldn't see (a hard plateau). Stuck failures now get eyes, since being blocked is fundamentally a visual problem.

## Verified Escapes: Sweep First, Vision Second

A static death-frame tells the vision model *where* Sonic failed, but it can only guess *why* and *what would have worked*. This pipeline makes escapes **experimental and verified** instead:

1. **Capture**: during every evaluation, whole-machine emulator savestates are captured every 60 frames into a small ring (~10 s of history, `core/diagnosis.py`). Savestates taken while the act's max-x was still improving are **pinned as frontier snapshots** — exempt from the trailing eviction — so when Sonic dies at the frontier and respawns at a checkpoint, the moments *just before the death* survive into the window (without pins, every experiment could only start post-respawn and no escape could ever verify). When a run becomes the working frontier, its window is persisted to `artifacts/diagnosis/window/`.
2. **Mechanical escape sweep** (`core/escape_sweep.py`): before any model is consulted, a battery of canonical Sonic moves — jump/high-jump holds, rolls, and run-up-then-jump sequences over several runway lengths — is replayed from every pinned savestate. Each experiment is milliseconds of emulator compute, so ~40 attempts cost seconds and zero model calls. Toggle with `SONIC_ESCAPE_SWEEP=0`.
3. **Agentic vision diagnosis** (fallback): only when the battery fails is the vision model dropped into an interactive session over the same window (standard OpenAI function calling), told which standard moves already failed, with tools:
   * `view_frame(frames_before_failure)` — rewind a dedicated, non-recording emulator to any captured moment; returns the authoritative RAM state plus a screenshot.
   * `try_actions(frames_before_failure, actions, hold_frames)` / `try_action_sequence(...)` — actually run a counterfactual input and report measured movement, ring/life changes, and whether it beat the frontier.
   * `finish_diagnosis(report)` — submit the findings. Budget: 6 tool calls, then a final report is forced.
4. **Verification is strict**: an escape only counts if it beats the run's frontier x **and Sonic survives it** — experiments track life-loss and keep stepping through a settle window past the scripted input, so a jump that peaks past the frontier while falling into a wider pit is rejected, not compiled.
5. **Compilation, not translation**: a verified escape is compiled *deterministically* into a guard prepended to the working policy (`core/frontier.py`) — the LLM never re-writes its own finding into code. Forward run-ups compile **position-gated** (hold the run-up until the measured launch x, then time-replay the jump), which live testing showed replays faithfully where band-anchored time replay missed precision jumps.

One sweep/diagnosis serves the whole generation and is cached until the frontier changes, so stagnant generations pay nothing. Disable diagnosis with `SONIC_AGENTIC_DIAGNOSIS=0`; on any error (no vision key, no savestate support, provider without tool calling) the pipeline silently falls back to the previous one-shot montage behavior.

The same persisted window is exposed through the MCP sidecar (`list_failure_window`, `view_failure_frame`, `try_failure_actions`), so you can interactively replay exactly the failure the mutator was reasoning about.

## How This Differs from Other LLM Game Agents

LLM-driven game agents broadly fall into two camps:

* **LLM-in-the-loop** — a vision-language model perceives the screen and chooses an action every turn or frame, so the LLM *is* the runtime policy. Most current work is here: harnesses and benchmarks such as [GamingAgent](https://github.com/lmgame-org/GamingAgent), [lmgame-Bench](https://arxiv.org/abs/2505.15146), and [Orak](https://arxiv.org/abs/2506.03610) (and the well-known "Claude/Gemini plays Pokémon" runs). Play is slow and costs an API call per decision.
* **LLM-as-code-evolver** — the LLM evolves a standalone program *offline*; the evolved program then plays at native speed. This is the lineage of [Evolution through Large Models (ELM)](https://arxiv.org/abs/2206.08896), [FunSearch](https://www.nature.com/articles/s41586-023-06924-6), and — for games specifically — [Learning Game-Playing Agents with Generative Code Optimization](https://arxiv.org/abs/2508.19506) (Atari).

This project sits firmly in the **code-evolver** camp, so the paradigm itself is not new. What we believe is a distinctive combination (we are not aware of another open-source project pulling all of these together):

1. **A real-time reflex platformer, not math or turn-based games.** The evolved `get_action(state)` runs at 60 fps on a momentum-driven Genesis platformer (Sonic via gym-/stable-retro), where most code-evolution work targets static optimization (FunSearch) or simpler Atari/turn-based games.
2. **Failure-conditioned, two-tier model routing** (see above): visual failures (Sonic stuck against geometry or killed by a hazard) go to a vision model from the failure frame; pure code faults (timeouts/infinite loops) go to a *local, free* LLM. Vision is used for **diagnosis at failure points**, not per-frame perception.
3. **Local-first.** The bulk of mutations run on a local model, with the cloud VLM called only on visual failures — so a full evolutionary run is nearly free.
4. **A hybrid of three methods in one loop** — a [Voyager](https://arxiv.org/abs/2305.16291)-style skill library, FunSearch-style crossover over a diversity-preserving policy pool, and VLM failure analysis — with a live dashboard and a continuous multi-act play-through fitness.

The payoff of evolving code rather than playing in the loop: the resulting policy is **fast, ~zero-marginal-cost at runtime, deterministic, and human-readable Python** you can diff to see exactly what it learned — versus an in-loop agent that re-pays latency and API cost on every frame.

### Efficiency: evolved policy vs. in-loop VLM (measured)

Measured on this repo's champion clearing Green Hill Act 1 (legacy gym-retro backend), versus a *projected* in-loop VLM agent over the same frames:

| scenario | API calls / run | $ / run | wall-clock / run | frames / $ |
|---|---|---|---|---|
| **evolved policy (runtime)** | **0** | **$0** | **~1 s** | **∞** |
| in-loop VLM (1 call / 12 frames) | 274 | $0.82 | ~7 min | ~4,000 |
| in-loop VLM (1 call / 4 frames) | 822 | $2.47 | ~20 min | ~1,300 |

The evolved `get_action` clears Act 1 in ~3,300 frames of local compute (~1 s headless on this machine; ~55 s even at real-time 60 fps) with **zero** API calls. An in-loop agent re-pays one model call per decision **every run**: at a coarse one-decision-per-12-frames cadence with $0.003/call + 1.5 s latency, that is ~$0.82 and ~7 minutes per Act-1 run — and it scales linearly with every run thereafter. The evolved side is bounded by emulator speed; the in-loop side is bounded by API latency.

Because training here is **local-first** (mutations run on a free local LLM), the one-time evolutionary cost is ≈ $0, so the evolved policy is cheaper from the very first run.

Reproduce or plug in your provider's numbers (no paid API is called — it is an explicit cost model in [`core/efficiency.py`](core/efficiency.py)):

```bash
python efficiency_report.py --measure                          # measure the evolved side live
python efficiency_report.py --usd-per-call 0.005 --latency 2.0  # project in-loop at your prices
```

*Stated honestly:* an in-loop VLM needs **no training** and generalizes zero-shot to new games — that is its real advantage. The code-evolver trades up-front, game-specific evolution for a runtime that is fast, deterministic, inspectable, and ~free to re-run. The in-loop figures above are a transparent projection; `--usd-per-call`, `--latency`, and `--frames-per-decision` are all configurable.

## Resilience Features

To ensure the pipeline can run continuously without manual intervention:
1.  **Policy Preflight Validation**: Before generated Python is imported, an AST validator checks syntax, requires a top-level `get_action(state)`, restricts imports to the optional `policies.skills` library, rejects executable top-level statements, and blocks obvious filesystem/code-execution builtins. Invalid candidates receive a fitness score of `0.0` and the pipeline continues.
2.  **Bounded Validator Repair**: A candidate that fails preflight is archived with its exact validator error, then sent to the local code model for one targeted repair. The repaired source is validated once and either evaluated normally or scored as a load failure; repairs never recurse and never invoke the vision model.
3.  **Runaway-Policy Timeout**: Each `get_action` call runs on a dedicated daemon worker thread (`core/policy_runner.py`) with a hard wall-clock timeout. If an evolved policy contains an infinite loop, the candidate is abandoned and scored as a failure instead of spinning a CPU core forever or hanging the process on exit.
4.  **Stateful Emulator Recording**: We manually enforce video buffer flushing by calling an extra `env.reset()` before teardown, ensuring that the emulator correctly writes the `.bk2` video files to disk even if the episode is manually terminated early.
5.  **Aggressive Cache Breaking**: Mutator prompts are seeded with a randomized cryptographically secure string to prevent local LLMs from entering endless prompt-caching loops.

> ⚠️ **Security note:** this pipeline executes model-generated code and ingests model-/operator-touchable artifact files. There are **three** untrusted inputs: (1) generated **policy** code (`get_action`), run under a restricted-builtins namespace in a memory-capped, timeout-bounded child process; (2) generated **skills** code (`policies/skills.py`), now run under the *same* restricted namespace as policies; and (3) **artifact files** — diagnosis savestate windows (`artifacts/diagnosis/`), the vision cache, and the semantic bank — which the loaders treat as data but which originate from model output or shared directories. The AST preflight + restricted builtins + path containment + the runtime timeout are layered defenses, **not a complete sandbox**: a sufficiently clever jailbroken model could still find a gap. Run the pipeline only in a trusted/isolated environment, treat any `policies/` or `artifacts/` from an untrusted source as executable/attacker input, and review generated code before reusing it elsewhere.
>
> **Containerized runs:** the [Dockerfile](Dockerfile) builds a non-root image. Harden the run further with `--read-only --tmpfs /tmp`, `--cap-drop=ALL`, and `--network=none` (drop the last only if your model endpoint is remote). Mount only what must persist (`artifacts/`, `policies/`) writable; mount the rest read-only.

## Setup

### Linux / WSL / Docker (recommended: Python 3.10-3.12 + stable-retro)

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements-linux.txt
python -m stable_retro.import /path/to/your/roms
python main.py
```

Or use the container (see the [Dockerfile](Dockerfile) header for the full
run command, including how to reach a model server on the host):

```bash
docker build -t sonic-llm-mutator .
docker run --rm -v "$PWD:/app" -v /path/to/roms:/roms --env-file .env \
    sonic-llm-mutator sh -c "python -m stable_retro.import /roms && python main.py"
```

### Windows (legacy: Python 3.8 + gym-retro)

`gym-retro` does not publish Windows wheels beyond Python 3.8 (which is EOL),
so native Windows stays on the pinned legacy environment — prefer WSL above
for new setups.

1.  Install dependencies:
    ```bash
    uv venv --python 3.8 venv
    .\venv\Scripts\Activate.ps1
    uv pip install -r requirements.txt              # core training loop
    uv pip install -r requirements-dashboard.txt    # optional: Streamlit dashboard
    uv pip install -r requirements-dev.txt          # optional: run tests + ruff
    ```
2.  Import the Sonic the Hedgehog ROM:
    ```bash
    python -m retro.import /path/to/your/roms
    ```

### Configure model endpoints

Copy [.env.example](.env.example) to `.env` and fill it in (`run_pipeline.ps1`
loads it automatically; on Linux source it with `set -a; . ./.env; set +a`),
or export the variables directly (PowerShell example):
    ```bash
    # Cloud Vision Provider (OpenRouter, Gemini, Anthropic, OpenAI, Kimi)
    $env:MACRO_API_KEY="your_api_key_here"
    $env:MACRO_BASE_URL="https://openrouter.ai/api/v1" # Or https://generativelanguage.googleapis.com/v1beta/openai/ for Gemini
    $env:MACRO_MODEL="anthropic/claude-3.5-sonnet"     # Or gemini-2.5-pro, gpt-4o, etc.

    # Local Code Provider (LM Studio, Ollama, llama.cpp, vLLM, Apple MLX)
    $env:MICRO_BASE_URL="http://localhost:11434/v1"    # Ollama default. Use 1234 for LM Studio.
    $env:MICRO_MODEL="qwen2.5-coder"                   # Your locally loaded model
    ```
    **Fully local (no cloud key):** point the macro tier at the same local server
    if your model has vision (e.g. a Gemma/Qwen-VL in LM Studio), and disable the
    proactive per-frame vision polling so it doesn't compete with code mutations
    on the one shared model — the death-frame analysis still runs on real failures:
    ```bash
    $env:MACRO_API_KEY="lm-studio"; $env:MACRO_BASE_URL="http://localhost:1234/v1"; $env:MACRO_MODEL="your-vision-model"
    $env:SONIC_PROACTIVE_VISION="0"   # skip proactive polling on a shared local endpoint
    ```
### Run

Run the evolutionary pipeline:
```bash
python main.py
```
For a bounded resume test, `--generations` means additional generations
from the current history endpoint:
```powershell
.\run_pipeline.ps1 -Generations 15 -Frames 12000
# Equivalent:
python main.py --generations 15 --frames 12000
```
Run the live dashboard in a separate terminal:
```bash
streamlit run dashboard.py
```

## Development

The unit tests stub the emulator, so they run quickly without `gym-retro`/`stable-retro` installed:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests
ruff check .
```

CI runs the same lint + test steps on Python 3.8, 3.11 and 3.12, plus a real
stable-retro emulator smoke test against the bundled homebrew Airstriker ROM
(`.github/workflows/evaluate_policy.yml`).

## Speedrun-First, Continuous Play-Through

Training starts at Green Hill Zone Act 1, but evaluation is a **continuous play-through**: when a policy clears an act, the game advances to the next one and the runner keeps going (detected via the emulator's `zone`/`act` values). Fitness rewards, in order of weight: **acts cleared**, then total distance, then speed (fewer frames), with rings/score as small tie-breakers and a completion bonus for reaching the current act's end zone. Per-act progress is reset on each transition so the x-coordinate dropping back to ~0 in the next act is not mistaken for getting stuck.

This rewards a policy that *generalizes* across levels rather than one overfit to Act 1's geometry. `state['zone']` and `state['act']` are exposed to the policy so it can branch per level (e.g. Labyrinth's water sections). The training frame budget (`max_frames`) spans several acts; weak candidates still terminate early via stuck-detection, so the larger budget only costs wall-clock on genuinely strong runs.

Mutation prompts receive compact frame traces with position, velocity, zone/act, rings, vision context, and the action taken. Fatal visual failures also use a small recent-frame montage when screenshots are available, giving the macro model more context than a single final frame.

The policy pool keeps a small amount of action-signature diversity instead of pruning strictly by score, so crossover has access to policies with different controller habits.

Every evaluated candidate is also retained in a structured population archive
under `artifacts/population/`, including candidates that do not beat the current
working policy. The archive deduplicates identical policy code, preserves the
best observed result and failure context, and tracks behavior/obstacle
specialists. Crossover parents are selected from this archive with a P-UCB-style
score that balances measured fitness with an exploration bonus for policies
that have been sampled less often. The evaluated working champion is admitted
before the first generation so early crossovers retain a strong parent.
Stagnation resets exploration context without discarding that champion. The
smaller legacy policy pool remains a fallback while a new archive is being
populated.

When the working policy's trace proves it is repeatedly stationary at one
zone/act/x frontier, one candidate slot receives a deterministic narrow
recovery guard while the other remains available for LLM mutation or
crossover. This preserves all established behavior and gives the search a
hill-climbing path around full-policy rewrites that would otherwise regress
earlier acts before fixing the current frontier.

## Emulator Backends

The runtime supports both the maintained `stable-retro` API and the existing `gym-retro` install. The default backend is `auto`: it tries the modern `stable_retro` import first and falls back to legacy `retro` when stable-retro is not installed.

Use `SONIC_RETRO_BACKEND` or the benchmark CLI flag to force a backend:

```powershell
$env:SONIC_RETRO_BACKEND="legacy"  # or "stable" / "auto"
.\venv38\Scripts\python.exe benchmark_policies.py --backend legacy --max-frames 900 --states GreenHillZone.Act1 --policies policies/champion_policy.py
```

Current native Windows Python 3.8 testing keeps `gym-retro` as the safe fallback because `stable-retro` 1.0.0 does not publish a Windows wheel for this environment. Run `--backend stable` in WSL/Linux or another environment where `pip install stable-retro` succeeds before switching training runs fully to stable-retro.

Use the benchmark CLI to compare policies across the primary speedrun target and broader generalization checks:

```bash
.\venv38\Scripts\python.exe benchmark_policies.py --max-frames 5000
```

Use `--action-repeat` to experiment with frame-skip style evaluation without changing the default training behavior:

```bash
.\venv38\Scripts\python.exe benchmark_policies.py --max-frames 5000 --action-repeat 3
```

For a quick emulator smoke test:

```bash
.\venv38\Scripts\python.exe benchmark_policies.py --backend legacy --max-frames 900 --states GreenHillZone.Act1 --policies policies/champion_policy.py
```
