# Sonic LLM Mutator

![Dashboard Screenshot](docs/dashboard.png)
This project uses a Large Language Model (LLM) as a genetic algorithm mutator to learn how to play Sonic the Hedgehog. The system drives Sonic through a retro emulator backend, exposes game state through an MCP server, and uses a local CI/CD pipeline script to iteratively test and evolve the Python script that controls Sonic.

## Watch it in Action

<div align="center">
  <a href="https://www.youtube.com/watch?v=5gRK_w-sXaI">
    <img src="https://img.youtube.com/vi/5gRK_w-sXaI/0.jpg" alt="Sonic Run 227" width="600">
  </a>
  <br>
  <em>A sample run showcasing the AI's learned policy after 227 generations of automated LLM mutation.</em>
</div>

## Architecture

1.  **Emulator MCP Server (`emulator/`)**: Wraps a retro emulator backend and exposes Sonic's game state (velocity, coordinates, surrounding tiles) as discrete tools.
2.  **LLM Mutator (`llm/`)**: Acts as the genetic mutation engine. It queries the MCP server when Sonic dies to understand the failure context, then rewrites the control policy. It routes heavy visual debugging to a cloud vision model (like OpenRouter or Gemini) and minor tweaks to any local LLM (like Ollama or LM Studio).
3.  **Core Orchestrator (`core/` & `main.py`)**: Runs the current policy, calculates fitness (penalizing stagnation), and manages the automated evolutionary pipeline.
4.  **Policies (`policies/`)**: Contains the generated Python scripts that decide the button presses for each frame.
5.  **Web Dashboard (`dashboard.py`)**: A live Streamlit interface that tracks fitness progression and visually plays `.mp4` recordings of both the Champion and Latest mutation attempts.

## Intelligent Universal LLM Routing

To maximize efficiency and minimize API costs, the mutator (`llm/mutator.py`) intelligently routes its requests based on exactly *how* Sonic failed the previous run. Using the standard OpenAI Python SDK, the pipeline universally supports almost every modern AI engine:

*   **Local LLMs (Micro-Mutations):** If Sonic fails due to getting "stuck" or "timing out", the pipeline assumes this is a physics or logic bug in the Python code. It passes the failure coordinate trace to a **local, free LLM** to debug the code without needing visual context. 
    *   *Supported:* **LM Studio, Ollama, llama.cpp, vLLM, SGLang, and Apple MLX** (any engine exposing a `/v1` endpoint).
*   **Cloud LLMs (Macro-Mutations):** If Sonic dies to a fatal hazard (like an enemy or a spike pit), the pipeline captures a screenshot of the death frame. This is sent to a heavy-duty **cloud vision model**, which acts as the "eyes" to visually analyze the level architecture and rewrite the policy.
    *   *Supported:* **Google Gemini, Anthropic Claude, OpenAI ChatGPT, Kimi (Moonshot), and OpenRouter.**

## How This Differs from Other LLM Game Agents

LLM-driven game agents broadly fall into two camps:

* **LLM-in-the-loop** — a vision-language model perceives the screen and chooses an action every turn or frame, so the LLM *is* the runtime policy. Most current work is here: harnesses and benchmarks such as [GamingAgent](https://github.com/lmgame-org/GamingAgent), [lmgame-Bench](https://arxiv.org/abs/2505.15146), and [Orak](https://arxiv.org/abs/2506.03610) (and the well-known "Claude/Gemini plays Pokémon" runs). Play is slow and costs an API call per decision.
* **LLM-as-code-evolver** — the LLM evolves a standalone program *offline*; the evolved program then plays at native speed. This is the lineage of [Evolution through Large Models (ELM)](https://arxiv.org/abs/2206.08896), [FunSearch](https://www.nature.com/articles/s41586-023-06924-6), and — for games specifically — [Learning Game-Playing Agents with Generative Code Optimization](https://arxiv.org/abs/2508.19506) (Atari).

This project sits firmly in the **code-evolver** camp, so the paradigm itself is not new. What we believe is a distinctive combination (we are not aware of another open-source project pulling all of these together):

1. **A real-time reflex platformer, not math or turn-based games.** The evolved `get_action(state)` runs at 60 fps on a momentum-driven Genesis platformer (Sonic via gym-/stable-retro), where most code-evolution work targets static optimization (FunSearch) or simpler Atari/turn-based games.
2. **Failure-conditioned, two-tier model routing** (see above): visual deaths go to a cloud vision model from the death frame; "stuck"/"timeout" code bugs go to a *local, free* LLM. Vision is used for **diagnosis at failure points**, not per-frame perception.
3. **Local-first.** The bulk of mutations run on a local model, with the cloud VLM called only on visual failures — so a full evolutionary run is nearly free.
4. **A hybrid of three methods in one loop** — a [Voyager](https://arxiv.org/abs/2305.16291)-style skill library, FunSearch-style crossover over a diversity-preserving policy pool, and VLM failure analysis — with a live dashboard and a continuous multi-act play-through fitness.

The payoff of evolving code rather than playing in the loop: the resulting policy is **fast, ~zero-marginal-cost at runtime, deterministic, and human-readable Python** you can diff to see exactly what it learned — versus an in-loop agent that re-pays latency and API cost on every frame.

## Resilience Features

To ensure the pipeline can run continuously without manual intervention:
1.  **Syntax Error Sandboxing**: The dynamic Python code loader is wrapped in error handling. If the LLM generates invalid code (e.g., a SyntaxError), the pipeline catches it, assigns the candidate a fitness score of `0.0`, and continues running seamlessly.
2.  **Runaway-Policy Timeout**: Each `get_action` call runs on a dedicated daemon worker thread (`core/policy_runner.py`) with a hard wall-clock timeout. If an evolved policy contains an infinite loop, the candidate is abandoned and scored as a failure instead of spinning a CPU core forever or hanging the process on exit.
3.  **Stateful Emulator Recording**: We manually enforce video buffer flushing by calling an extra `env.reset()` before teardown, ensuring that the emulator correctly writes the `.bk2` video files to disk even if the episode is manually terminated early.
4.  **Aggressive Cache Breaking**: Mutator prompts are seeded with a randomized cryptographically secure string to prevent local LLMs from entering endless prompt-caching loops.

> ⚠️ **Security note:** evolved policies are arbitrary LLM-generated Python that is `exec`'d **in-process** with your full user privileges (`load_policy` in `main.py`). The timeout above bounds runtime, but it does **not** sandbox filesystem or network access. Run the pipeline only in a trusted/isolated environment (e.g. a container or VM), and review generated `policies/` before reusing them elsewhere.

## Setup

1.  Install dependencies:
    ```bash
    uv venv --python 3.8 venv
    .\venv\Scripts\Activate.ps1
    uv pip install -r requirements.txt              # core training loop
    uv pip install -r requirements-dashboard.txt    # optional: Streamlit dashboard
    uv pip install -r requirements-dev.txt          # optional: run tests + ruff
    ```
    On Linux/WSL you can use the maintained `stable-retro` backend instead of
    `gym-retro` (`pip install stable-retro`); the runtime auto-detects it.
2.  Import the Sonic the Hedgehog ROM:
    ```bash
    python -m retro.import /path/to/your/roms
    ```
3.  Configure API keys in your environment (Powershell example):
    ```bash
    # Cloud Vision Provider (OpenRouter, Gemini, Anthropic, OpenAI, Kimi)
    $env:MACRO_API_KEY="your_api_key_here"
    $env:MACRO_BASE_URL="https://openrouter.ai/api/v1" # Or https://generativelanguage.googleapis.com/v1beta/openai/ for Gemini
    $env:MACRO_MODEL="anthropic/claude-3.5-sonnet"     # Or gemini-2.5-pro, gpt-4o, etc.

    # Local Code Provider (LM Studio, Ollama, llama.cpp, vLLM, Apple MLX)
    $env:MICRO_BASE_URL="http://localhost:11434/v1"    # Ollama default. Use 1234 for LM Studio.
    $env:MICRO_MODEL="qwen2.5-coder"                   # Your locally loaded model
    ```
4.  Run the evolutionary pipeline:
    ```bash
    python main.py
    ```
5.  Run the live dashboard in a separate terminal:
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

CI runs the same lint + test steps on Python 3.8 and 3.11 (`.github/workflows/evaluate_policy.yml`).

## Speedrun-First, Continuous Play-Through

Training starts at Green Hill Zone Act 1, but evaluation is a **continuous play-through**: when a policy clears an act, the game advances to the next one and the runner keeps going (detected via the emulator's `zone`/`act` values). Fitness rewards, in order of weight: **acts cleared**, then total distance, then speed (fewer frames), with rings/score as small tie-breakers and a completion bonus for reaching the current act's end zone. Per-act progress is reset on each transition so the x-coordinate dropping back to ~0 in the next act is not mistaken for getting stuck.

This rewards a policy that *generalizes* across levels rather than one overfit to Act 1's geometry. `state['zone']` and `state['act']` are exposed to the policy so it can branch per level (e.g. Labyrinth's water sections). The training frame budget (`max_frames`) spans several acts; weak candidates still terminate early via stuck-detection, so the larger budget only costs wall-clock on genuinely strong runs.

Mutation prompts receive compact frame traces with position, velocity, zone/act, rings, vision context, and the action taken. Fatal visual failures also use a small recent-frame montage when screenshots are available, giving the macro model more context than a single final frame.

The policy pool keeps a small amount of action-signature diversity instead of pruning strictly by score, so crossover has access to policies with different controller habits.

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
