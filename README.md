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

## Resilience Features

To ensure the pipeline can run continuously without manual intervention:
1.  **Syntax Error Sandboxing**: The dynamic Python code loader is wrapped in error handling. If the LLM generates invalid code (e.g., a SyntaxError), the pipeline catches it, assigns the candidate a fitness score of `0.0`, and continues running seamlessly.
2.  **Stateful Emulator Recording**: We manually enforce video buffer flushing by calling an extra `env.reset()` before teardown, ensuring that the emulator correctly writes the `.bk2` video files to disk even if the episode is manually terminated early.
3.  **Aggressive Cache Breaking**: Mutator prompts are seeded with a randomized cryptographically secure string to prevent local LLMs from entering endless prompt-caching loops.

## Setup

1.  Install dependencies:
    ```bash
    uv venv --python 3.8 venv
    .\venv\Scripts\Activate.ps1
    uv pip install -r requirements.txt
    ```
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

## Speedrun-First Evaluation

The current training target is Green Hill Zone Act 1 speedrun performance. Fitness now favors distance and fewer frames, with rings and game score treated as small tie-breakers. A completion bonus is awarded once a policy reaches the Act 1 end-zone threshold.

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

For a quick emulator smoke test:

```bash
.\venv38\Scripts\python.exe benchmark_policies.py --backend legacy --max-frames 900 --states GreenHillZone.Act1 --policies policies/champion_policy.py
```
