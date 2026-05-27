# Speedrun-First Evaluation Design

## Goal

Optimize Green Hill Zone Act 1 speedrun performance as the primary training target, while using broader level coverage as a benchmark suite to detect brittle policies and regressions.

## Current Findings

The current champion reaches about `x=9767` on `GreenHillZone.Act1` in roughly `3130` frames, then stalls. Cross-level checks show poor generalization: the champion performs much worse on Green Hill Acts 2 and 3 than some non-champion policies. The evaluator also overwrites "stuck" failures with "lost a life", which sends bad feedback to the mutator and routes stagnation failures as visual/fatal failures.

## Target Behavior

The evaluator should preserve accurate termination reasons, report enough metrics to distinguish distance from speed, and make Act 1 speed the strongest optimization pressure. Rings and score should remain small tie-breakers rather than dominating policy selection.

The project should include a reusable benchmark command that evaluates one or more policy files against multiple installed Sonic states. These benchmark results should be informational until Act 1 completion and speed become reliable.

Memory used by prompts should be compact and structured. `memory/semantic_bank.json` should be deduplicated and validated before prompt injection. The legacy free-text lessons file should not be used for mutation prompts.

## Implementation Boundaries

This pass should focus on the evaluation and mutation pipeline, not hand-tuning a final Sonic route. It may add tests, benchmark tooling, CLI options, docs, and memory hygiene. Generated policies and ignored artifacts remain runtime outputs.

## Testing

Automated tests should cover failure reason preservation, speedrun-oriented scoring, semantic memory filtering, and benchmark result formatting. Emulator smoke tests should run with `venv38` where `gym-retro` and the imported Sonic states are available.
