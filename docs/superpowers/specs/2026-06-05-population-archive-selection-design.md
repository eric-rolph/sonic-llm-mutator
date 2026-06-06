# Population Archive and Exploration-Aware Selection Design

## Goal

Retain every evaluated Sonic policy as a reusable search artifact and select
crossover parents using measured quality, behavioral diversity, and an
exploration bonus instead of uniform random sampling.

## Scope

This slice adds:

- A cheap policy preflight validator before generated modules are imported.
- A persistent candidate archive under `artifacts/population/`.
- Structured records containing policy code, fitness components, failure
  context, trace, behavior descriptor, obstacle key, and parent-selection
  visits.
- Deduplication by policy-code hash while retaining the best evaluation.
- P-UCB-style weighted parent selection from the strongest and most diverse
  archived candidates.
- Evaluation-loop integration that records every candidate, including failed
  and non-promoted candidates.
- Admission of the evaluated working baseline before the first generation.
- Non-destructive stagnation handling that preserves the working champion.
- Preservation of requested mutation temperature when visual routing falls
  back to the local model.

Savestate replay, multi-turn repair, LLM raters, and sandboxing are explicitly
out of scope for this slice.

## Architecture

`core/population.py` owns persistence and selection. It exposes small pure
helpers for deriving behavior descriptors and obstacle keys, plus a
`PopulationArchive` class for recording evaluations and sampling parents.

`core/policy_validator.py` parses candidate source before import, requires the
policy contract, and rejects imports or obvious dangerous builtins outside the
small trusted policy surface. It is a preflight guard rather than a complete
sandbox.

The existing pool remains available as a compatibility fallback. The
evaluation loop records each candidate immediately after evaluation. Candidate
generation asks the archive for two parents; if fewer than two archived
candidates exist, it falls back to the legacy pool or ordinary mutation.

## Candidate Records

Each unique policy hash maps to one JSON record and one Python source artifact.
The record stores:

- `policy_id`, `code_path`, and `code_hash`
- Best observed `fitness` and `components`
- `failure_reason`, `trace`, and `reasoning`
- `behavior_descriptor` derived from returned actions
- `obstacle_key` derived from the last trace point and failure category
- `evaluations`, `selection_visits`, and timestamps

Re-evaluating identical code increments `evaluations`. Better evaluations
replace the best-result fields; weaker evaluations do not erase stronger data.

## Selection

Selection first filters to a bounded elite set while preserving one leader per
behavior descriptor and obstacle key. It then assigns each candidate:

`normalized_fitness + exploration_constant * sqrt(total_visits + 1) / (visits + 1)`

Two distinct parents are sampled without replacement using these positive
scores as weights. Selected records increment their visit counters.

## Error Handling

Malformed or missing archive metadata is treated as an empty archive. A failed
archive write must not abort training. Generated records use atomic metadata
replacement so an interrupted write does not corrupt the prior database.

## Testing

Focused unit tests cover:

- Policy syntax, contract, import, and dangerous-builtin validation.
- Stable obstacle keys from failure traces.
- Recording and updating duplicate policies.
- Preservation of behavior and obstacle specialists.
- Exploration bonuses favoring under-visited candidates.
- Evaluation-loop candidate generation using archived parents.

The existing full unit suite and Ruff lint remain the final verification gate.
