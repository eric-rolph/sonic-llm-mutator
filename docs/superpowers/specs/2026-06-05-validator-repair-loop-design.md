# Bounded Validator Repair Loop Design

## Goal

Recover useful candidates from syntax, policy-contract, and preflight-validation
failures before spending emulator time, using the validator's exact error as
feedback to one local-model repair attempt.

## Scope

When generated candidate code fails `load_policy`:

1. Preserve the invalid source and exact validation error in the population
   archive.
2. Ask the local micro model to repair only the validation failure.
3. Validate and load the repaired source once.
4. Evaluate it normally if valid, or record it as a load failure if still
   invalid.

The loop performs at most one repair attempt. It never invokes the vision model,
does not recursively repair a repair, and does not repair emulator gameplay
failures.

## Components

`MutatorClient.repair_policy(candidate_code, validation_error)` builds a compact
local-only prompt containing the invalid code, exact validator feedback, and
the trusted policy contract. It returns cleaned Python source and reasoning.

`prepare_candidate_policy(candidate_path, code, reasoning, mutator, archive)`
owns candidate source preparation:

- Write and load the original candidate.
- On failure, archive the invalid original with zero fitness.
- Request one repair and overwrite the candidate path with repaired code.
- Return the final code, combined reasoning, loaded policy, and final error.

The main evaluation loop consumes this result exactly as it consumes generated
candidates today.

## Failure Handling

If repair inference raises, returns invalid source, or fails loading, the
candidate receives the ordinary zero-fitness load failure. Archive errors remain
non-fatal. The original invalid source is always preserved before repair.

## Testing

Tests prove:

- Repairs always use the local micro model and include exact validator feedback.
- Valid candidates do not trigger repair.
- Invalid candidates trigger exactly one repair.
- Original invalid source is archived before repair.
- Failed repairs do not recurse.
- Repaired code, not invalid original code, proceeds to evaluation.
