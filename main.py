"""Evolutionary pipeline orchestration: generations, promotion, archiving.

The separable pieces live in ``core``: the emulator episode loop in
``core.evaluation``, the FunSearch parent pool in ``core.pool``, and the
deterministic frontier guards in ``core.frontier``. This module wires them to
the LLM mutator and the persistent history/population archives.
"""

import argparse
import concurrent.futures
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from core.diagnosis import (
    DiagnosisSession,
    FailureSnapshotRing,
    load_failure_window,
    window_key,
)
from core.escape_sweep import sweep_frontier_escapes
from core.evaluation import build_policy_load_failure, evaluate_policy
from core.frontier import (
    build_diagnosis_guard_candidate,
    build_frontier_guard_candidate,
    diagnosis_guard_marker,
    frontier_guard_marker,
    llm_guard_marker,
    recently_attempted_frontier_guard,
)
from core.fsio import atomic_write_text
from core.history import EvolutionHistory
from core.policy_loader import load_policy
from core.pool import load_pool_codes, update_pool
from core.population import PopulationArchive
from llm.mutator import MutatorClient


def evaluate_working_baseline(env, working_path, mutator, max_frames=5000, verbose=True, action_repeat=1, snapshot_sink=None):
    context = {
        "working_fitness": -1.0,
        "last_failure_reason": "Initial seed run",
        "last_screenshot": None,
        "last_trace": [],
        "components": {},
    }
    if env is None or not os.path.exists(working_path):
        return context

    try:
        policy = load_policy(working_path)
    except Exception as e:
        context["last_failure_reason"] = f"Failed to load working policy: {e}"
        return context

    if verbose:
        print("Evaluating current working policy baseline...")
    fitness, frames_alive, max_x, failure_reason, screenshot_path, trace, components = evaluate_policy(
        env,
        policy,
        mutator,
        max_frames=max_frames,
        verbose=verbose,
        action_repeat=action_repeat,
        snapshot_sink=snapshot_sink,
    )
    if verbose:
        print(f"Baseline Fitness: {fitness:.2f} (Max X: {max_x}, Frames: {frames_alive})")

    context.update(
        {
            "working_fitness": fitness,
            "last_failure_reason": failure_reason,
            "last_screenshot": screenshot_path,
            "last_trace": trace,
            "components": components,
        }
    )
    return context


def seed_population_baseline(archive, working_path, baseline_context):
    """Admit the evaluated working policy so early crossover has a strong parent."""
    if baseline_context.get("working_fitness", -1.0) < 0 or not os.path.exists(working_path):
        return False
    with open(working_path, "r", encoding="utf-8") as f:
        code = f.read()
    return record_candidate_evaluation(
        archive,
        code,
        baseline_context["working_fitness"],
        baseline_context.get("components", {}),
        baseline_context.get("last_failure_reason", "Working policy baseline"),
        baseline_context.get("last_trace", []),
        "Working policy baseline",
    )


def build_stagnation_escape_context(working_fitness, last_trace, last_screenshot):
    """Preserve the champion while asking mutations to explore a distinct strategy."""
    return {
        "working_fitness": working_fitness,
        "last_failure_reason": (
            "Stagnation plateau: preserve the current working policy, but try a distinct "
            "minimal strategy that is not already represented in recent candidates."
        ),
        "last_trace": last_trace,
        "last_screenshot": last_screenshot,
    }


def select_working_frontier_context(current_context, candidate_context, promoted):
    """Keep mutation feedback aligned with the code that will be mutated next."""
    return dict(candidate_context if promoted else current_context)


def diagnosable_failure(failure_reason):
    """Only real visual frontiers get agentic diagnosis.

    Timeouts are code bugs (local model territory) and the stagnation-escape
    pseudo-failure intentionally asks for a *different* strategy, so replaying
    the old frontier would anchor the model right back to it.
    """
    reason = str(failure_reason or "").lower()
    if "timeout" in reason:
        return False
    return "stuck" in reason or "fatal" in reason or "lost a life" in reason


DIAGNOSIS_REPORT_PATH = os.path.join("artifacts", "diagnosis", "latest_report.json")


def persist_diagnosis_report(result, failure_reason, report_path=DIAGNOSIS_REPORT_PATH):
    """Write the latest diagnosis where the dashboard can show it."""
    if not result or not result.get("report"):
        return
    try:
        atomic_write_text(
            report_path,
            json.dumps(
                {
                    "report": result.get("report", ""),
                    "evidence_screenshot": result.get("evidence_screenshot"),
                    "verified_experiments": result.get("verified_experiments", []),
                    "failure_reason": str(failure_reason or ""),
                    "created_at": int(time.time()),
                },
                indent=2,
            ),
        )
    except Exception as e:
        print(f"Failed to persist diagnosis report: {e}")


REDIAGNOSE_AFTER_STAGNANT_GENERATIONS = 3


def maybe_diagnose_frontier(
    mutator,
    frontier,
    cache,
    emulator_available=True,
    session_factory=None,
    report_path=DIAGNOSIS_REPORT_PATH,
    stagnation_counter=0,
    window_loader=None,
):
    """Run agentic diagnosis on the frontier; reuse while it is unchanged.

    Returns ``{"report", "evidence_screenshot", "verified_experiments"}`` or
    ``None``. The cache also remembers failed attempts so a broken diagnosis
    setup is not retried every generation — but a frontier that stays stuck
    for ``REDIAGNOSE_AFTER_STAGNANT_GENERATIONS`` more generations gets a
    fresh experiment budget (live-observed: one no-escape diagnosis froze the
    investigation for the rest of the run while mutations stayed flat).
    """
    if not emulator_available or os.environ.get("SONIC_AGENTIC_DIAGNOSIS", "1") == "0":
        return None
    window_dir = frontier.get("window")
    if not window_dir or not diagnosable_failure(frontier.get("failure_reason")):
        return None
    window = (window_loader or load_failure_window)(window_dir)
    if window is None:
        return None

    key = window_key(window)
    if cache.get("key") == key:
        stagnated_since = stagnation_counter - cache.get("stagnation_counter", 0)
        found_escape = bool((cache.get("result") or {}).get("verified_experiments"))
        if found_escape or stagnated_since < REDIAGNOSE_AFTER_STAGNANT_GENERATIONS:
            return cache.get("result")
        print(
            f"Frontier still stuck after {stagnated_since} more generations; "
            "re-running agentic diagnosis with a fresh experiment budget."
        )

    # Mechanical escape sweep first: dozens of canonical inputs replayed at the
    # frontier cost seconds of emulator compute and zero model calls. The vision
    # session (minutes of model time, ~6 experiments) is reserved for obstacles
    # the standard battery cannot beat.
    sweep_note = ""
    if os.environ.get("SONIC_ESCAPE_SWEEP", "1") != "0":
        experiments, summary = sweep_frontier_escapes(window, session_factory=session_factory)
        if experiments:
            result = {
                "report": summary,
                "evidence_screenshot": None,
                "verified_experiments": experiments,
            }
            persist_diagnosis_report(result, frontier.get("failure_reason"), report_path=report_path)
            cache["key"] = key
            cache["result"] = result
            cache["stagnation_counter"] = stagnation_counter
            return result
        sweep_note = (
            "\nNote: a mechanical sweep already replayed the standard escapes at this "
            "frontier (RIGHT,B holds, RIGHT,UP,B high jump, RIGHT,DOWN roll, run-up "
            "jumps with 30/60/120-frame runways) and NONE beat it. Spend your "
            "experiments on genuinely different approaches."
        )

    session = None
    try:
        build_session = session_factory or DiagnosisSession
        session = build_session(window)
        print("Running agentic failure diagnosis on the working frontier...")
        result = mutator.diagnose_failure(
            session, str(frontier.get("failure_reason") or "") + sweep_note, frontier.get("trace")
        )
        if result:
            print("Diagnosis complete.")
            persist_diagnosis_report(result, frontier.get("failure_reason"), report_path=report_path)
    except Exception as e:
        print(f"Diagnosis skipped: {e}")
        result = None
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    cache["key"] = key
    cache["result"] = result
    cache["stagnation_counter"] = stagnation_counter
    return result


def persist_frontier_window(ring, failure_reason, directory=None):
    """Persist the frontier run's savestate window; never fails the loop."""
    if ring is None:
        return None
    try:
        if directory is None:
            return ring.persist(failure_reason=failure_reason)
        return ring.persist(directory, failure_reason=failure_reason)
    except Exception as e:
        print(f"Failed to persist diagnosis window: {e}")
        return None


def preserve_frontier_screenshot(
    screenshot_path,
    destination="artifacts/failures/working_frontier.png",
):
    """Copy a frontier image away from shared screenshot paths."""
    if not screenshot_path or not os.path.exists(screenshot_path):
        return screenshot_path
    if os.path.abspath(screenshot_path) == os.path.abspath(destination):
        return destination
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(screenshot_path, destination)
    return destination


def clear_candidate_recording(record_dir, candidate_idx):
    candidate_bk2_path = os.path.join(record_dir, f"candidate_{candidate_idx}.bk2")
    if os.path.exists(candidate_bk2_path):
        os.remove(candidate_bk2_path)


def record_candidate_evaluation(
    archive,
    code,
    fitness,
    components,
    failure_reason,
    trace,
    reasoning,
):
    """Record a candidate without allowing archive failures to stop training."""
    try:
        archive.record_evaluation(
            code,
            fitness=fitness,
            components=components,
            failure_reason=failure_reason,
            trace=trace,
            reasoning=reasoning,
        )
        return True
    except Exception as e:
        print(f"Failed to record candidate in population archive: {e}")
        return False


def prepare_candidate_policy(candidate_path, code, reasoning, mutator, archive):
    """Load a candidate, making at most one local repair after validation failure."""
    atomic_write_text(candidate_path, code)

    try:
        policy = load_policy(candidate_path)
        return {
            "code": code,
            "reasoning": reasoning,
            "policy": policy,
            "load_error": None,
        }
    except Exception as original_error:
        original_error_text = str(original_error)
        record_candidate_evaluation(
            archive,
            code,
            0.0,
            {"load_error": original_error_text, "repair_stage": "original"},
            f"Policy failed to load before repair: {original_error_text}",
            [],
            reasoning,
        )

    try:
        repaired_code, repair_reasoning = mutator.repair_policy(code, original_error_text)
    except Exception as repair_error:
        return {
            "code": code,
            "reasoning": reasoning,
            "policy": None,
            "load_error": repair_error,
        }

    repaired_code = str(repaired_code or "")
    repair_reasoning = str(repair_reasoning or "")
    combined_reasoning = f"{reasoning}\nValidator repair: {repair_reasoning}".strip()
    atomic_write_text(candidate_path, repaired_code)
    try:
        repaired_policy = load_policy(candidate_path)
        repaired_error = None
    except Exception as e:
        repaired_policy = None
        repaired_error = e

    return {
        "code": repaired_code,
        "reasoning": combined_reasoning,
        "policy": repaired_policy,
        "load_error": repaired_error,
    }


def generate_candidates(
    mutator,
    working_code,
    last_failure_reason,
    last_screenshot,
    recent_history,
    last_trace,
    n_candidates,
    pool_codes,
    crossover_probability=0.30,
    parent_selector=None,
    diagnosis_report=None,
    verified_experiments=None,
    frontier=None,
):
    """Request ``n_candidates`` new policies from the mutator.

    Most are mutations of the working policy; when the pool is large enough a
    fraction become FunSearch crossovers of two pooled parents. At most one
    slot goes to a deterministic candidate: a guard compiled from a VERIFIED
    diagnosis experiment when one exists, else the stationary-frontier guard.
    Returns a list of ``(code, reasoning)`` tuples. A request that errors
    falls back to the working policy so every slot is filled.
    """
    candidates_code = [None] * n_candidates
    first_llm_slot = 0

    deterministic = None
    reasoning_label = None
    # A measured escape beats the heuristic recovery guard: try the best
    # verified experiment (furthest measured x) first.
    for experiment in sorted(
        verified_experiments or [], key=lambda e: e.get("max_x", 0), reverse=True
    ):
        guard = build_diagnosis_guard_candidate(working_code, experiment)
        if guard is not None and not recently_attempted_frontier_guard(
            diagnosis_guard_marker(guard), recent_history
        ):
            deterministic = guard
            reasoning_label = "Diagnosed guard (verified input)"
            break

    if deterministic is None:
        frontier_guard = build_frontier_guard_candidate(working_code, last_trace)
        if frontier_guard is not None and not recently_attempted_frontier_guard(
            frontier_guard_marker(frontier_guard),
            recent_history,
        ):
            deterministic = frontier_guard
            reasoning_label = "Deterministic frontier guard"

    if n_candidates > 0 and deterministic is not None:
        print(f"Adding deterministic candidate: {reasoning_label}.")
        candidates_code[0] = (deterministic, reasoning_label)
        first_llm_slot = 1

    # max_workers=1 serialises requests so we never hammer a local LLM that
    # cannot service concurrent completions.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        for c in range(first_llm_slot, n_candidates):
            parent_pair = None
            can_attempt_crossover = parent_selector is not None or len(pool_codes) >= 2
            if can_attempt_crossover and random.random() < crossover_probability:
                print(f"Requesting Crossover {c+1}/{n_candidates} (FunSearch)...")
                if parent_selector is not None:
                    try:
                        parent_pair = parent_selector()
                    except Exception as e:
                        print(f"Population parent selection failed: {e}")
                if parent_pair is None and len(pool_codes) >= 2:
                    parent_pair = random.sample(pool_codes, 2)

            if parent_pair is not None:
                future = executor.submit(
                    mutator.crossover_policies,
                    parent_pair[0],
                    parent_pair[1],
                    recent_history,
                    temperature=0.7,
                )
            else:
                temperature = 0.7 if c == 0 else 0.9
                print(f"Requesting Mutation {c+1}/{n_candidates} (Temp: {temperature})...")
                future = executor.submit(
                    mutator.mutate_policy,
                    working_code,
                    last_failure_reason,
                    last_screenshot,
                    recent_history,
                    temperature,
                    last_trace,
                    diagnosis_report,
                    frontier=frontier,
                )
            futures[future] = c

        for future in concurrent.futures.as_completed(futures):
            c = futures[future]
            try:
                new_code, reasoning = future.result()
                candidates_code[c] = (new_code, reasoning)
                print(f"Mutation {c+1} received.")
            except Exception as e:
                print(f"Mutation {c+1} failed: {e}")
                candidates_code[c] = (working_code, f"Failed: {e}")
    return candidates_code


def resolve_end_generation(start_gen, max_generations, generations=None):
    if generations is None:
        return max_generations
    return start_gen + generations - 1


def derive_resume_state(history_entries):
    if not history_entries:
        return {
            "all_time_champion_fitness": -1.0,
            "start_generation": 1,
            "stagnation_counter": 0,
            "champion_max_frames": None,
        }
    def number(entry, key, default, converter):
        try:
            value = converter(entry.get(key, default))
            if isinstance(value, float) and not math.isfinite(value):
                return default
            return value
        except (OverflowError, TypeError, ValueError):
            return default

    champion_entry = max(
        history_entries, key=lambda entry: number(entry, "fitness", -1.0, float)
    )

    return {
        "all_time_champion_fitness": number(champion_entry, "fitness", -1.0, float),
        "start_generation": number(history_entries[-1], "generation", 0, int) + 1,
        "stagnation_counter": number(
            history_entries[-1], "stagnation_counter", 0, int
        ),
        "champion_max_frames": number(champion_entry, "max_frames", None, int),
    }


def resolve_working_fitness_floor(
    baseline_fitness,
    champion_fitness,
    champion_max_frames,
    current_max_frames,
    verbose=True,
):
    """Pick the promotion bar for a (possibly resumed) run.

    Fitness scales with the frame budget, so flooring the bar at a historical
    champion measured under a *different* ``max_frames`` makes the bar
    unreachable (smaller budget) or trivial (larger budget) and the loop
    stagnates forever. Legacy histories without a recorded budget keep the old
    flooring behaviour.
    """
    if champion_max_frames is not None and int(champion_max_frames) != int(current_max_frames):
        if verbose and champion_fitness > baseline_fitness:
            print(
                f"Historical champion fitness {champion_fitness:.2f} was measured at "
                f"max_frames={champion_max_frames}, but this run uses max_frames={current_max_frames}. "
                f"Using the re-evaluated baseline {baseline_fitness:.2f} as the promotion bar."
            )
        return baseline_fitness
    return max(baseline_fitness, champion_fitness)


def promotion_confirmed(original_fitness, retest_fitness, bar):
    """A would-be promotion is only confirmed when the candidate beats the bar on
    BOTH the original evaluation and a retest.

    Evaluation has run-to-run variance (observed live: the same policy scoring
    54397 or 50541 depending on get_action timing under load), so a single lucky
    eval must not become the champion.
    """
    return original_fitness > bar and retest_fitness > bar


def retest_candidate_fitness(env, mutator, candidate_path, max_frames, action_repeat):
    """Re-evaluate a candidate once to confirm a fitness gain reproduces.

    Returns the retest fitness, or None if it could not be run (the caller then
    keeps the original single-eval decision).
    """
    try:
        policy = load_policy(candidate_path)
    except Exception as e:
        print(f"Retest load failed: {e}")
        return None
    try:
        return evaluate_policy(
            env,
            policy,
            mutator,
            max_frames,
            action_repeat=action_repeat,
            snapshot_sink=FailureSnapshotRing(),
        )[0]
    except Exception as e:
        print(f"Retest eval failed: {e}")
        return None


def candidate_is_promotable(env, policy, components):
    return env is not None and policy is not None and not components.get("runtime_error")


def candidate_beats_current_best(fitness, promotable, best_fitness, best_promotable):
    return fitness > best_fitness or (
        fitness == best_fitness and promotable and not best_promotable
    )


def choose_generation_archive_path(best_candidate_path, working_path):
    return best_candidate_path or working_path


def render_candidate_video(bk2_path, mp4_path):
    if not os.path.exists(bk2_path):
        return False
    if os.path.exists(mp4_path):
        os.remove(mp4_path)
    try:
        result = subprocess.run(
            [sys.executable, "render_video.py", bk2_path, mp4_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and os.path.exists(mp4_path)


def run_evaluation_loop(
    max_generations=500,
    max_frames=12000,
    n_candidates=2,
    stagnation_limit=5,
    action_repeat=1,
    generations=None,
):
    # max_frames now spans several acts (~3-4k frames each) so a strong policy
    # can play through. Weak candidates still terminate early via stuck-detection,
    # so the larger budget only costs wall-clock for genuinely good runs.
    history = EvolutionHistory()
    mutator = MutatorClient()
    population = PopulationArchive()

    try:
        from emulator.sonic_env import SonicEnvWrapper
        os.makedirs("artifacts/videos/tmp", exist_ok=True)
        env = SonicEnvWrapper(record_path="artifacts/videos/tmp")
    except ImportError:
        print("Warning: retro emulator backend not available. Running in dry-run mode.")
        env = None

    champion_path = os.path.join("policies", "champion_policy.py")
    working_path = os.path.join("policies", "working_policy.py")

    if not os.path.exists(champion_path):
        print("Creating initial seed policy.")
        mutator.write_seed_policy(champion_path)

    if not os.path.exists(working_path):
        shutil.copy(champion_path, working_path)

    resume_state = derive_resume_state(history.history)
    all_time_champion_fitness = resume_state["all_time_champion_fitness"]
    start_gen = resume_state["start_generation"]
    stagnation_counter = resume_state["stagnation_counter"]
    baseline_ring = FailureSnapshotRing()
    baseline_context = evaluate_working_baseline(
        env, working_path, mutator, max_frames, action_repeat=action_repeat, snapshot_sink=baseline_ring
    )
    working_fitness = resolve_working_fitness_floor(
        baseline_context["working_fitness"],
        all_time_champion_fitness,
        resume_state["champion_max_frames"],
        max_frames,
    )
    working_frontier = {
        "failure_reason": baseline_context["last_failure_reason"],
        "trace": baseline_context["last_trace"],
        "screenshot": preserve_frontier_screenshot(baseline_context["last_screenshot"]),
        "window": persist_frontier_window(baseline_ring, baseline_context["last_failure_reason"]),
        "frontier": baseline_context.get("components", {}).get("frontier"),
    }
    seed_population_baseline(population, working_path, baseline_context)

    diagnosis_cache = {}
    end_gen = resolve_end_generation(start_gen, max_generations, generations)
    for gen in range(start_gen, end_gen + 1):
        print(f"\n--- Generation {gen} ---")

        mutation_frontier = working_frontier
        if stagnation_counter >= stagnation_limit:
            print(
                f"Stagnation detected ({stagnation_counter} gens without improvement). "
                "Keeping the working champion and requesting a distinct strategy."
            )
            escape = build_stagnation_escape_context(
                working_fitness,
                working_frontier["trace"],
                working_frontier["screenshot"],
            )
            working_fitness = escape["working_fitness"]
            mutation_frontier = {
                "failure_reason": escape["last_failure_reason"],
                "trace": escape["last_trace"],
                "screenshot": escape["last_screenshot"],
            }
            stagnation_counter = 0

        with open(working_path, "r", encoding="utf-8") as f:
            working_code = f.read()

        best_candidate_code = None
        best_candidate_fitness = -1.0
        best_candidate_reason = ""
        best_candidate_screenshot = None
        best_candidate_reasoning = ""
        best_candidate_trace = []
        best_candidate_components = {}
        best_candidate_idx = -1
        best_candidate_path = None
        best_candidate_promotable = False
        best_candidate_ring = None
        attempted_frontier_markers = set()

        # Diagnose the frontier once per generation (cached while unchanged):
        # the vision model replays the failure and experiments with inputs.
        diagnosis = maybe_diagnose_frontier(
            mutator,
            mutation_frontier,
            diagnosis_cache,
            emulator_available=env is not None,
            stagnation_counter=stagnation_counter,
        )
        diagnosis_report = diagnosis.get("report") if diagnosis else None
        evidence_screenshot = diagnosis.get("evidence_screenshot") if diagnosis else None
        verified_experiments = diagnosis.get("verified_experiments") if diagnosis else None

        # Generate candidates
        recent_history = history.get_recent_history(3)
        pool_codes = load_pool_codes()
        candidates_code = generate_candidates(
            mutator,
            working_code,
            mutation_frontier["failure_reason"],
            evidence_screenshot or mutation_frontier["screenshot"],
            recent_history,
            mutation_frontier["trace"],
            n_candidates,
            pool_codes,
            parent_selector=population.select_parent_codes,
            diagnosis_report=diagnosis_report,
            verified_experiments=verified_experiments,
            frontier=mutation_frontier.get("frontier"),
        )

        # Evaluate candidates
        for c, (new_code, reasoning) in enumerate(candidates_code):
            print(f"\nEvaluating Candidate {c+1}...")
            candidate_path = os.path.join("policies", f"candidate_{c}.py")
            clear_candidate_recording("artifacts/videos/tmp", c)
            prepared = prepare_candidate_policy(
                candidate_path,
                new_code,
                reasoning,
                mutator,
                population,
            )
            new_code = prepared["code"]
            reasoning = prepared["reasoning"]
            policy = prepared["policy"]
            load_error = prepared["load_error"]
            if load_error is not None:
                print(f"Failed to load policy after one repair attempt: {load_error}")

            candidate_ring = FailureSnapshotRing()
            if env is None:
                fitness = 0.0
                failure_reason = "Mock failure."
                screenshot_path = "artifacts/failures/mock_screenshot.png"
                trace = []
                components = {}
            elif policy is None:
                fitness, frames_alive, max_x, failure_reason, screenshot_path, trace, components = build_policy_load_failure(load_error)
            else:
                fitness, frames_alive, max_x, failure_reason, screenshot_path, trace, components = evaluate_policy(
                    env,
                    policy,
                    mutator,
                    max_frames,
                    action_repeat=action_repeat,
                    snapshot_sink=candidate_ring,
                )
                print(f"Candidate {c+1} Fitness: {fitness:.2f} (Max X: {max_x}, Frames: {frames_alive})")

                # Flush the bk2 video buffer to disk (stable-retro only flushes on reset)
                try:
                    env.reset()
                except Exception:
                    pass

                bk2_files = glob.glob("artifacts/videos/tmp/*.bk2")
                if bk2_files:
                    latest_bk2 = max(bk2_files, key=os.path.getmtime)
                    candidate_bk2_path = f"artifacts/videos/tmp/candidate_{c}.bk2"
                    if os.path.exists(candidate_bk2_path) and candidate_bk2_path != latest_bk2:
                        os.remove(candidate_bk2_path)
                    if candidate_bk2_path != latest_bk2:
                        os.rename(latest_bk2, candidate_bk2_path)

            marker = (
                frontier_guard_marker(new_code)
                or diagnosis_guard_marker(new_code)
                or llm_guard_marker(new_code)
            )
            if reasoning in (
                "Deterministic frontier guard",
                "Diagnosed guard (verified input)",
                "LLM structured guard",
            ) and marker is not None:
                components = dict(components)
                components["frontier_guard_marker"] = marker
                attempted_frontier_markers.add(marker)

            record_candidate_evaluation(
                population,
                new_code,
                fitness,
                components,
                failure_reason,
                trace,
                reasoning,
            )

            promotable = candidate_is_promotable(env, policy, components)
            if candidate_beats_current_best(
                fitness,
                promotable,
                best_candidate_fitness,
                best_candidate_promotable,
            ):
                best_candidate_fitness = fitness
                best_candidate_code = new_code
                best_candidate_reason = failure_reason
                best_candidate_screenshot = preserve_frontier_screenshot(
                    screenshot_path,
                    f"artifacts/failures/generation_{gen}_best.png",
                )
                best_candidate_reasoning = reasoning
                best_candidate_trace = trace
                best_candidate_components = components
                best_candidate_idx = c
                best_candidate_path = candidate_path
                best_candidate_promotable = promotable
                best_candidate_ring = candidate_ring

        if attempted_frontier_markers:
            best_candidate_components = dict(best_candidate_components)
            best_candidate_components["frontier_guard_markers"] = sorted(
                attempted_frontier_markers
            )

        # Promote or Stagnate
        best_bk2 = f"artifacts/videos/tmp/candidate_{best_candidate_idx}.bk2"
        latest_mp4 = "artifacts/videos/latest.mp4"

        latest_video_rendered = False
        if os.path.exists(best_bk2):
            print("Rendering video for generation's best candidate...")
            try:
                latest_video_rendered = render_candidate_video(best_bk2, latest_mp4)
                if not latest_video_rendered:
                    print("Failed to render video: renderer exited without a video.")
            except Exception as e:
                print(f"Failed to render video: {e}")

        # Retest a would-be winner before promoting it: eval variance means a
        # single lucky run must not become the champion. Require the gain to
        # reproduce and record the conservative (min) fitness.
        if (
            env is not None
            and best_candidate_promotable
            and best_candidate_fitness > working_fitness
            and best_candidate_path
            and os.path.exists(best_candidate_path)
        ):
            retest_fitness = retest_candidate_fitness(
                env, mutator, best_candidate_path, max_frames, action_repeat
            )
            if retest_fitness is not None:
                confirmed = promotion_confirmed(
                    best_candidate_fitness, retest_fitness, working_fitness
                )
                print(
                    f"Retest of best candidate: {best_candidate_fitness:.2f} -> "
                    f"{retest_fitness:.2f} (bar {working_fitness:.2f}); "
                    f"{'confirmed' if confirmed else 'NOT reproduced -- variance, not promoting'}"
                )
                best_candidate_fitness = min(best_candidate_fitness, retest_fitness)
                if not confirmed:
                    best_candidate_promotable = False

        promoted = best_candidate_promotable and best_candidate_fitness > working_fitness
        # Only a promoted run may replace the persisted diagnosis window: a
        # losing candidate's failure is not the frontier the next mutations
        # will be asked to fix.
        candidate_window = (
            persist_frontier_window(best_candidate_ring, best_candidate_reason) if promoted else None
        )
        working_frontier = select_working_frontier_context(
            working_frontier,
            {
                "failure_reason": best_candidate_reason,
                "trace": best_candidate_trace,
                "screenshot": preserve_frontier_screenshot(best_candidate_screenshot)
                if promoted
                else best_candidate_screenshot,
                "window": candidate_window,
                "frontier": best_candidate_components.get("frontier"),
            },
            promoted=promoted,
        )

        if promoted:
            print(f"Working policy improved from {working_fitness:.2f} -> {best_candidate_fitness:.2f}")
            working_fitness = best_candidate_fitness
            stagnation_counter = 0

            # Update working file
            atomic_write_text(working_path, best_candidate_code)

            # Update the FunSearch genetic population pool
            update_pool(best_candidate_code, best_candidate_fitness)

            if best_candidate_fitness > all_time_champion_fitness:
                print(f"NEW ALL-TIME CHAMPION! {all_time_champion_fitness:.2f} -> {best_candidate_fitness:.2f}")
                all_time_champion_fitness = best_candidate_fitness

                # Extract and save skills (Voyager Skill Library feature)
                print("Extracting new skills from this champion policy...")
                try:
                    mutator.extract_and_save_skills(best_candidate_code)
                except Exception as e:
                    print(f"Failed to extract skills: {e}")

                # Convert video for new champion
                champion_mp4 = "artifacts/videos/champion.mp4"
                if latest_video_rendered:
                    try:
                        if os.path.exists(champion_mp4):
                            os.remove(champion_mp4)
                        shutil.copy(latest_mp4, champion_mp4)
                    except Exception as e:
                        print(f"Failed to copy champion video: {e}")

                # Update all-time champion file
                atomic_write_text(champion_path, best_candidate_code)

            # Archive the winning candidate
            history.log_generation(gen, choose_generation_archive_path(best_candidate_path, working_path), best_candidate_fitness, best_candidate_reason, best_candidate_screenshot, best_candidate_reasoning, stagnation_counter, best_candidate_components, max_frames=max_frames)
        else:
            print(f"No candidate beat the working policy ({working_fitness:.2f}). Stagnation counter: {stagnation_counter + 1}")
            stagnation_counter += 1

            if "fatal" in best_candidate_reason.lower() or "stuck" in best_candidate_reason.lower() or "timeout" in best_candidate_reason.lower():
                print("Extracting a lesson learned from this failure...")
                try:
                    mutator.extract_lesson(best_candidate_reason, best_candidate_trace)
                except Exception as e:
                    print(f"Failed to extract lesson: {e}")

            # We log the generation even if it failed so the dashboard updates
            history.log_generation(gen, choose_generation_archive_path(best_candidate_path, working_path), best_candidate_fitness, best_candidate_reason, best_candidate_screenshot, best_candidate_reasoning, stagnation_counter, best_candidate_components, max_frames=max_frames)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Sonic policy evolution loop.")
    parser.add_argument("--generations", type=int, help="Number of additional generations to run.")
    parser.add_argument("--frames", type=int, default=12000, help="Maximum emulator frames per candidate.")
    parser.add_argument("--candidates", type=int, default=2, help="Candidates evaluated per generation.")
    args = parser.parse_args(argv)

    run_evaluation_loop(
        generations=args.generations,
        max_frames=args.frames,
        n_candidates=max(1, args.candidates),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
