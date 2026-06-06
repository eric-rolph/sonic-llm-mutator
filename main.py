import argparse
import ast
import concurrent.futures
import glob
import hashlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))

from core.actions import action_string_to_array
from core.evaluator import calculate_fitness
from core.history import EvolutionHistory
from core.policy_loader import load_policy
from core.policy_runner import PolicyRunner, PolicyTimeout
from core.policy_validator import validate_policy_source
from core.population import PopulationArchive
from core.trace_context import build_screenshot_montage, build_trace_entry
from llm.mutator import MutatorClient

# Cadence for in-loop sampling, measured in emulator frames. Counter-based
# checks below ensure these still fire when action_repeat does not evenly divide
# the interval (a plain `frames % N == 0` test would silently skip).
VISION_INTERVAL = 300   # proactive cloud-vision tagging (~5s at 60fps)
TRACE_INTERVAL = 30     # rich trace entries (~0.5s at 60fps)
CONTEXT_SCREENSHOT_SLOTS = 3  # bound on-disk context shots to a small ring
# After an act transition the emulator keeps reporting the previous act's x for
# a few frames before it resets to the new act's start. Re-baseline max_x to the
# live x (and suspend stuck-detection) for this many frames so the stale value
# is not treated as current-act progress.
LEVEL_SETTLE_FRAMES = 150
# Terminate a run once Sonic makes no forward progress for this many frames
# (~8s at 60fps). Counted in frames (not loop iterations) so the threshold is
# unaffected by action_repeat.
STUCK_FRAME_LIMIT = 500


def atomic_write_text(filepath, text):
    """Write UTF-8 text without exposing a truncated destination on failure."""
    directory = os.path.dirname(filepath) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".policy-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def evaluate_working_baseline(env, working_path, mutator, max_frames=5000, verbose=True, action_repeat=1):
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


def build_policy_load_failure(error):
    reason = f"Policy failed to load: {error}"
    return 0.0, 0, 0, reason, None, [], {"load_error": str(error)}


def capture_screenshot(env, filepath=None):
    try:
        if filepath is not None:
            return env.get_screenshot(filepath)
        return env.get_screenshot()
    except TypeError:
        return env.get_screenshot()


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


def evaluate_policy(env, policy, mutator, max_frames=5000, verbose=True, action_repeat=1):
    obs = env.reset()
    frames_alive = 0
    max_x = 0
    done = False
    stuck_counter = 0
    failure_reason = None
    state = {}
    completion_x = None
    action_repeat = max(1, int(action_repeat or 1))

    # Continuous play-through bookkeeping. max_x is the furthest point in the
    # *current* act; cumulative_distance banks the distance of acts already
    # cleared, and levels_cleared counts how many acts were beaten this episode.
    levels_cleared = 0
    cumulative_distance = 0
    prev_zone_act = None
    settle_frames_left = 0
    settle_origin_x = 0
    runtime_error = None

    current_vision_context = "UNKNOWN"

    trace = []
    context_screenshots = []
    vision_poll_count = 0
    # Initialise so both samplers fire on the first iteration. Using elapsed
    # frames rather than `frames_alive % N == 0` means the cadence is preserved
    # even when action_repeat does not evenly divide the interval.
    last_vision_frame = -VISION_INTERVAL
    last_trace_frame = -TRACE_INTERVAL

    # Proactive vision polling tags the upcoming hazard every VISION_INTERVAL
    # frames. It is a slow model call, so it runs single-flight on a background
    # thread. Disable it (SONIC_PROACTIVE_VISION=0) when the vision model shares
    # an endpoint with the code model: the polls would compete for the same local
    # model and rarely finish before a fast headless eval ends. The death-frame
    # analysis on a fatal failure is unaffected either way.
    proactive_vision = os.environ.get("SONIC_PROACTIVE_VISION", "1") != "0"

    try:
        runner = PolicyRunner(policy)
    except Exception as e:
        reason = f"Policy runner failed to start: {type(e).__name__}: {e}"
        return (
            0.0,
            0,
            0,
            reason,
            capture_screenshot(env),
            [],
            {"runtime_error": reason},
        )
    vision_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
        if proactive_vision else None
    )
    vision_future = None

    def valid_zone_act(zone_act):
        return zone_act[0] is not None and zone_act[1] is not None

    def is_forward_transition(previous, current):
        if not valid_zone_act(previous) or not valid_zone_act(current):
            return False
        try:
            previous_zone, previous_act = map(int, previous)
            current_zone, current_act = map(int, current)
        except (TypeError, ValueError):
            return False
        return current_zone > previous_zone or (
            current_zone == previous_zone and current_act > previous_act
        )

    def update_authoritative_progress(authoritative_state, frames_advanced):
        nonlocal completion_x, cumulative_distance, levels_cleared
        nonlocal max_x, prev_zone_act, settle_frames_left, settle_origin_x
        nonlocal stuck_counter

        zone_act = (authoritative_state.get("zone"), authoritative_state.get("act"))
        transitioned = False
        if prev_zone_act is None or (not valid_zone_act(prev_zone_act) and valid_zone_act(zone_act)):
            prev_zone_act = zone_act
        elif is_forward_transition(prev_zone_act, zone_act):
            settle_origin_x = max_x
            cumulative_distance += max_x
            levels_cleared += 1
            max_x = 0
            stuck_counter = 0
            settle_frames_left = LEVEL_SETTLE_FRAMES
            prev_zone_act = zone_act
            completion_x = None
            transitioned = True
            if verbose:
                print(
                    f"Level cleared! Entering zone {zone_act[0]} act {zone_act[1]} "
                    f"(total cleared: {levels_cleared})"
                )

        state_completion_x = authoritative_state.get("screen_x_end") or authoritative_state.get("completion_x")
        if state_completion_x:
            completion_x = state_completion_x

        current_x = authoritative_state.get("x_pos", 0)
        if transitioned:
            # The act flag can change before x resets. Do not count that stale
            # high x in both the cleared act and the new act.
            if current_x < settle_origin_x:
                max_x = current_x
            return zone_act

        if settle_frames_left > 0:
            settle_frames_left = max(0, settle_frames_left - frames_advanced)
            if current_x < settle_origin_x and current_x > max_x:
                max_x = current_x
            stuck_counter = 0
        elif current_x > max_x:
            max_x = current_x
            stuck_counter = 0
        elif authoritative_state.get("level_end_bonus"):
            stuck_counter = 0
        else:
            stuck_counter += frames_advanced
        return zone_act

    try:
        state = env.get_state()
        zone_act = update_authoritative_progress(state, 0)
        while not done and frames_alive < max_frames:
            if proactive_vision:
                # Pick up the most recent finished vision result without blocking.
                if vision_future is not None and vision_future.done():
                    try:
                        current_vision_context = vision_future.result()
                    except Exception:
                        current_vision_context = "UNKNOWN"
                    vision_future = None

                # Kick off the next proactive vision poll in the background.
                if frames_alive - last_vision_frame >= VISION_INTERVAL and vision_future is None:
                    last_vision_frame = frames_alive
                    slot = vision_poll_count % CONTEXT_SCREENSHOT_SLOTS
                    vision_poll_count += 1
                    context_path = f"artifacts/failures/context_slot{slot}.png"
                    shot = capture_screenshot(env, context_path)
                    if shot:
                        context_screenshots.append(shot)
                        context_screenshots = context_screenshots[-CONTEXT_SCREENSHOT_SLOTS:]
                        vision_future = vision_executor.submit(mutator.analyze_environment, shot)

            policy_state = dict(state)
            policy_state["vision_context"] = current_vision_context

            try:
                action_string = runner.get_action(policy_state, timeout=0.5)
            except PolicyTimeout:
                if verbose:
                    print("Policy timed out (infinite loop?)")
                done = True
                runtime_error = "PolicyTimeout: get_action exceeded 0.5 seconds"
                # "timeout" keyword routes this to the local code model (it is a
                # code bug, not a visual hazard) and triggers lesson extraction.
                failure_reason = "Policy code timeout (likely an infinite loop in get_action)."
                break
            except Exception as e:
                if verbose:
                    print(f"Policy threw exception: {e}")
                done = True
                runtime_error = f"{type(e).__name__}: {e}"
                failure_reason = f"Policy code exception: {runtime_error}"
                break

            if type(action_string) is not str:
                done = True
                runtime_error = f"get_action returned non-string {type(action_string).__name__}"
                failure_reason = f"Policy code returned a non-string action: {type(action_string).__name__}"
                break
            if len(action_string) > 128:
                done = True
                runtime_error = f"get_action returned an oversized action string ({len(action_string)} characters)"
                failure_reason = "Policy code returned an oversized action string."
                break

            # Record a rich trace entry at a fixed frame cadence.
            if frames_alive - last_trace_frame >= TRACE_INTERVAL:
                last_trace_frame = frames_alive
                trace.append(build_trace_entry(frames_alive, policy_state, action_string))

            action = action_string_to_array(action_string)

            frames_before_action = frames_alive
            for _ in range(action_repeat):
                if done or frames_alive >= max_frames:
                    break
                obs, reward, done, info = env.step(action)
                frames_alive += 1

            state = env.get_state()
            zone_act = update_authoritative_progress(state, frames_alive - frames_before_action)

            if stuck_counter > STUCK_FRAME_LIMIT:
                if verbose:
                    print("Sonic got stuck! Terminating run.")
                done = True
                level_suffix = ""
                if zone_act and zone_act[0] is not None:
                    level_suffix = f" (zone {zone_act[0]} act {zone_act[1]})"
                # "stuck" keyword routes this to the local code model (a physics/
                # logic bug) and triggers lesson extraction.
                failure_reason = "Sonic got stuck: stopped making forward progress for 8 seconds." + level_suffix
                break
    finally:
        runner.close()
        if vision_executor is not None:
            vision_executor.shutdown(wait=False)

    if failure_reason is not None:
        pass
    elif frames_alive >= max_frames:
        failure_reason = "Timeout reached."
    elif not done:
        failure_reason = "Unknown early termination."
    else:
        failure_reason = "Sonic lost a life or hit a fatal obstacle."

    if runtime_error is not None:
        fitness = 0.0
        max_x = 0
        components = {"runtime_error": runtime_error}
    elif completion_x:
        fitness, components = calculate_fitness(
            max_x,
            frames_alive,
            state.get('rings', 0),
            state.get('score', 0),
            completion_x=completion_x,
            levels_cleared=levels_cleared,
            cumulative_distance=cumulative_distance,
        )
    else:
        fitness, components = calculate_fitness(
            max_x,
            frames_alive,
            state.get('rings', 0),
            state.get('score', 0),
            levels_cleared=levels_cleared,
            cumulative_distance=cumulative_distance,
        )

    screenshot_path = capture_screenshot(env)
    montage_path = build_screenshot_montage(
        context_screenshots + ([screenshot_path] if screenshot_path else []),
        "artifacts/failures/latest_context_montage.png",
    )
    if montage_path:
        screenshot_path = montage_path

    return fitness, frames_alive, max_x, failure_reason, screenshot_path, trace[-10:], components

def policy_action_signature(code):
    actions = set()
    for match in re.finditer(r"return\s+['\"]([^'\"]*)['\"]", code):
        actions.add(match.group(1).strip() or "NOOP")
    if not actions:
        return "dynamic"
    return "|".join(sorted(actions))


def parse_pool_fitness(path):
    name = os.path.basename(path)
    if name.startswith("pool_"):
        name = name[len("pool_"):]
    if name.endswith(".py"):
        name = name[:-len(".py")]
    value = name.split("_", 1)[0]
    return float(value)


def update_pool(code, fitness, pool_dir="policies/pool", max_size=6):
    os.makedirs(pool_dir, exist_ok=True)
    code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest()[:8]
    new_path = os.path.join(pool_dir, f"pool_{fitness:.2f}_{code_hash}.py")
    atomic_write_text(new_path, code)

    pool = []
    for path in glob.glob(os.path.join(pool_dir, "pool_*.py")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                pool_code = f.read()
            pool.append(
                {
                    "fitness": parse_pool_fitness(path),
                    "path": path,
                    "signature": policy_action_signature(pool_code),
                }
            )
        except (OSError, UnicodeError, ValueError):
            pass

    species_leaders = {}
    for entry in sorted(pool, key=lambda item: item["fitness"], reverse=True):
        species_leaders.setdefault(entry["signature"], entry)

    selected_paths = []
    for entry in sorted(species_leaders.values(), key=lambda item: item["fitness"], reverse=True):
        if len(selected_paths) < max_size:
            selected_paths.append(entry["path"])

    for entry in sorted(pool, key=lambda item: item["fitness"], reverse=True):
        if len(selected_paths) >= max_size:
            break
        if entry["path"] not in selected_paths:
            selected_paths.append(entry["path"])

    for entry in pool:
        if entry["path"] not in selected_paths:
            try:
                os.remove(entry["path"])
            except OSError:
                pass


def load_pool_codes(pool_dir="policies/pool"):
    """Read every policy currently in the FunSearch population pool."""
    codes = []
    for pf in glob.glob(os.path.join(pool_dir, "pool_*.py")):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                codes.append(f.read())
        except (OSError, UnicodeError):
            pass
    return codes


def build_frontier_guard_candidate(working_code, trace, sample_count=3, x_radius=25):
    """Add one narrow recovery guard when the working policy repeatedly stalls."""
    samples = list(trace or [])[-sample_count:]
    if len(samples) < sample_count:
        return None

    zone = samples[-1].get("zone")
    act = samples[-1].get("act")
    xs = [int(sample.get("x", 0)) for sample in samples]
    velocities = [abs(float(sample.get("x_velocity", 0) or 0)) for sample in samples]
    if any((sample.get("zone"), sample.get("act")) != (zone, act) for sample in samples):
        return None
    if max(xs) - min(xs) > x_radius or max(velocities) >= 0.5:
        return None

    frontier_x = round(sum(xs) / len(xs))
    marker = f"# FRONTIER_GUARD zone={zone} act={act} x={frontier_x}"
    for existing_zone, existing_act, existing_x in re.findall(
        r"# FRONTIER_GUARD zone=(\S+) act=(\S+) x=(-?\d+)",
        working_code,
    ):
        if (
            existing_zone == str(zone)
            and existing_act == str(act)
            and abs(int(existing_x) - frontier_x) <= x_radius * 2
        ):
            return None

    last_action = str(samples[-1].get("action", ""))
    recovery_action = "RIGHT,B" if "DOWN" in last_action or "B" not in last_action else "RIGHT"

    try:
        tree = ast.parse(working_code)
        function = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_action"
        )
        first_body_line = function.body[0].lineno - 1
    except (SyntaxError, StopIteration, IndexError):
        return None

    lines = working_code.splitlines(keepends=True)
    indent = lines[first_body_line][:len(lines[first_body_line]) - len(lines[first_body_line].lstrip())]
    lower = frontier_x - x_radius
    upper = frontier_x + x_radius
    guard = [
        f"{indent}{marker}\n",
        f"{indent}if (\n",
        f"{indent}    state.get(\"zone\") == {zone!r}\n",
        f"{indent}    and state.get(\"act\") == {act!r}\n",
        f"{indent}    and {lower} <= state.get(\"x_pos\", 0) <= {upper}\n",
        f"{indent}    and abs(state.get(\"x_velocity\", 0)) < 0.5\n",
        f"{indent}):\n",
        f"{indent}    return \"{recovery_action}\"\n",
        "\n",
    ]
    candidate = "".join(lines[:first_body_line] + guard + lines[first_body_line:])
    try:
        validate_policy_source(candidate)
    except Exception:
        return None
    return candidate


def frontier_guard_marker(code):
    match = re.search(r"# FRONTIER_GUARD zone=\S+ act=\S+ x=-?\d+", code or "")
    return match.group(0) if match else None


def recently_attempted_frontier_guard(marker, recent_history):
    for entry in recent_history or []:
        recorded_marker = entry.get("frontier_guard_marker")
        if recorded_marker is None:
            recorded_marker = entry.get("components", {}).get("frontier_guard_marker")
        recorded_markers = entry.get("components", {}).get("frontier_guard_markers", [])
        if marker in recorded_markers:
            return True
        if recorded_marker == marker:
            return True
    return False


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
):
    """Request ``n_candidates`` new policies from the mutator.

    Most are mutations of the working policy; when the pool is large enough a
    fraction become FunSearch crossovers of two pooled parents. Returns a list
    of ``(code, reasoning)`` tuples. A request that errors falls back to the
    working policy so every slot is filled.
    """
    candidates_code = [None] * n_candidates
    first_llm_slot = 0
    frontier_guard = build_frontier_guard_candidate(working_code, last_trace)
    if frontier_guard is not None and recently_attempted_frontier_guard(
        frontier_guard_marker(frontier_guard),
        recent_history,
    ):
        frontier_guard = None
    if n_candidates > 0 and frontier_guard is not None:
        print("Adding deterministic frontier-guard candidate.")
        candidates_code[0] = (frontier_guard, "Deterministic frontier guard")
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
        }
    def number(entry, key, default, converter):
        try:
            value = converter(entry.get(key, default))
            if isinstance(value, float) and not math.isfinite(value):
                return default
            return value
        except (OverflowError, TypeError, ValueError):
            return default

    return {
        "all_time_champion_fitness": max(
            number(entry, "fitness", -1.0, float) for entry in history_entries
        ),
        "start_generation": number(history_entries[-1], "generation", 0, int) + 1,
        "stagnation_counter": number(
            history_entries[-1], "stagnation_counter", 0, int
        ),
    }


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
    baseline_context = evaluate_working_baseline(env, working_path, mutator, max_frames, action_repeat=action_repeat)
    working_fitness = baseline_context["working_fitness"]
    working_fitness = max(working_fitness, all_time_champion_fitness)
    working_frontier = {
        "failure_reason": baseline_context["last_failure_reason"],
        "trace": baseline_context["last_trace"],
        "screenshot": preserve_frontier_screenshot(baseline_context["last_screenshot"]),
    }
    seed_population_baseline(population, working_path, baseline_context)

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
        attempted_frontier_markers = set()

        # Generate candidates
        recent_history = history.get_recent_history(3)
        pool_codes = load_pool_codes()
        candidates_code = generate_candidates(
            mutator,
            working_code,
            mutation_frontier["failure_reason"],
            mutation_frontier["screenshot"],
            recent_history,
            mutation_frontier["trace"],
            n_candidates,
            pool_codes,
            parent_selector=population.select_parent_codes,
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

            marker = frontier_guard_marker(new_code)
            if reasoning == "Deterministic frontier guard" and marker is not None:
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

        if best_candidate_promotable and best_candidate_fitness > working_fitness:
            print(f"Working policy improved from {working_fitness:.2f} -> {best_candidate_fitness:.2f}")
            working_fitness = best_candidate_fitness
            stagnation_counter = 0
            working_frontier = select_working_frontier_context(
                working_frontier,
                {
                    "failure_reason": best_candidate_reason,
                    "trace": best_candidate_trace,
                    "screenshot": preserve_frontier_screenshot(best_candidate_screenshot),
                },
                promoted=True,
            )

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
            history.log_generation(gen, choose_generation_archive_path(best_candidate_path, working_path), best_candidate_fitness, best_candidate_reason, best_candidate_screenshot, best_candidate_reasoning, stagnation_counter, best_candidate_components)
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
            history.log_generation(gen, choose_generation_archive_path(best_candidate_path, working_path), best_candidate_fitness, best_candidate_reason, best_candidate_screenshot, best_candidate_reasoning, stagnation_counter, best_candidate_components)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Sonic policy evolution loop.")
    parser.add_argument("--generations", type=int, help="Number of additional generations to run.")
    parser.add_argument("--frames", type=int, default=12000, help="Maximum emulator frames per candidate.")
    args = parser.parse_args(argv)

    run_evaluation_loop(generations=args.generations, max_frames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
