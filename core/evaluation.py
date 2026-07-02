"""Run one policy episode against the emulator and score it.

This is the inner loop of the evolutionary pipeline: continuous multi-act
play-through with stuck detection, trace sampling for mutation prompts, and
cadenced proactive vision polling.
"""

import concurrent.futures
import os

from core.actions import action_string_to_array
from core.evaluator import calculate_fitness
from core.policy_runner import PolicyRunner, PolicyTimeout
from core.trace_context import build_screenshot_montage, build_trace_entry
from llm.mutator import vision_location_key

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
# When the stuck detector fires this far behind the act's max-x after a death,
# the run really ended with a DEATH AT THE FRONTIER (respawn puts Sonic behind
# his own max-x, which the stuck counter can never beat). Classifying that as
# "stuck at the respawn x" aimed diagnosis and guards at the wrong coordinates.
FRONTIER_RESPAWN_MARGIN = 200


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


def poll_vision_label(mutator, screenshot_path, location_key):
    """Run one proactive vision poll, storing the label in the mutator's cache.

    The store hook is duck-typed so lightweight test/benchmark mutators that
    only define ``analyze_environment`` keep working unchanged.
    """
    label = mutator.analyze_environment(screenshot_path)
    store = getattr(mutator, "store_vision_context", None)
    if store is not None and location_key:
        store(location_key, label)
    return label


def evaluate_policy(env, policy, mutator, max_frames=5000, verbose=True, action_repeat=1, snapshot_sink=None):
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
    deaths = 0
    last_lives = None

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
        # Savestate ring for agentic failure diagnosis; cadence and error
        # handling live inside the sink, so this is a cheap no-op most frames.
        # max_x is the settle-aware per-act frontier the sink needs for honest
        # experiment verdicts (raw x_pos goes stale across act transitions).
        if snapshot_sink is not None:
            snapshot_sink.record(env, frames_alive, state, act_max_x=max_x)
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
                    # A location whose label is already cached resolves
                    # synchronously: same context every run, no API call.
                    location_key = vision_location_key(state)
                    cache_lookup = getattr(mutator, "cached_vision_context", None)
                    cached_context = (
                        cache_lookup(location_key)
                        if cache_lookup is not None and location_key
                        else None
                    )
                    if cached_context:
                        current_vision_context = cached_context
                    else:
                        slot = vision_poll_count % CONTEXT_SCREENSHOT_SLOTS
                        vision_poll_count += 1
                        context_path = f"artifacts/failures/context_slot{slot}.png"
                        shot = capture_screenshot(env, context_path)
                        if shot:
                            context_screenshots.append(shot)
                            context_screenshots = context_screenshots[-CONTEXT_SCREENSHOT_SLOTS:]
                            vision_future = vision_executor.submit(
                                poll_vision_label, mutator, shot, location_key
                            )

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
            if snapshot_sink is not None:
                snapshot_sink.record(env, frames_alive, state, act_max_x=max_x)

            lives_now = state.get("lives")
            if isinstance(lives_now, int):
                if isinstance(last_lives, int) and lives_now < last_lives:
                    deaths += 1
                last_lives = lives_now

            if stuck_counter > STUCK_FRAME_LIMIT:
                done = True
                current_x = int(state.get("x_pos", 0) or 0)
                zone_label = ""
                if zone_act and zone_act[0] is not None:
                    zone_label = f"zone {zone_act[0]} act {zone_act[1]}"
                if deaths > 0 and max_x - current_x > FRONTIER_RESPAWN_MARGIN:
                    # A death at the frontier, not a physical stall: the respawn
                    # put Sonic behind his own max-x, which the stuck counter can
                    # never beat. Report the REAL frontier so diagnosis and
                    # guards aim at the death spot, not the respawn point.
                    # ("lost a life" keeps this diagnosable + vision-routed.)
                    if verbose:
                        print(f"Sonic died at the frontier (x={max_x}) and respawned behind it. Terminating run.")
                    failure_reason = (
                        f"Sonic lost a life at the frontier ({zone_label}, x={max_x}) "
                        f"and respawned behind it (run ended at x={current_x})."
                    )
                else:
                    if verbose:
                        print("Sonic got stuck! Terminating run.")
                    # "stuck" keeps the existing lesson-extraction/routing.
                    failure_reason = (
                        "Sonic got stuck: stopped making forward progress for 8 seconds."
                        + (f" ({zone_label})" if zone_label else "")
                    )
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

    # Authoritative frontier for downstream guard/diagnosis targeting. After a
    # death-then-respawn the trace tail sits at the respawn point, so consumers
    # that aim at "where the run actually got blocked" must use this instead.
    if runtime_error is None and zone_act and zone_act[0] is not None:
        components = dict(components)
        components["frontier"] = {"zone": zone_act[0], "act": zone_act[1], "x": int(max_x)}

    screenshot_path = capture_screenshot(env)
    montage_path = build_screenshot_montage(
        context_screenshots + ([screenshot_path] if screenshot_path else []),
        "artifacts/failures/latest_context_montage.png",
    )
    if montage_path:
        screenshot_path = montage_path

    return fitness, frames_alive, max_x, failure_reason, screenshot_path, trace[-10:], components
