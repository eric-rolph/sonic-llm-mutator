import os
import time
import sys
import importlib.util
sys.path.insert(0, os.path.abspath("."))

from core.evaluator import calculate_fitness
from core.history import EvolutionHistory
from core.trace_context import build_screenshot_montage, build_trace_entry
from llm.mutator import MutatorClient

def load_policy(filepath):
    """Loads a python script dynamically."""
    spec = importlib.util.spec_from_file_location("current_policy", filepath)
    policy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy_module)
    return policy_module


def evaluate_working_baseline(env, working_path, mutator, max_frames=5000, verbose=True, action_repeat=1):
    context = {
        "working_fitness": -1.0,
        "last_failure_reason": "Initial seed run",
        "last_screenshot": None,
        "last_trace": [],
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
        }
    )
    return context


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
    
    current_vision_context = "UNKNOWN"
    
    trace = []
    context_screenshots = []
    
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    while not done and frames_alive < max_frames:
        state = env.get_state()
        state_completion_x = state.get("screen_x_end") or state.get("completion_x")
        if state_completion_x:
            completion_x = state_completion_x
        
        # Proactive Vision Polling: every 300 frames (~5 sec)
        if frames_alive % 300 == 0:
            context_path = f"artifacts/failures/context_{frames_alive:06d}.png"
            shot = capture_screenshot(env, context_path)
            if shot:
                context_screenshots.append(shot)
                context_screenshots = context_screenshots[-3:]
                current_vision_context = mutator.analyze_environment(shot)
        
        state['vision_context'] = current_vision_context

        try:
            future = executor.submit(policy.get_action, state)
            action_string = future.result(timeout=0.5)
        except concurrent.futures.TimeoutError:
            if verbose:
                print("Policy timed out (infinite loop?)")
            action_string = ""
            done = True
            break
        except Exception as e:
            if verbose:
                print(f"Policy threw exception: {e}")
            action_string = "RIGHT"

        if not isinstance(action_string, str):
            action_string = "RIGHT"
        
        # Record rich trace every 30 frames (0.5 seconds at 60fps).
        if frames_alive % 30 == 0:
            trace.append(build_trace_entry(frames_alive, state, action_string))

        buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        action = [0] * 12
        for p in action_string.split(','):
            if p.strip() in buttons:
                action[buttons.index(p.strip())] = 1
        
        for _ in range(action_repeat):
            if done or frames_alive >= max_frames:
                break
            obs, reward, done, info = env.step(action)
            frames_alive += 1
        
        current_x = state.get('x_pos', 0)
        if current_x > max_x:
            max_x = current_x
            stuck_counter = 0
        else:
            stuck_counter += 1

        if stuck_counter > 500:
            if verbose:
                print("Sonic got stuck! Terminating run.")
            done = True
            failure_reason = "Sonic stopped making forward progress for 8 seconds."
            break
            
    if failure_reason is not None:
        pass
    elif frames_alive >= max_frames:
        failure_reason = "Timeout reached."
    elif not done:
        failure_reason = "Unknown early termination."
    else:
        failure_reason = "Sonic lost a life or hit a fatal obstacle."
        
    executor.shutdown(wait=False)
    
    if completion_x:
        fitness, components = calculate_fitness(
            max_x,
            frames_alive,
            state.get('rings', 0),
            state.get('score', 0),
            completion_x=completion_x,
        )
    else:
        fitness, components = calculate_fitness(max_x, frames_alive, state.get('rings', 0), state.get('score', 0))

    screenshot_path = capture_screenshot(env)
    montage_path = build_screenshot_montage(
        context_screenshots + ([screenshot_path] if screenshot_path else []),
        "artifacts/failures/latest_context_montage.png",
    )
    if montage_path:
        screenshot_path = montage_path
    
    return fitness, frames_alive, max_x, failure_reason, screenshot_path, trace[-10:], components

def policy_action_signature(code):
    import re

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
    import glob
    import hashlib

    os.makedirs(pool_dir, exist_ok=True)
    code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest()[:8]
    new_path = os.path.join(pool_dir, f"pool_{fitness:.2f}_{code_hash}.py")
    with open(new_path, "w") as f:
        f.write(code)

    pool = []
    for path in glob.glob(os.path.join(pool_dir, "pool_*.py")):
        try:
            with open(path, "r") as f:
                pool_code = f.read()
            pool.append(
                {
                    "fitness": parse_pool_fitness(path),
                    "path": path,
                    "signature": policy_action_signature(pool_code),
                }
            )
        except (OSError, ValueError):
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

def run_evaluation_loop(max_generations=500, max_frames=5000, n_candidates=2, stagnation_limit=5, action_repeat=1):
    history = EvolutionHistory()
    mutator = MutatorClient()
    
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
        import shutil
        shutil.copy(champion_path, working_path)
        
    all_time_champion_fitness = -1.0
    start_gen = 1
    history_file = "artifacts/history.json"
    if os.path.exists(history_file):
        try:
            import json
            with open(history_file, "r") as f:
                hist_data = json.load(f)
                if hist_data:
                    all_time_champion_fitness = max([entry.get('fitness', -1.0) for entry in hist_data])
                    start_gen = hist_data[-1].get('generation', 0) + 1
        except Exception as e:
            print(f"Failed to load history for fitness: {e}")
            
    stagnation_counter = 0
    baseline_context = evaluate_working_baseline(env, working_path, mutator, max_frames, action_repeat=action_repeat)
    working_fitness = baseline_context["working_fitness"]
    last_failure_reason = baseline_context["last_failure_reason"]
    last_trace = baseline_context["last_trace"]
    last_screenshot = baseline_context["last_screenshot"]

    for gen in range(start_gen, max_generations + 1):
        print(f"\n--- Generation {gen} ---")
        
        if stagnation_counter >= stagnation_limit:
            print(f"Stagnation detected ({stagnation_counter} gens without improvement). Triggering blankRestart!")
            mutator.write_seed_policy(working_path)
            working_fitness = -1.0
            stagnation_counter = 0
            last_failure_reason = "blankRestart due to stagnation"
            last_trace = []
        
        with open(working_path, 'r') as f:
            working_code = f.read()

        best_candidate_code = None
        best_candidate_fitness = -1.0
        best_candidate_reason = ""
        best_candidate_screenshot = None
        best_candidate_reasoning = ""
        best_candidate_trace = []
        best_candidate_components = {}
        best_candidate_idx = -1

        # Generate candidates
        candidates_code = [None] * n_candidates
        recent_history = history.get_recent_history(3)
        
        import glob
        import random
        pool_files = glob.glob("policies/pool/pool_*.py")
        pool_codes = []
        for pf in pool_files:
            with open(pf, "r") as f:
                pool_codes.append(f.read())
        
        import concurrent.futures
        # Ensure max_workers=1 to prevent hammering local LLMs that don't support concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {}
            for c in range(n_candidates):
                if len(pool_codes) >= 2 and random.random() < 0.30:
                    print(f"Requesting Crossover {c+1}/{n_candidates} (FunSearch)...")
                    parent_a, parent_b = random.sample(pool_codes, 2)
                    future = executor.submit(
                        mutator.crossover_policies,
                        parent_a,
                        parent_b,
                        recent_history,
                        temperature=0.7
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
                        last_trace
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
            
        # Evaluate candidates
        for c, (new_code, reasoning) in enumerate(candidates_code):
            print(f"\nEvaluating Candidate {c+1}...")
            candidate_path = os.path.join("policies", f"candidate_{c}.py")
            clear_candidate_recording("artifacts/videos/tmp", c)
            with open(candidate_path, 'w') as f:
                f.write(new_code)

            load_error = None
            try:
                policy = load_policy(candidate_path)
            except Exception as e:
                print(f"Failed to load policy (SyntaxError?): {e}")
                load_error = e
                policy = None

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
                
                import glob
                bk2_files = glob.glob("artifacts/videos/tmp/*.bk2")
                if bk2_files:
                    latest_bk2 = max(bk2_files, key=os.path.getmtime)
                    candidate_bk2_path = f"artifacts/videos/tmp/candidate_{c}.bk2"
                    if os.path.exists(candidate_bk2_path) and candidate_bk2_path != latest_bk2:
                        os.remove(candidate_bk2_path)
                    if candidate_bk2_path != latest_bk2:
                        os.rename(latest_bk2, candidate_bk2_path)
                
            if fitness > best_candidate_fitness:
                best_candidate_fitness = fitness
                best_candidate_code = new_code
                best_candidate_reason = failure_reason
                best_candidate_screenshot = screenshot_path
                best_candidate_reasoning = reasoning
                best_candidate_trace = trace
                best_candidate_components = components
                best_candidate_idx = c
                
        # Promote or Stagnate
        best_bk2 = f"artifacts/videos/tmp/candidate_{best_candidate_idx}.bk2"
        latest_mp4 = "artifacts/videos/latest.mp4"
        
        if os.path.exists(best_bk2):
            import subprocess
            print(f"Rendering video for generation's best candidate...")
            try:
                subprocess.Popen([sys.executable, "render_video.py", best_bk2, latest_mp4], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Failed to render video: {e}")

        if best_candidate_fitness > working_fitness:
            print(f"Working policy improved from {working_fitness:.2f} -> {best_candidate_fitness:.2f}")
            working_fitness = best_candidate_fitness
            stagnation_counter = 0
            
            # Update working file
            with open(working_path, 'w') as f:
                f.write(best_candidate_code)
                
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
                if os.path.exists(latest_mp4):
                    try:
                        import shutil
                        if os.path.exists(champion_mp4):
                            os.remove(champion_mp4)
                        shutil.copy(latest_mp4, champion_mp4)
                    except Exception as e:
                        print(f"Failed to copy champion video: {e}")
                
                # Update all-time champion file
                with open(champion_path, 'w') as f:
                    f.write(best_candidate_code)
            
            # Archive the winning candidate
            history.log_generation(gen, working_path, best_candidate_fitness, best_candidate_reason, best_candidate_screenshot, best_candidate_reasoning, stagnation_counter, best_candidate_components)
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
            history.log_generation(gen, working_path, best_candidate_fitness, best_candidate_reason, best_candidate_screenshot, best_candidate_reasoning, stagnation_counter, best_candidate_components)
            
        last_failure_reason = best_candidate_reason
        last_trace = best_candidate_trace
        last_screenshot = best_candidate_screenshot

if __name__ == "__main__":
    run_evaluation_loop()
