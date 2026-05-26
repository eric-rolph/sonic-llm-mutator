import os
import time
import sys
import importlib.util
from core.evaluator import calculate_fitness
from core.history import EvolutionHistory
from llm.mutator import MutatorClient
from emulator.sonic_env import SonicEnvWrapper

def load_policy(filepath):
    """Loads a python script dynamically."""
    spec = importlib.util.spec_from_file_location("current_policy", filepath)
    policy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy_module)
    return policy_module

def evaluate_policy(env, policy, mutator, max_frames=5000):
    obs = env.reset()
    frames_alive = 0
    max_x = 0
    done = False
    stuck_counter = 0
    last_x = 0
    
    current_vision_context = "UNKNOWN"
    
    # Store coordinate trace
    trace = []
    
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    while not done and frames_alive < max_frames:
        state = env.get_state()
        
        # Proactive Vision Polling: every 300 frames (~5 sec)
        if frames_alive % 300 == 0:
            shot = env.get_screenshot()
            if shot:
                current_vision_context = mutator.analyze_environment(shot)
        
        state['vision_context'] = current_vision_context
        
        # Record trace every 30 frames (0.5 seconds at 60fps)
        if frames_alive % 30 == 0:
            trace.append((state.get('x_pos', 0), state.get('y_pos', 0)))
        
        try:
            future = executor.submit(policy.get_action, state)
            action_string = future.result(timeout=0.5)
        except concurrent.futures.TimeoutError:
            print("Policy timed out (infinite loop?)")
            action_string = ""
            done = True
            break
        except Exception as e:
            print(f"Policy threw exception: {e}")
            action_string = "RIGHT"
            
        buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        action = [0] * 12
        for p in action_string.split(','):
            if p.strip() in buttons:
                action[buttons.index(p.strip())] = 1
        
        obs, reward, done, info = env.step(action)
        frames_alive += 1
        
        current_x = state.get('x_pos', 0)
        if current_x > max_x:
            max_x = current_x
            stuck_counter = 0
        else:
            stuck_counter += 1
            
        last_x = current_x
        
        if stuck_counter > 500:
            print("Sonic got stuck! Terminating run.")
            done = True
            failure_reason = "Sonic stopped making forward progress for 8 seconds."
            break
            
    if frames_alive >= max_frames:
        failure_reason = "Timeout reached."
    elif not done:
        failure_reason = "Unknown early termination."
    else:
        failure_reason = "Sonic lost a life or hit a fatal obstacle."
        
    executor.shutdown(wait=False)
    
    fitness, components = calculate_fitness(max_x, frames_alive, state.get('rings', 0), state.get('score', 0))
    screenshot_path = env.get_screenshot()
    
    return fitness, frames_alive, max_x, failure_reason, screenshot_path, trace[-10:], components

def update_pool(code, fitness):
    import glob
    os.makedirs("policies/pool", exist_ok=True)
    pool_files = glob.glob("policies/pool/pool_*.py")
    
    pool = []
    for f in pool_files:
        try:
            val = float(os.path.basename(f).replace("pool_", "").replace(".py", ""))
            pool.append((val, f))
        except:
            pass
            
    new_path = f"policies/pool/pool_{fitness:.2f}.py"
    with open(new_path, "w") as f:
        f.write(code)
    pool.append((fitness, new_path))
    
    pool.sort(key=lambda x: x[0], reverse=True)
    while len(pool) > 3:
        worst_val, worst_file = pool.pop()
        try:
            os.remove(worst_file)
        except:
            pass

def run_evaluation_loop(max_generations=500, max_frames=5000, n_candidates=2, stagnation_limit=5):
    history = EvolutionHistory()
    mutator = MutatorClient()
    
    try:
        from emulator.sonic_env import SonicEnvWrapper
        os.makedirs("artifacts/videos/tmp", exist_ok=True)
        env = SonicEnvWrapper(record_path="artifacts/videos/tmp")
    except ImportError:
        print("Warning: stable-retro not installed. Running in dry-run mode.")
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
            
    working_fitness = -1.0
    stagnation_counter = 0
    last_failure_reason = "Initial seed run"
    last_trace = []
    last_screenshot = None

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
            with open(candidate_path, 'w') as f:
                f.write(new_code)
                
            try:
                policy = load_policy(candidate_path)
            except Exception as e:
                print(f"Failed to load policy (SyntaxError?): {e}")
                policy = None
                
            if env is None or policy is None:
                fitness = 0.0
                failure_reason = "Mock failure."
                screenshot_path = "artifacts/failures/mock_screenshot.png"
                trace = []
                components = {}
            else:
                fitness, frames_alive, max_x, failure_reason, screenshot_path, trace, components = evaluate_policy(env, policy, mutator, max_frames)
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
