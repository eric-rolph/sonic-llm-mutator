import os
import time
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

def run_evaluation_loop(max_generations=1, max_frames=200):
    # Initialize components
    history = EvolutionHistory()
    mutator = MutatorClient()
    
    try:
        from emulator.sonic_env import SonicEnvWrapper
        env = SonicEnvWrapper()
    except ImportError:
        print("Warning: stable-retro not installed. Running in dry-run mode.")
        env = None

    policy_path = os.path.join("policies", "current_policy.py")
    
    for gen in range(1, max_generations + 1):
        print(f"\n--- Generation {gen} ---")
        
        # 1. Load Policy
        if not os.path.exists(policy_path):
            print(f"Policy file {policy_path} not found. Creating a seed policy.")
            mutator.write_seed_policy(policy_path)
            
        policy = load_policy(policy_path)
        
        if env is None:
            print("Dry run: Skipping actual environment step. Mutating directly.")
            screenshot_path = "artifacts/failures/mock_screenshot.png"
            # create mock screenshot
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            with open(screenshot_path, "w") as f:
                f.write("mock")
            failure_reason = "Mock failure."
            fitness = 0.0
            
        else:
            # 2. Run Policy
            obs = env.reset()
            frames_alive = 0
            max_x = 0
            
            done = False
            stuck_counter = 0
            last_x = 0
            
            while not done and frames_alive < max_frames:
                # Extract state
                state = env.get_state()
                
                # Policy decides action
                try:
                    action_string = policy.get_action(state)
                except Exception as e:
                    print(f"Policy threw exception: {e}")
                    action_string = "RIGHT" # fallback
                    
                # Map action string to array
                buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
                action = [0] * 12
                for p in action_string.split(','):
                    if p.strip() in buttons:
                        action[buttons.index(p.strip())] = 1
                
                # Step env
                obs, reward, done, info = env.step(action)
                frames_alive += 1
                
                # Track max X and check if stuck
                current_x = state.get('x_pos', 0)
                if current_x > max_x:
                    max_x = current_x
                    stuck_counter = 0
                elif current_x <= last_x + 1:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                    
                last_x = current_x
                
                # Stuck threshold (e.g. 5 seconds at 60fps = 300 frames)
                if stuck_counter > 300:
                    print("Sonic got stuck! Terminating run.")
                    done = True
                    failure_reason = "Sonic stopped making forward progress for 5 seconds."
                    break
            
            if frames_alive >= max_frames:
                failure_reason = "Timeout reached."
            elif not done:
                failure_reason = "Unknown early termination."
            else:
                failure_reason = "Sonic lost a life or hit a fatal obstacle."
            
            # 3. Calculate Fitness
            fitness = calculate_fitness(max_x, frames_alive, state.get('rings', 0), state.get('score', 0))
            print(f"Fitness achieved: {fitness:.2f} (Max X: {max_x}, Frames: {frames_alive})")
            
            # 4. Capture screenshot
            screenshot_path = env.get_screenshot()

        # 5. Mutate via LLM
        print("Requesting LLM Mutation...")
        recent_history = history.get_recent_history(3)
        
        with open(policy_path, 'r') as f:
            current_code = f.read()
            
        new_code, llm_reasoning = mutator.mutate_policy(current_code, failure_reason, screenshot_path, recent_history)
        
        # 6. Log History & Archive Old Policy
        history.log_generation(gen, policy_path, fitness, failure_reason, screenshot_path, llm_reasoning)
        
        # 7. Write New Policy
        with open(policy_path, 'w') as f:
            f.write(new_code)
            
        print("New policy written. Loop restarting.")

if __name__ == "__main__":
    run_evaluation_loop()
