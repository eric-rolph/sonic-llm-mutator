import subprocess
import os

def run_local_ci():
    """
    Simulates a CI/CD pipeline locally to avoid headless display issues on GitHub runners.
    In a real scenario, this would run on every 'Merge Request' created by the LLM.
    """
    print("=== Starting Local CI Pipeline ===")
    
    # Check if the policy exists
    policy_path = os.path.join("policies", "current_policy.py")
    if not os.path.exists(policy_path):
        print("Error: No proposed policy found.")
        return False
        
    print(f"Evaluating proposed policy: {policy_path}")
    
    # We run main.py in a single generation mode to test it
    # We could modify main.py to take arguments, but for the simulator,
    # we'll just execute it and let it run one cycle if it's configured for it.
    
    # A robust CI would compare the fitness of the new policy against the main branch.
    # Here we just run the execution loop.
    try:
        # Run main.py as a subprocess
        env = os.environ.copy()
        result = subprocess.run(["python", "main.py"], env=env, capture_output=True, text=True)
        
        print("--- CI Output ---")
        print(result.stdout)
        
        if result.returncode == 0:
            print("=== Local CI Pipeline SUCCESS ===")
            print("Policy merged to main branch.")
            return True
        else:
            print("=== Local CI Pipeline FAILED ===")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"CI Execution Error: {e}")
        return False

if __name__ == "__main__":
    run_local_ci()
