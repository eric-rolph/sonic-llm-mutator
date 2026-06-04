import subprocess
import os
import sys

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
        
    print(f"Checking proposed policy exists: {policy_path}")
    
    # A robust CI would also compare candidate fitness against the main branch.
    # The local smoke check stays bounded and dependency-light by running tests.
    try:
        env = os.environ.copy()
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        print("--- CI Output ---")
        print(result.stdout)
        
        if result.returncode == 0:
            print("=== Local CI Pipeline SUCCESS ===")
            return True
        else:
            print("=== Local CI Pipeline FAILED ===")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"CI Execution Error: {e}")
        return False

def main():
    return 0 if run_local_ci() else 1


if __name__ == "__main__":
    raise SystemExit(main())
