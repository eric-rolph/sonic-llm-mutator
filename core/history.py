import json
import os
import shutil
import time

class EvolutionHistory:
    def __init__(self, log_path="artifacts/history.json", archive_dir="policies/archive"):
        self.log_path = log_path
        self.archive_dir = archive_dir
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def log_generation(self, generation, policy_file, fitness, failure_reason, screenshot_path, llm_reasoning):
        # Archive the policy file
        timestamp = int(time.time())
        archive_name = f"gen_{generation}_{timestamp}.py"
        archive_path = os.path.join(self.archive_dir, archive_name)
        
        if os.path.exists(policy_file):
            shutil.copy2(policy_file, archive_path)

        entry = {
            "generation": generation,
            "timestamp": timestamp,
            "fitness": fitness,
            "failure_reason": failure_reason,
            "screenshot": screenshot_path,
            "archive_path": archive_path,
            "llm_reasoning": llm_reasoning
        }
        self.history.append(entry)
        self._save_history()

    def get_recent_history(self, num_entries=3):
        """Returns the most recent failed generations to provide context to the LLM."""
        return self.history[-num_entries:] if self.history else []
