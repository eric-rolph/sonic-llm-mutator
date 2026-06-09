import json
import os
import shutil
import tempfile
import time


class EvolutionHistory:
    def __init__(self, log_path="artifacts/history.json", archive_dir="policies/archive"):
        self.log_path = log_path
        self.archive_dir = archive_dir
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [entry for entry in payload if isinstance(entry, dict)]

    def _save_history(self):
        directory = os.path.dirname(self.log_path)
        fd, temp_path = tempfile.mkstemp(
            dir=directory,
            prefix=".history-",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.log_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def log_generation(self, generation, policy_file, fitness, failure_reason, screenshot_path, llm_reasoning, stagnation_counter=0, components=None, max_frames=None):
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
            "components": components or {},
            "stagnation_counter": stagnation_counter,
            "failure_reason": failure_reason,
            "screenshot": screenshot_path,
            "archive_path": archive_path,
            "llm_reasoning": llm_reasoning,
            # Frame budget the fitness was measured under. Fitness numbers are
            # only comparable across runs that share a budget (see
            # resolve_working_fitness_floor in main.py).
            "max_frames": max_frames,
        }
        self.history.append(entry)
        self._save_history()

    def get_recent_history(self, num_entries=3):
        """Returns the most recent failed generations to provide context to the LLM."""
        return self.history[-num_entries:] if self.history else []
