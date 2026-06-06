import json
import os
import tempfile
import unittest

from core.history import EvolutionHistory


class EvolutionHistoryTests(unittest.TestCase):
    def make_history(self, directory):
        return EvolutionHistory(
            log_path=os.path.join(directory, "artifacts", "history.json"),
            archive_dir=os.path.join(directory, "policies", "archive"),
        )

    def test_loads_existing_valid_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "artifacts", "history.json")
            os.makedirs(os.path.dirname(log_path))
            expected = [{"generation": 7, "fitness": 123.0}]
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(expected, f)

            history = self.make_history(tmp)

            self.assertEqual(history.history, expected)

    def test_missing_history_file_starts_with_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = self.make_history(tmp)

            self.assertEqual(history.history, [])

    def test_empty_history_file_recovers_with_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "artifacts", "history.json")
            os.makedirs(os.path.dirname(log_path))
            open(log_path, "w", encoding="utf-8").close()

            history = self.make_history(tmp)

            self.assertEqual(history.history, [])

    def test_truncated_history_file_recovers_with_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "artifacts", "history.json")
            os.makedirs(os.path.dirname(log_path))
            with open(log_path, "w", encoding="utf-8") as f:
                f.write('[{"generation": 7')

            history = self.make_history(tmp)

            self.assertEqual(history.history, [])

    def test_valid_json_with_wrong_shape_recovers_with_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "artifacts", "history.json")
            os.makedirs(os.path.dirname(log_path))
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({"generation": 7}, f)

            history = self.make_history(tmp)

            self.assertEqual(history.history, [])

    def test_history_filters_non_object_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "artifacts", "history.json")
            os.makedirs(os.path.dirname(log_path))
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([None, {"generation": 7}], f)

            history = self.make_history(tmp)

            self.assertEqual(history.history, [{"generation": 7}])

    def test_failed_save_preserves_existing_history_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = self.make_history(tmp)
            history.history = [{"generation": 1}]
            history._save_history()
            history.history.append({"not_json_serializable": object()})

            with self.assertRaises(TypeError):
                history._save_history()

            with open(history.log_path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), [{"generation": 1}])
            self.assertEqual(
                os.listdir(os.path.dirname(history.log_path)),
                ["history.json"],
            )

    def test_save_rejects_non_finite_numbers_without_corrupting_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = self.make_history(tmp)
            history.history = [{"generation": 1, "fitness": 10.0}]
            history._save_history()
            history.history.append({"generation": 2, "fitness": float("nan")})

            with self.assertRaises(ValueError):
                history._save_history()

            with open(history.log_path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), [{"generation": 1, "fitness": 10.0}])


if __name__ == "__main__":
    unittest.main()
