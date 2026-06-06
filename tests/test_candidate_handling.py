import os
import tempfile
import unittest

from main import (
    build_policy_load_failure,
    clear_candidate_recording,
    load_policy,
    prepare_candidate_policy,
    record_candidate_evaluation,
)


class RecordingArchive:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def record_evaluation(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error


class RepairingMutator:
    def __init__(self, repaired_code=None, error=None):
        self.repaired_code = repaired_code
        self.error = error
        self.calls = []

    def repair_policy(self, code, validation_error):
        self.calls.append((code, validation_error))
        if self.error:
            raise self.error
        return self.repaired_code, "validator repair"


class CandidateHandlingTests(unittest.TestCase):
    def test_build_policy_load_failure_uses_specific_reason(self):
        fitness, frames, max_x, reason, screenshot, trace, components = build_policy_load_failure(
            SyntaxError("bad syntax")
        )

        self.assertEqual(fitness, 0.0)
        self.assertEqual(frames, 0)
        self.assertEqual(max_x, 0)
        self.assertIn("Policy failed to load", reason)
        self.assertIn("bad syntax", reason)
        self.assertIsNone(screenshot)
        self.assertEqual(trace, [])
        self.assertEqual(components["load_error"], "bad syntax")

    def test_clear_candidate_recording_removes_stale_bk2(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale_path = os.path.join(tmp, "candidate_0.bk2")
            with open(stale_path, "w", encoding="utf-8") as f:
                f.write("stale")

            clear_candidate_recording(tmp, 0)

            self.assertFalse(os.path.exists(stale_path))

    def test_record_candidate_evaluation_forwards_all_candidate_metadata(self):
        archive = RecordingArchive()
        trace = [{"zone": 0, "act": 1, "x": 1077}]
        components = {"distance": 2000}

        recorded = record_candidate_evaluation(
            archive,
            "def get_action(state):\n    return 'RIGHT'\n",
            123.0,
            components,
            "stuck",
            trace,
            "try jumping",
        )

        self.assertTrue(recorded)
        _, kwargs = archive.calls[0]
        self.assertEqual(kwargs["fitness"], 123.0)
        self.assertEqual(kwargs["components"], components)
        self.assertEqual(kwargs["failure_reason"], "stuck")
        self.assertEqual(kwargs["trace"], trace)
        self.assertEqual(kwargs["reasoning"], "try jumping")

    def test_record_candidate_evaluation_does_not_stop_training_on_archive_error(self):
        archive = RecordingArchive(error=OSError("disk full"))

        recorded = record_candidate_evaluation(archive, "code", 0.0, {}, "fatal", [], "")

        self.assertFalse(recorded)

    def test_load_policy_validates_before_executing_top_level_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = os.path.join(tmp, "candidate.py")
            marker_path = os.path.join(tmp, "owned.txt")
            source = (
                f'open({marker_path!r}, "w").write("bad")\n'
                "def get_action(state):\n"
                "    return 'RIGHT'\n"
            )
            with open(policy_path, "w", encoding="utf-8") as f:
                f.write(source)

            with self.assertRaises(ValueError):
                load_policy(policy_path)

            self.assertFalse(os.path.exists(marker_path))

    def test_prepare_candidate_policy_does_not_repair_valid_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = os.path.join(tmp, "candidate.py")
            archive = RecordingArchive()
            mutator = RepairingMutator("def get_action(state):\n    return 'LEFT'\n")
            source = "def get_action(state):\n    return 'RIGHT'\n"

            result = prepare_candidate_policy(candidate_path, source, "generated", mutator, archive)

        self.assertEqual(result["code"], source)
        self.assertIsNotNone(result["policy"])
        self.assertIsNone(result["load_error"])
        self.assertEqual(mutator.calls, [])
        self.assertEqual(archive.calls, [])

    def test_prepare_candidate_policy_archives_invalid_source_then_repairs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = os.path.join(tmp, "candidate.py")
            archive = RecordingArchive()
            repaired = "def get_action(state):\n    return 'RIGHT'\n"
            mutator = RepairingMutator(repaired)
            invalid = "def broken(:\n    pass"

            result = prepare_candidate_policy(candidate_path, invalid, "generated", mutator, archive)

        self.assertEqual(len(mutator.calls), 1)
        self.assertIn("syntax", mutator.calls[0][1].lower())
        self.assertEqual(archive.calls[0][0][0], invalid)
        self.assertEqual(archive.calls[0][1]["fitness"], 0.0)
        self.assertEqual(result["code"], repaired)
        self.assertIsNotNone(result["policy"])
        self.assertIsNone(result["load_error"])
        self.assertIn("validator repair", result["reasoning"])

    def test_prepare_candidate_policy_does_not_recurse_when_repair_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = os.path.join(tmp, "candidate.py")
            archive = RecordingArchive()
            mutator = RepairingMutator("still invalid")

            result = prepare_candidate_policy(candidate_path, "also invalid", "generated", mutator, archive)

        self.assertEqual(len(mutator.calls), 1)
        self.assertIsNone(result["policy"])
        self.assertIsNotNone(result["load_error"])
        self.assertEqual(result["code"], "still invalid")

    def test_prepare_candidate_policy_treats_empty_repair_as_load_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = os.path.join(tmp, "candidate.py")
            archive = RecordingArchive()
            mutator = RepairingMutator(None)

            result = prepare_candidate_policy(candidate_path, "invalid", "generated", mutator, archive)

        self.assertEqual(len(mutator.calls), 1)
        self.assertEqual(result["code"], "")
        self.assertIsNone(result["policy"])
        self.assertIsNotNone(result["load_error"])


if __name__ == "__main__":
    unittest.main()
