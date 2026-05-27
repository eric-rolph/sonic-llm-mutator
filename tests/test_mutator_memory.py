import unittest
from contextlib import redirect_stdout
from io import StringIO

from llm.mutator import (
    MutatorClient,
    dedupe_lessons,
    extract_json_object,
    normalize_lesson,
    normalize_vision_context,
    select_relevant_lessons,
)


class MutatorMemoryTests(unittest.TestCase):
    def test_extract_json_object_handles_fenced_json(self):
        text = """Here is the lesson:
```json
{"x": 9767, "hazard": "Pit", "lesson": "When at X=9767, jump."}
```
"""

        self.assertEqual(
            extract_json_object(text),
            {"x": 9767, "hazard": "Pit", "lesson": "When at X=9767, jump."},
        )

    def test_normalize_lesson_rejects_malformed_entries(self):
        self.assertIsNone(normalize_lesson({"x": "oops", "lesson": "jump"}))
        self.assertIsNone(normalize_lesson({"x": 100, "hazard": "Pit"}))
        self.assertEqual(
            normalize_lesson({"x": "287", "hazard": "Pit", "lesson": "Jump"}),
            {"x": 287, "hazard": "Pit", "lesson": "Jump"},
        )

    def test_dedupe_lessons_collapses_nearby_duplicate_lessons(self):
        lessons = [
            {"x": 9767, "hazard": "Pit", "lesson": "Jump to avoid pit"},
            {"x": 9768, "hazard": "Pit", "lesson": "Jump to avoid pit"},
            {"x": 3061, "hazard": "Stuck", "lesson": "Hold RIGHT"},
        ]

        deduped = dedupe_lessons(lessons, x_tolerance=5)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["x"], 9767)
        self.assertEqual(deduped[1]["x"], 3061)

    def test_select_relevant_lessons_filters_by_coordinate_and_limit(self):
        lessons = [
            {"x": 100, "hazard": "Pit", "lesson": "Jump at 100"},
            {"x": 200, "hazard": "Pit", "lesson": "Jump at 200"},
            {"x": 2000, "hazard": "Wall", "lesson": "Do not include"},
        ]

        relevant = select_relevant_lessons(lessons, current_x=150, radius=1000, limit=1)

        self.assertEqual(relevant, ["- Jump at 200"])

    def test_normalize_vision_context_handles_empty_model_content(self):
        self.assertEqual(normalize_vision_context(None), "UNKNOWN")
        self.assertEqual(normalize_vision_context(" spikes "), "SPIKES")

    def test_mutate_policy_uses_micro_model_when_screenshot_missing(self):
        class RecordingMutator(MutatorClient):
            def __init__(self):
                self.called = []

            def _call_micro_model(self, prompt, temperature=0.7):
                self.called.append("micro")
                return "def get_action(state):\n    return 'RIGHT'", "micro"

            def _call_macro_model(self, prompt, image_path):
                self.called.append("macro")
                return "def get_action(state):\n    return 'LEFT'", "macro"

        mutator = RecordingMutator()

        with redirect_stdout(StringIO()):
            code, _ = mutator.mutate_policy(
                current_code="def get_action(state):\n    return 'RIGHT'",
                failure_reason="Sonic lost a life or hit a fatal obstacle.",
                screenshot_path=None,
                recent_history=[],
            )

        self.assertEqual(mutator.called, ["micro"])
        self.assertIn("return 'RIGHT'", code)


if __name__ == "__main__":
    unittest.main()
