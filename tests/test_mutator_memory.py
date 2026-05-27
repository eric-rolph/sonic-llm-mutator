import unittest

from llm.mutator import (
    dedupe_lessons,
    extract_json_object,
    normalize_lesson,
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


if __name__ == "__main__":
    unittest.main()
