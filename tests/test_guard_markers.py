"""The recorded marker must be the one a candidate INTRODUCED.

Agency review (confirmed by both skeptics): the old type-priority search over
the whole candidate file recorded a stale promoted FRONTIER_GUARD marker for
diagnosis/LLM guard candidates, so the retry dedupe never fired (failed guards
were rebuilt every generation) — and a same-type search suppressed legitimate
new frontier guards behind their promoted predecessor's marker.
"""

import unittest

from core.frontier import guard_markers, new_guard_marker

BASE = """def get_action(state):
    # FRONTIER_GUARD zone=0 act=1 x=1077
    if 1052 <= state.get("x_pos", 0) <= 1102:
        return "RIGHT,B"
    return 'RIGHT'
"""


class NewGuardMarkerTests(unittest.TestCase):
    def test_new_diagnosis_marker_wins_over_inherited_frontier_marker(self):
        candidate = BASE.replace(
            "def get_action(state):\n",
            "def get_action(state):\n    # DIAGNOSIS_GUARD zone=0 act=1 x=4200\n",
        )
        self.assertEqual(
            new_guard_marker(BASE, candidate), "# DIAGNOSIS_GUARD zone=0 act=1 x=4200"
        )

    def test_new_frontier_marker_not_shadowed_by_promoted_predecessor(self):
        # Same TYPE as the inherited guard: the old search returned the OLD
        # marker (insertion lands below the previous marker comment).
        candidate = BASE.replace(
            "def get_action(state):\n",
            "def get_action(state):\n    # FRONTIER_GUARD zone=0 act=1 x=4929\n",
        )
        self.assertEqual(
            new_guard_marker(BASE, candidate), "# FRONTIER_GUARD zone=0 act=1 x=4929"
        )

    def test_llm_guard_marker_detected(self):
        candidate = BASE + "\n# LLM_GUARD zone=0 act=1 x=4268\n"
        self.assertEqual(new_guard_marker(BASE, candidate), "# LLM_GUARD zone=0 act=1 x=4268")

    def test_no_new_marker_returns_none(self):
        self.assertIsNone(new_guard_marker(BASE, BASE))
        self.assertIsNone(new_guard_marker(BASE, "def get_action(state):\n    return 'LEFT'\n"))

    def test_guard_markers_finds_all_types(self):
        code = (
            "# FRONTIER_GUARD zone=0 act=1 x=1\n"
            "# DIAGNOSIS_GUARD zone=0 act=1 x=2\n"
            "# LLM_GUARD zone=0 act=1 x=3\n"
        )
        self.assertEqual(len(guard_markers(code)), 3)


if __name__ == "__main__":
    unittest.main()
