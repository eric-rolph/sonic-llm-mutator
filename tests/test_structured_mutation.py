import unittest
from contextlib import redirect_stdout
from io import StringIO

from llm.mutator import MutatorClient

WORKING = "def get_action(state):\n    return 'RIGHT,DOWN'\n"
TRACE = [{"zone": 0, "act": 1, "x": 4268, "x_velocity": 0.0, "action": "RIGHT,DOWN"}]


def make_mutator(proposal_raw):
    """A MutatorClient whose model calls are stubbed: the structured proposal
    returns ``proposal_raw`` and any free-form rewrite is recorded."""

    class StubMutator(MutatorClient):
        def __init__(self):
            self.proposal_calls = 0
            self.freeform_calls = 0

        def _request_guard_proposal(self, prompt, image_path, temperature):
            self.proposal_calls += 1
            return proposal_raw

        def _call_macro_model(self, *args, **kwargs):
            self.freeform_calls += 1
            return "def get_action(state):\n    return 'LEFT'", "freeform"

        def _call_micro_model(self, *args, **kwargs):
            self.freeform_calls += 1
            return "def get_action(state):\n    return 'LEFT'", "freeform"

    return StubMutator()


class StructuredMutationTests(unittest.TestCase):
    FRONTIER = {"zone": 0, "act": 1, "x": 4268}

    def test_valid_proposal_becomes_preserving_guard_no_rewrite(self):
        mutator = make_mutator('{"action": "RIGHT,B", "hold_frames": 20}')
        with redirect_stdout(StringIO()):
            code, reasoning = mutator.mutate_policy(
                WORKING, "Sonic got stuck: no progress (zone 0 act 1)", "shot.png", [],
                coordinate_trace=TRACE, frontier=self.FRONTIER,
            )
        self.assertEqual(reasoning, "LLM structured guard")
        self.assertIn("# LLM_GUARD zone=0 act=1 x=4268", code)
        self.assertIn("return 'RIGHT,DOWN'", code)  # champion preserved, not rewritten
        self.assertEqual(mutator.proposal_calls, 1)
        self.assertEqual(mutator.freeform_calls, 0)  # never fell back to a rewrite

    def test_invalid_proposal_falls_back_to_freeform_rewrite(self):
        mutator = make_mutator("this is not json")
        with redirect_stdout(StringIO()):
            code, reasoning = mutator.mutate_policy(
                WORKING, "Sonic got stuck (zone 0 act 1)", "shot.png", [],
                coordinate_trace=TRACE, frontier=self.FRONTIER,
            )
        self.assertEqual(reasoning, "freeform")
        self.assertIn("LEFT", code)
        self.assertEqual(mutator.proposal_calls, 1)
        self.assertEqual(mutator.freeform_calls, 1)

    def test_no_explicit_frontier_skips_structured_path_entirely(self):
        # The structured path requires the orchestrator's AUTHORITATIVE frontier:
        # trace-tail coordinates sit at the respawn point after a death, and
        # stagnation-escape generations (which pass no frontier) must explore a
        # distinct strategy instead of being re-anchored at the plateau.
        mutator = make_mutator('{"action": "RIGHT,B", "hold_frames": 20}')
        with redirect_stdout(StringIO()):
            _, reasoning = mutator.mutate_policy(
                WORKING, "Stagnation plateau: try a distinct minimal strategy.",
                "shot.png", [], coordinate_trace=TRACE,
            )
        self.assertEqual(mutator.proposal_calls, 0)
        self.assertEqual(mutator.freeform_calls, 1)
        self.assertEqual(reasoning, "freeform")

    def test_explicit_frontier_overrides_respawn_trace_tail(self):
        # After a death-then-respawn the trace tail is at the RESPAWN point
        # (x~332); the guard must target the authoritative frontier (x=4268).
        respawn_trace = [{"zone": 0, "act": 1, "x": 332, "x_velocity": 0.0, "action": "RIGHT"}]
        mutator = make_mutator('{"action": "RIGHT,B", "hold_frames": 25}')
        with redirect_stdout(StringIO()):
            code, reasoning = mutator.mutate_policy(
                WORKING,
                "Sonic lost a life at the frontier (zone 0 act 1, x=4268) and respawned behind it.",
                "shot.png",
                [],
                coordinate_trace=respawn_trace,
                frontier={"zone": 0, "act": 1, "x": 4268},
            )
        self.assertEqual(reasoning, "LLM structured guard")
        self.assertIn("# LLM_GUARD zone=0 act=1 x=4268", code)  # death spot
        self.assertNotIn("x=332", code)                          # not the respawn point

    def test_code_timeout_skips_structured_path(self):
        # A code fault (timeout) is not a geometry problem; go straight to the
        # code model rewrite.
        mutator = make_mutator('{"action": "RIGHT,B", "hold_frames": 20}')
        with redirect_stdout(StringIO()):
            mutator.mutate_policy(
                WORKING, "Policy code timeout (infinite loop).", "shot.png", [],
                coordinate_trace=TRACE, frontier=self.FRONTIER,
            )
        self.assertEqual(mutator.proposal_calls, 0)
        self.assertEqual(mutator.freeform_calls, 1)


if __name__ == "__main__":
    unittest.main()
