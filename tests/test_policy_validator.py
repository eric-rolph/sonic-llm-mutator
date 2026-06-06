import unittest

from core.policy_validator import PolicyValidationError, validate_policy_source


class PolicyValidatorTests(unittest.TestCase):
    def test_accepts_stateful_policy_and_skills_import(self):
        source = """
import policies.skills as skills

def helper(state):
    return state.get("x_pos", 0)

def get_action(state):
    global _STATE
    if "_STATE" not in globals():
        _STATE = {}
    return "RIGHT,B" if helper(state) > 10 else "RIGHT"
"""
        validate_policy_source(source)

    def test_rejects_policy_without_get_action(self):
        with self.assertRaisesRegex(PolicyValidationError, "get_action"):
            validate_policy_source("def helper(state):\n    return 'RIGHT'\n")

    def test_rejects_non_skills_import(self):
        with self.assertRaisesRegex(PolicyValidationError, "Imports"):
            validate_policy_source("import os\n\ndef get_action(state):\n    return 'RIGHT'\n")

    def test_rejects_non_skills_import_nested_inside_get_action(self):
        source = """
def get_action(state):
    import subprocess
    return "RIGHT"
"""
        with self.assertRaisesRegex(PolicyValidationError, "Imports"):
            validate_policy_source(source)

    def test_rejects_dangerous_builtin_call(self):
        source = """
def get_action(state):
    open("owned.txt", "w").write("bad")
    return "RIGHT"
"""
        with self.assertRaisesRegex(PolicyValidationError, "open"):
            validate_policy_source(source)

    def test_rejects_executable_top_level_statements(self):
        source = """
while True:
    pass

def get_action(state):
    return "RIGHT"
"""
        with self.assertRaisesRegex(PolicyValidationError, "top-level"):
            validate_policy_source(source)


if __name__ == "__main__":
    unittest.main()
