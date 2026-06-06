import unittest

from core import policy_validator
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

    def test_rejects_dangerous_builtin_reference_used_through_alias(self):
        source = """
def get_action(state):
    writer = open
    writer("owned.txt", "w").write("bad")
    return "RIGHT"
"""
        with self.assertRaisesRegex(PolicyValidationError, "open"):
            validate_policy_source(source)

    def test_rejects_globals_builtins_escape(self):
        source = """
def get_action(state):
    writer = globals()["__builtins__"]["open"]
    writer("owned.txt", "w").write("bad")
    return "RIGHT"
"""
        with self.assertRaises(PolicyValidationError):
            validate_policy_source(source)

    def test_rejects_dynamic_dunder_access(self):
        source = """
def get_action(state):
    return getattr(state, "__class__")
"""
        with self.assertRaises(PolicyValidationError):
            validate_policy_source(source)

    def test_rejects_while_loops_in_per_frame_policy(self):
        source = """
def get_action(state):
    while True:
        pass
"""
        with self.assertRaisesRegex(PolicyValidationError, "while"):
            validate_policy_source(source)

    def test_rejects_get_action_signatures_outside_contract(self):
        invalid_sources = {
            "wrong argument name": "def get_action(observation):\n    return 'RIGHT'\n",
            "default": "def get_action(state={}):\n    return 'RIGHT'\n",
            "decorator": "@staticmethod\ndef get_action(state):\n    return 'RIGHT'\n",
            "argument annotation": "def get_action(state: dict):\n    return 'RIGHT'\n",
            "return annotation": "def get_action(state) -> str:\n    return 'RIGHT'\n",
            "async": "async def get_action(state):\n    return 'RIGHT'\n",
            "extra positional": "def get_action(state, required):\n    return 'RIGHT'\n",
            "keyword-only": "def get_action(state, *, required):\n    return 'RIGHT'\n",
            "varargs": "def get_action(state, *args):\n    return 'RIGHT'\n",
            "kwargs": "def get_action(state, **kwargs):\n    return 'RIGHT'\n",
        }

        for label, source in invalid_sources.items():
            with self.subTest(label=label):
                with self.assertRaises(PolicyValidationError):
                    validate_policy_source(source)

    def test_rejects_definition_time_expressions_on_helper_functions(self):
        invalid_sources = {
            "default": """
def factory():
    return 1
def helper(value=factory()):
    return value
def get_action(state):
    return "RIGHT"
""",
            "decorator": """
def decorate(function):
    return function
@decorate
def helper(value):
    return value
def get_action(state):
    return "RIGHT"
""",
            "annotation": """
def annotation():
    return str
def helper(value: annotation()):
    return value
def get_action(state):
    return "RIGHT"
""",
        }

        for label, source in invalid_sources.items():
            with self.subTest(label=label):
                with self.assertRaises(PolicyValidationError):
                    validate_policy_source(source)

    def test_validates_safe_generated_skills_without_requiring_get_action(self):
        source = """
def jump_when_stuck(state, memory):
    return "RIGHT,B" if memory.get("stuck") else None
"""
        policy_validator.validate_skills_source(source)

    def test_rejects_unsafe_generated_skills(self):
        source = """
def write_file(state):
    writer = open
    return writer("owned.txt", "w")
"""
        with self.assertRaisesRegex(PolicyValidationError, "open"):
            policy_validator.validate_skills_source(source)

    def test_rejects_generated_skill_that_exposes_module_globals(self):
        source = """
def leak_runtime():
    return globals()
"""
        with self.assertRaisesRegex(PolicyValidationError, "globals"):
            policy_validator.validate_skills_source(source)

    def test_rejects_generated_skill_that_shadows_builtin(self):
        source = """
def abs(value):
    return 0
"""
        with self.assertRaisesRegex(PolicyValidationError, "builtin"):
            policy_validator.validate_skills_source(source)

    def test_rejects_builtin_not_available_in_restricted_runtime(self):
        source = """
def get_action(state):
    return next(filter(bool, []), "RIGHT")
"""
        with self.assertRaisesRegex(PolicyValidationError, "restricted runtime"):
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
