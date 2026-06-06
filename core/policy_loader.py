import builtins
import os
import types

from core.policy_validator import (
    SAFE_POLICY_BUILTIN_NAMES,
    validate_policy_source,
    validate_skills_source,
)

SAFE_POLICY_BUILTINS = {
    name: getattr(builtins, name)
    for name in SAFE_POLICY_BUILTIN_NAMES
}


def _safe_policy_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name not in {"policies", "policies.skills"}:
        raise ImportError("Policies may only import policies.skills.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def load_policy(filepath):
    """Load validated generated code with a restricted builtin namespace."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    validate_policy_source(source)

    skills_path = os.path.join("policies", "skills.py")
    if os.path.exists(skills_path):
        with open(skills_path, "r", encoding="utf-8") as f:
            validate_skills_source(f.read())

    policy_module = types.ModuleType("current_policy")
    safe_builtins = dict(SAFE_POLICY_BUILTINS)
    safe_builtins["__import__"] = _safe_policy_import
    policy_module.__dict__.update(
        {
            "__builtins__": safe_builtins,
            "__file__": filepath,
            "__package__": "",
        }
    )
    exec(compile(source, filepath, "exec"), policy_module.__dict__)
    return policy_module
