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


def _deny_import(name, globals=None, locals=None, fromlist=(), level=0):
    raise ImportError("Imports are not available in the restricted runtime.")


def _restricted_skills_module(skills_path):
    """Build the ``policies.skills`` module under the SAME restricted builtins
    as a policy.

    Previously a policy's ``import policies.skills`` was serviced by the real
    import system, so ``policies/skills.py`` executed with the FULL builtins
    module -- the AST allowlist was the only thing standing between generated
    skill code and ``open``/``eval``/``__import__``. Skills now run in the
    restricted namespace too, closing that asymmetry.
    """
    with open(skills_path, "r", encoding="utf-8") as f:
        source = f.read()
    validate_skills_source(source)

    skills_module = types.ModuleType("policies.skills")
    safe_builtins = dict(SAFE_POLICY_BUILTINS)
    safe_builtins["__import__"] = _deny_import  # skills may not import anything
    skills_module.__dict__.update(
        {
            "__builtins__": safe_builtins,
            "__file__": skills_path,
            "__package__": "policies",
        }
    )
    exec(compile(source, skills_path, "exec"), skills_module.__dict__)
    return skills_module


def _make_restricted_import(policies_pkg, skills_module):
    """A policy ``__import__`` that resolves only to the in-memory restricted
    ``policies`` / ``policies.skills`` objects -- never the real filesystem
    package, so the restricted namespace cannot be escaped via import."""

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("Policies may only import policies.skills.")
        if name == "policies":
            return policies_pkg
        if name == "policies.skills":
            if skills_module is None:
                raise ImportError("No skills library is available.")
            # `from policies.skills import x` wants the module; `import
            # policies.skills [as y]` wants the top package to attribute-walk.
            return skills_module if fromlist else policies_pkg
        raise ImportError("Policies may only import policies.skills.")

    return _restricted_import


def load_policy(filepath):
    """Load validated generated code with a restricted builtin namespace."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    validate_policy_source(source)

    skills_path = os.path.join("policies", "skills.py")
    skills_module = (
        _restricted_skills_module(skills_path) if os.path.exists(skills_path) else None
    )

    policies_pkg = types.ModuleType("policies")
    policies_pkg.__dict__["__path__"] = []  # mark as a package for attribute walks
    policies_pkg.skills = skills_module

    policy_module = types.ModuleType("current_policy")
    safe_builtins = dict(SAFE_POLICY_BUILTINS)
    safe_builtins["__import__"] = _make_restricted_import(policies_pkg, skills_module)
    policy_module.__dict__.update(
        {
            "__builtins__": safe_builtins,
            "__file__": filepath,
            "__package__": "",
        }
    )
    exec(compile(source, filepath, "exec"), policy_module.__dict__)
    return policy_module
