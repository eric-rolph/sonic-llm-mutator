import ast


class PolicyValidationError(ValueError):
    """Raised when generated policy source violates the policy contract."""


_DANGEROUS_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}


def _validate_import(node):
    if isinstance(node, ast.Import):
        allowed = all(alias.name == "policies.skills" for alias in node.names)
    else:
        allowed = node.module in {"policies", "policies.skills"} and all(
            alias.name == "skills" for alias in node.names
        )
    if not allowed:
        raise PolicyValidationError("Imports are restricted to policies.skills.")


def validate_policy_source(source):
    """Validate generated policy syntax and its small trusted execution contract."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise PolicyValidationError(f"Policy syntax error: {e}") from e

    get_action = None
    for node in tree.body:
        allowed_top_level = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not allowed_top_level and not is_docstring:
            raise PolicyValidationError(
                "Policy may not execute statements at top-level; keep logic inside functions."
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_action":
            get_action = node

    if get_action is None:
        raise PolicyValidationError("Policy must define a top-level get_action(state) function.")
    if not get_action.args.args:
        raise PolicyValidationError("Policy get_action must accept a state argument.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_CALLS:
                raise PolicyValidationError(f"Policy may not call dangerous builtin {node.func.id}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PolicyValidationError("Policy may not access dunder attributes.")
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            raise PolicyValidationError("Policy may not access __builtins__.")

    return tree
