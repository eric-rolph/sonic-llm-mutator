import ast
import builtins
import re

# A format replacement field whose name reaches into an attribute or index:
# "{0.attr}", "{x[key]}", "{.attr}". Plain "{}"/"{0}"/"{name}" are unaffected.
_FORMAT_FIELD_ACCESS = re.compile(r"\{[^{}]*[.\[][^{}]*\}")

SAFE_POLICY_BUILTIN_NAMES = {
    "ArithmeticError",
    "Exception",
    "IndexError",
    "KeyError",
    "TypeError",
    "ValueError",
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "globals",
    "hasattr",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_ALL_BUILTIN_NAMES = set(dir(builtins))


class PolicyValidationError(ValueError):
    """Raised when generated policy source violates the policy contract."""


_DANGEROUS_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "input",
    "iter",
    "locals",
    "open",
    "setattr",
    "vars",
}


def _function_arguments(node):
    return node.args.posonlyargs + node.args.args + node.args.kwonlyargs


def _validate_function_definition(node):
    if isinstance(node, ast.AsyncFunctionDef):
        raise PolicyValidationError("Async functions are not allowed.")
    if node.decorator_list:
        raise PolicyValidationError("Function decorators are not allowed.")
    if node.args.defaults or any(default is not None for default in node.args.kw_defaults):
        raise PolicyValidationError("Function default arguments are not allowed.")
    if node.returns is not None or any(arg.annotation is not None for arg in _function_arguments(node)):
        raise PolicyValidationError("Function annotations are not allowed.")
    if node.args.vararg is not None and node.args.vararg.annotation is not None:
        raise PolicyValidationError("Function annotations are not allowed.")
    if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
        raise PolicyValidationError("Function annotations are not allowed.")


def _validate_get_action_signature(node):
    positional = node.args.posonlyargs + node.args.args
    has_exact_signature = (
        len(positional) == 1
        and positional[0].arg == "state"
        and not node.args.kwonlyargs
        and node.args.vararg is None
        and node.args.kwarg is None
    )
    if not has_exact_signature:
        raise PolicyValidationError(
            "Policy get_action must accept exactly one required positional state argument."
        )


def _validate_import(node):
    if isinstance(node, ast.Import):
        allowed = all(alias.name == "policies.skills" for alias in node.names)
    else:
        allowed = node.module in {"policies", "policies.skills"} and all(
            alias.name == "skills" for alias in node.names
        )
    if not allowed:
        raise PolicyValidationError("Imports are restricted to policies.skills.")


def _validate_source(source, require_get_action, allow_skills_import):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise PolicyValidationError(f"Policy syntax error: {e}") from e

    get_actions = []
    for node in tree.body:
        allowed_top_level = isinstance(node, ast.FunctionDef) or (
            allow_skills_import and isinstance(node, (ast.Import, ast.ImportFrom))
        )
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not allowed_top_level and not is_docstring:
            raise PolicyValidationError(
                "Policy may not execute statements at top-level; keep logic inside functions."
            )
        if isinstance(node, ast.FunctionDef) and node.name == "get_action":
            get_actions.append(node)
        if (
            not require_get_action
            and isinstance(node, ast.FunctionDef)
            and node.name in _ALL_BUILTIN_NAMES
        ):
            raise PolicyValidationError(
                f"Generated skills may not shadow builtin {node.name}."
            )

    if require_get_action and not get_actions:
        raise PolicyValidationError("Policy must define a top-level get_action(state) function.")
    if require_get_action and len(get_actions) > 1:
        raise PolicyValidationError("Policy must define exactly one top-level get_action function.")
    if require_get_action:
        _validate_get_action_signature(get_actions[0])

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not allow_skills_import:
                raise PolicyValidationError("Generated skills may not import modules.")
            _validate_import(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _validate_function_definition(node)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PolicyValidationError("Policy may not access dunder attributes.")
        if isinstance(node, ast.While):
            raise PolicyValidationError("Policy may not use while loops in per-frame actions.")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("__")
            and node.value.endswith("__")
        ):
            raise PolicyValidationError("Policy may not reference dunder names.")
        # str.format field access ("{0.__class__}".format(x)) walks attributes
        # without an ast.Attribute node, sidestepping the dunder-attribute check
        # above -- a namespace-disclosure gadget. Generated code never needs
        # format-field templates, so reject any "{ <field> . | [ }" form.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _FORMAT_FIELD_ACCESS.search(node.value)
        ):
            raise PolicyValidationError(
                "Policy may not use str.format field access ('{0.attr}'/'{0[key]}')."
            )
        if isinstance(node, ast.Name):
            if not require_get_action and node.id == "globals":
                raise PolicyValidationError("Generated skills may not reference globals.")
            if node.id in _DANGEROUS_CALLS:
                raise PolicyValidationError(
                    f"Policy may not reference dangerous builtin {node.id}."
                )
            if node.id == "__builtins__":
                raise PolicyValidationError("Policy may not access __builtins__.")
            if node.id in _ALL_BUILTIN_NAMES and node.id not in SAFE_POLICY_BUILTIN_NAMES:
                raise PolicyValidationError(
                    f"Builtin {node.id} is unavailable in the restricted runtime."
                )

    return tree


def validate_policy_source(source):
    """Validate generated policy syntax and its small trusted execution contract."""
    return _validate_source(source, require_get_action=True, allow_skills_import=True)


def validate_skills_source(source):
    """Validate a generated standalone skills library before writing or importing it."""
    return _validate_source(source, require_get_action=False, allow_skills_import=False)
