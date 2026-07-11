# safe_math.py
"""
Safe arithmetic evaluation for Tier 0.

Never uses eval(). Parses expressions via Python's `ast` module and only
permits a whitelist of numeric operators. Supports multi-operator
expressions with correct precedence (e.g. "2 + 3 * 4").

Also handles natural language math patterns:
  "9 squared", "3 cubed", "square root of 16", "sqrt(25)", etc.
"""

import ast
import math
import operator
import re
from typing import Optional

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CANDIDATE_PATTERN = re.compile(r'[\d\.\s\+\-\*\/\%\^\(\)]{3,}')
_MAX_EXPONENT = 1000
_MAX_RESULT_MAGNITUDE = 1e15


class SafeMathError(ValueError):
    pass


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeMathError("Non-numeric constant")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise SafeMathError(f"Operator not allowed: {op_type.__name__}")
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise SafeMathError("Exponent too large")
        if op_type in (ast.Div, ast.Mod, ast.FloorDiv) and right == 0:
            raise SafeMathError("Division by zero")
        result = _ALLOWED_BINOPS[op_type](left, right)
        if abs(result) > _MAX_RESULT_MAGNITUDE:
            raise SafeMathError("Result magnitude too large")
        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise SafeMathError(f"Unary operator not allowed: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand))

    raise SafeMathError(f"Disallowed expression node: {type(node).__name__}")


def safe_eval(expr: str) -> float:
    expr = expr.strip().replace("^", "**")
    if not expr:
        raise SafeMathError("Empty expression")
    if len(expr) > 200:
        raise SafeMathError("Expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeMathError(f"Invalid syntax: {e}")
    return _eval_node(tree.body)


# ── Natural language math patterns ─────────────────────────────────────────
# Each entry: (compiled_regex, handler_fn)
# handler_fn receives the re.Match and returns float or None

def _num(s: str) -> Optional[float]:
    """Parse a number string safely."""
    try:
        return float(s.replace(',', ''))
    except ValueError:
        return None

_NL_PATTERNS = [
    # "9 squared" / "9 to the power of 2" / "9^2"
    (re.compile(r'(\d+(?:\.\d+)?)\s+squared', re.I),
     lambda m: (n := _num(m.group(1))) and n ** 2),

    # "9 cubed"
    (re.compile(r'(\d+(?:\.\d+)?)\s+cubed', re.I),
     lambda m: (n := _num(m.group(1))) and n ** 3),

    # "square root of 16" / "sqrt of 16" / "sqrt(16)"
    (re.compile(r'(?:square\s+root\s+of\s+|sqrt\s+of\s+|sqrt\s*\(?\s*)(\d+(?:\.\d+)?)\)?', re.I),
     lambda m: (n := _num(m.group(1))) is not None and math.sqrt(n)),

    # "cube root of 27"
    (re.compile(r'cube\s+root\s+of\s+(\d+(?:\.\d+)?)', re.I),
     lambda m: (n := _num(m.group(1))) is not None and n ** (1/3)),

    # "9 to the power of 3" / "9 to the 3rd power" / "9 to the 3"
    (re.compile(r'(\d+(?:\.\d+)?)\s+to\s+the\s+(?:power\s+of\s+)?(\d+(?:\.\d+)?)', re.I),
     lambda m: (b := _num(m.group(1))) is not None and (e := _num(m.group(2))) is not None and b ** e),

    # "factorial of 5" / "5 factorial" / "5!"
    (re.compile(r'(?:factorial\s+of\s+(\d+)|(\d+)\s+factorial|(\d+)!)', re.I),
     lambda m: (n := _num(next(x for x in m.groups() if x))) is not None and float(math.factorial(int(n)))),

    # "10% of 200" / "15 percent of 80"
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)', re.I),
     lambda m: (a := _num(m.group(1))) is not None and (b := _num(m.group(2))) is not None and (a / 100) * b),

    # "half of 80" / "a third of 90"
    (re.compile(r'half\s+of\s+(\d+(?:\.\d+)?)', re.I),
     lambda m: (n := _num(m.group(1))) is not None and n / 2),
]


def _try_natural_language(text: str) -> Optional[float]:
    """
    Try each natural language math pattern in order.
    Returns the result of the first match, or None.
    """
    for pattern, handler in _NL_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                result = handler(m)
                if result is not False and result is not None:
                    if abs(float(result)) <= _MAX_RESULT_MAGNITUDE:
                        return float(result)
            except (ValueError, OverflowError, TypeError):
                continue
    return None


def _format_result(result: float) -> str:
    """Format a float result cleanly — no unnecessary decimals."""
    if result == int(result) and abs(result) < 1e12:
        return str(int(result))
    return f"{result:.6g}"


def extract_and_evaluate(text: str) -> Optional[float]:
    """
    Try natural language patterns first (squared, cubed, sqrt, etc.),
    then fall back to symbol-based arithmetic scanning.
    Returns None if no safely-evaluable expression is found.
    """
    # Natural language first — higher precision for word-based queries
    nl_result = _try_natural_language(text)
    if nl_result is not None:
        return nl_result

    # Symbol-based fallback
    candidates = _CANDIDATE_PATTERN.findall(text)
    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        candidate = candidate.strip().rstrip(".")
        if not candidate or not any(c.isdigit() for c in candidate):
            continue
        # Require at least one operator — bare numbers like "240" should NOT
        # be returned as answers since they're just quantities mentioned in the
        # problem text, not the answer itself.
        has_operator = any(op in candidate for op in ['+', '-', '*', '/', '%', '^', '**'])
        if not has_operator:
            continue
        try:
            return safe_eval(candidate)
        except SafeMathError:
            continue
    return None