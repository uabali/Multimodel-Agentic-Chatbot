"""
AST-based safe calculator (from Final-Project — no eval()).
"""

import ast
import operator

from langchain_core.tools import tool

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        return _SAFE_OPS[op](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        return _SAFE_OPS[op](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely (supports +, -, *, /, **, %, //).

    Args:
        expression: Math expression, e.g. "2 + 3 * 4", "2**10"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Calculation error: {e}"
