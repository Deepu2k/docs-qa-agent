"""Tools the agent can call: JSON schemas for the API, plus the Python that runs them."""
from __future__ import annotations

import ast
import operator
from datetime import date

from .retrieval import VectorIndex

TOOL_SCHEMAS = [
    {
        "name": "search_docs",
        "description": (
            "Semantic search over the Acme Robotics employee handbook (time off, benefits, "
            "expenses, travel, remote work, security, IT, on-call, incidents, offices and more). "
            "Returns the 4 most relevant chunks, each labeled with a chunk ID like "
            "[expense-policy#1]. Call it before answering any question about company policy. "
            "If results look off-topic, search again with different wording."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for, in plain words"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression exactly. Supports + - * / // % ** and parentheses. "
            "Use it for every calculation, however simple, instead of doing math in your head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '350 * 4'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_date",
        "description": "Return today's date as YYYY-MM-DD plus the weekday. Use for questions about 'today', deadlines or upcoming dates.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# ------------------------------------------------------------ search_docs

_index: VectorIndex | None = None
_index_kwargs: dict = {}
TOP_K = 4


def configure_index(**kwargs) -> None:
    """Change retrieval settings (embedder, chunk_size, overlap, title_prefix) before the first search."""
    global _index, _index_kwargs
    _index, _index_kwargs = None, kwargs


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex(**_index_kwargs)
    return _index


def search_docs(query: str) -> str:
    hits = get_index().search(query, k=TOP_K)
    if not hits:
        return "No matching chunks found. Try different wording."
    return "\n\n".join(f"[{c.id}] (from: {c.title}, similarity {s:.2f})\n{c.text}" for c, s in hits)


# ------------------------------------------------------------ calculate

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    """Evaluate only numbers and arithmetic operators. Never eval() model output."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


def calculate(expression: str) -> str:
    try:
        result = _eval(ast.parse(expression.replace(",", "").replace("$", ""), mode="eval"))
    except Exception as e:  # the error goes back to the model so it can fix its input
        return f"Error: {e}"
    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 6)
    return str(result)


# ------------------------------------------------------------ get_current_date

def get_current_date() -> str:
    d = date.today()
    return f"{d.isoformat()} ({d:%A})"


TOOL_FUNCS = {"search_docs": search_docs, "calculate": calculate, "get_current_date": get_current_date}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Run a tool by name. Returns (output, is_error)."""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"Error: unknown tool {name}", True
    try:
        out = func(**args)
    except Exception as e:
        return f"Error running {name}: {e}", True
    return out, out.startswith("Error")
