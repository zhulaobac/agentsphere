"""Working code-check evaluator: syntax, registered predicates, and a restricted expression language."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from agentsphere.evaluation.types import EvalResult, EvalSample

CheckFn = Callable[[EvalSample], bool]

_ALLOWED_NODES = (
    ast.Expression,
    ast.Compare,
    ast.BoolOp,
    ast.UnaryOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Call,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


class CodeCheckEvaluator:
    name = "code-check"
    version = "0.1.0"
    kind = "code"

    def __init__(
        self,
        checks: dict[str, CheckFn] | None = None,
        *,
        name: str | None = None,
        version: str | None = None,
    ) -> None:
        self.checks = dict(checks or {})
        self.checks.setdefault("syntax", _syntax_ok)
        if name:
            self.name = name
        if version:
            self.version = version

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        names = list(sample.case.metadata.get("code_checks") or ["syntax"])
        expressions = list(sample.case.metadata.get("code_expressions") or [])
        failures: list[str] = []
        for name in names:
            fn = self.checks.get(name)
            if fn is None:
                failures.append(f"unknown check {name}")
                continue
            try:
                if not fn(sample):
                    failures.append(f"check {name} failed")
            except Exception as exc:  # noqa: BLE001 — evaluator must not crash the experiment
                failures.append(f"check {name} error: {exc}")
        context = {
            "output": sample.output,
            "expected": sample.case.expected,
            "metadata": sample.case.metadata,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
        }
        for expr in expressions:
            try:
                if not _safe_eval(str(expr), context):
                    failures.append(f"expression failed: {expr}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"expression error {expr}: {exc}")
        passed = not failures
        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="ok" if passed else "; ".join(failures),
            details={"failures": failures},
            evaluator=self.name,
            evaluator_version=self.version,
        )


def _syntax_ok(sample: EvalSample) -> bool:
    source = sample.output or ""
    if not source.strip():
        return False
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _safe_eval(expr: str, context: dict[str, Any]) -> bool:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression node {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in {"len", "str", "int", "float", "bool"}:
                raise ValueError("only len/str/int/float/bool calls are allowed")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private names are not allowed")
    compiled = compile(tree, "<code-check>", "eval")
    return bool(eval(compiled, {"__builtins__": {}}, context))  # noqa: S307 — restricted AST
