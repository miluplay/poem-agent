"""verbose 模式的 CLI 展示函数。"""

from __future__ import annotations

import json


def _format_action(action: str, action_input: dict) -> str:
    if action == "finish" and "answer" in action_input:
        action_input = {**action_input, "answer": "（见最终作答）"}
    args = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in action_input.items()
    )
    return f"{action}({args})"


def _print_step_decision(step: int, decision: dict) -> None:
    thought = decision.get("thought") or "（无）"
    print(f"\n[步骤 {step}] 💭 {thought}")
    print(f"          🔧 {_format_action(decision['action'], decision['action_input'])}")


def _print_step_error(step: int, error: str) -> None:
    print(f"\n[步骤 {step}] 💭 （模型返回格式非法）")
    _print_observation(f"错误：{error}")


def _print_observation(summary: str) -> None:
    print(f"          👀 {summary}")


def _print_final_separator() -> None:
    print("\n───────── 最终作答 ─────────")
