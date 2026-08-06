"""模型决策与工具动作输入的协议解析。"""

from __future__ import annotations

import json
import re

from ..contracts import AgentDecision


_DETAIL_ACTION_FIELDS = frozenset(
    {"poem_id", "target_id", "target_ids", "fields"}
)


def normalize_detail_action_input(action_input) -> str:
    """只提取 canonical poem_id；已知上下文字段不参与任何授权。"""
    if not isinstance(action_input, dict):
        raise ValueError("action_input 必须是对象")
    unknown = set(action_input) - _DETAIL_ACTION_FIELDS
    if unknown:
        raise ValueError(
            "包含未知字段: " + "、".join(sorted(unknown))
        )
    if "poem_id" not in action_input:
        raise ValueError("缺少 poem_id")
    poem_id = action_input["poem_id"]
    if not isinstance(poem_id, str) or not poem_id.strip():
        raise ValueError("poem_id 必须是非空字符串")
    return poem_id.strip()


def parse_decision(raw: str) -> AgentDecision | None:
    """解析 LLM 输出为决策字典；格式错误由循环回填给模型修正。"""
    if not raw or not isinstance(raw, str):
        return None

    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    obj = _try_json(text)
    if obj is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = _try_json(text[start : end + 1])
    if not isinstance(obj, dict):
        return None

    action = obj.get("action")
    if not isinstance(action, str) or not action:
        return None
    action_input = obj.get("action_input", {})
    if not isinstance(action_input, dict):
        return None
    return {
        "thought": obj.get("thought", ""),
        "action": action,
        "action_input": action_input,
    }


def parse_json_object(raw: str) -> dict | None:
    """读取可能被 Markdown 或解释文字包裹的 JSON 对象。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    obj = _try_json(text)
    if isinstance(obj, dict):
        return obj
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        obj = _try_json(text[start : end + 1])
    return obj if isinstance(obj, dict) else None


def _try_json(value: str) -> dict | None:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
