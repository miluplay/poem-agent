"""Agent 的稳定对外接口。"""

from .decisions import normalize_detail_action_input, parse_decision
from .detail_policy import build_target_detail_checklist
from .finalization import force_finish
from .observation import _summarize_observation
from .prompts import SYSTEM_INSTRUCTION
from .runner import run_agent
from ..tools import TOOLS


def _assign_session_poem(session_poems: dict[int, str], poem_id: str) -> int:
    """兼容旧调用方的会话编号辅助函数；新代码使用 AgentSession。"""
    for poem_number, known_poem_id in session_poems.items():
        if known_poem_id == poem_id:
            return poem_number
    poem_number = len(session_poems) + 1
    session_poems[poem_number] = poem_id
    return poem_number


__all__ = [
    "SYSTEM_INSTRUCTION",
    "TOOLS",
    "build_target_detail_checklist",
    "force_finish",
    "normalize_detail_action_input",
    "parse_decision",
    "run_agent",
]
