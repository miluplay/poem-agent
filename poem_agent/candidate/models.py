"""Candidate Pool 的领域模型与 legacy target 规范化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..request import normalize_target_fields


THEME_SEPARATOR = "；"
VISIBLE_CANDIDATE_LIMIT = 5
FROZEN_TARGET_LIMIT = 4


class CandidatePoolProtocolError(ValueError):
    """模型提交的内容不符合 Candidate Pool 协议。"""


@dataclass(frozen=True)
class Target:
    target_id: int
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def snapshot(self) -> dict:
        return {
            "target_id": self.target_id,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
        }


@dataclass
class QueryTask:
    target_id: int
    kind: str
    query: dict
    status: str = "pending"
    candidate_ids: list[str] = field(default_factory=list)
    attempt_count: int = 0
    recovery_status: str | None = None


SearchFunction = Callable[..., list[dict]]


def _target_identity(
    target: Target,
) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
    return target.author, target.dynasty, target.title, target.themes


def normalize_targets(raw_targets) -> list[Target]:
    if not isinstance(raw_targets, list):
        raise CandidatePoolProtocolError("targets 必须是列表")
    normalized = []
    seen: set[tuple] = set()
    for index, raw in enumerate(raw_targets, start=1):
        fields = normalize_target_fields(
            raw,
            location=f"targets[{index}]",
            error_factory=CandidatePoolProtocolError,
            require_all_fields=False,
        )
        value = fields.identity()
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if not 1 <= len(normalized) <= 4:
        raise CandidatePoolProtocolError(
            "targets 规范化去重后必须有 1–4 个，不能静默截断"
        )
    return [
        Target(index, *values)
        for index, values in enumerate(normalized, start=1)
    ]

