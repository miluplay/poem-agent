"""稳定跨模块数据边界的静态类型合同。"""

from typing import NotRequired, TypedDict


class EvidenceItem(TypedDict):
    evidence_id: str
    text: str | None
    poem_id: str | None
    title: NotRequired[str]
    poem_number: NotRequired[int]
    citation: NotRequired[str]
    dangling: NotRequired[bool]
    reason: NotRequired[str]


class AgentDecision(TypedDict):
    thought: str
    action: str
    action_input: dict


class AnalysisSupport(TypedDict):
    level: str
    target_ids: list[int]
    verdict: str


class TargetDetailChecklistItem(TypedDict):
    target_id: int
    covered: bool
    activated_poem_ids: list[str]
    activatable_cached_poem_ids: list[str]
    visible_candidate_poem_ids: list[str]


class FinalResult(TypedDict):
    answer: str
    evidence: list[EvidenceItem]
    analysis_support: AnalysisSupport
    degraded: bool
    candidate_pool: NotRequired[dict | None]
