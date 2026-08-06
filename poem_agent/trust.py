"""兼容入口：Evidence 已迁移到 :mod:`poem_agent.evidence`。"""

from .evidence.citations import (
    _append_degraded_notice,
    _strip_invalid_citation_markers,
    collect_evidence,
    list_dangling_citations,
)
from .evidence.integrity import (
    answer_integrity_fallback,
    answer_integrity_gate,
    is_answer_suspiciously_incomplete,
)

__all__ = [
    "answer_integrity_fallback",
    "answer_integrity_gate",
    "collect_evidence",
    "is_answer_suspiciously_incomplete",
    "list_dangling_citations",
]
