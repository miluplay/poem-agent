"""最终回答的证据绑定、支撑评估与完整性保护。"""

from .citations import collect_evidence, list_dangling_citations
from .integrity import answer_integrity_fallback, is_answer_suspiciously_incomplete
from .support import evaluate_analysis_support

__all__ = [
    "answer_integrity_fallback",
    "collect_evidence",
    "evaluate_analysis_support",
    "is_answer_suspiciously_incomplete",
    "list_dangling_citations",
]

