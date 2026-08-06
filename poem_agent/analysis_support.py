"""兼容入口：分析支撑实现已迁移到 :mod:`poem_agent.evidence.support`。"""

from .evidence.support import (
    FORCE_FALLBACK_VERDICT,
    INSUFFICIENT_VERDICT,
    LEVELS,
    PARTIAL_VERDICT,
    SUFFICIENT_LIMITED_VERDICT,
    SUFFICIENT_VERDICT,
    AnalysisAssessmentProtocolError,
    SupportEvaluation,
    append_required_support_notice,
    evaluate_analysis_support,
    force_fallback_analysis_support,
    normalize_model_support_notices,
    required_support_notice,
    validate_analysis_assessment,
)

__all__ = [
    "FORCE_FALLBACK_VERDICT",
    "INSUFFICIENT_VERDICT",
    "LEVELS",
    "PARTIAL_VERDICT",
    "SUFFICIENT_LIMITED_VERDICT",
    "SUFFICIENT_VERDICT",
    "AnalysisAssessmentProtocolError",
    "SupportEvaluation",
    "append_required_support_notice",
    "evaluate_analysis_support",
    "force_fallback_analysis_support",
    "normalize_model_support_notices",
    "required_support_notice",
    "validate_analysis_assessment",
]
