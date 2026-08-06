"""Candidate Pool 的稳定兼容接口。"""

from .candidate.models import (
    FROZEN_TARGET_LIMIT,
    THEME_SEPARATOR,
    VISIBLE_CANDIDATE_LIMIT,
    CandidatePoolProtocolError,
    QueryTask,
    SearchFunction,
    Target,
    normalize_targets,
)
from .candidate.pool import CandidatePool
from .retrieval import retrieve_all_poems


__all__ = [
    "CandidatePool",
    "CandidatePoolProtocolError",
    "FROZEN_TARGET_LIMIT",
    "QueryTask",
    "SearchFunction",
    "THEME_SEPARATOR",
    "Target",
    "VISIBLE_CANDIDATE_LIMIT",
    "normalize_targets",
    "retrieve_all_poems",
]
