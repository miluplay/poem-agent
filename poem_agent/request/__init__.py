"""完整用户请求协议的稳定对外接口。"""

from .models import (
    ConsolidatedRequest,
    ConsolidatedRequestProtocolError,
    NormalizedTargetFields,
    RequestTarget,
    RequestTask,
    ResolvedRequest,
    ResolvedTarget,
    ResolvedTask,
    TARGET_FIELDS,
    resolve_consolidated_request,
)
from .normalization import (
    normalize_consolidated_request,
    normalize_target_fields,
)
from .rendering import render_consolidated_request, render_resolved_request
from .semantics import request_semantics_error


__all__ = [
    "ConsolidatedRequest",
    "ConsolidatedRequestProtocolError",
    "NormalizedTargetFields",
    "RequestTarget",
    "RequestTask",
    "ResolvedRequest",
    "ResolvedTarget",
    "ResolvedTask",
    "TARGET_FIELDS",
    "normalize_consolidated_request",
    "normalize_target_fields",
    "render_consolidated_request",
    "render_resolved_request",
    "request_semantics_error",
    "resolve_consolidated_request",
]
