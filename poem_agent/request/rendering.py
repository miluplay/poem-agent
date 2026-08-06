"""完整请求的确定性中文渲染。"""

from __future__ import annotations

from .models import (
    ConsolidatedRequest,
    RequestTarget,
    ResolvedRequest,
    ResolvedTarget,
    _ASPECT_LABELS,
    _FIELD_LABELS,
)

def render_consolidated_request(request: ConsolidatedRequest) -> str:
    if not isinstance(request, ConsolidatedRequest):
        raise TypeError("request 必须是 ConsolidatedRequest")
    targets = {target.target_ref: target for target in request.targets}
    sentences: list[str] = []
    for task in request.tasks:
        descriptions = [
            _render_target(targets[target_ref])
            for target_ref in task.target_refs
        ]
        objects = _join_chinese(descriptions)
        if task.type == "search":
            sentence = f"查找{objects}"
        elif task.type == "read":
            sentence = (
                f"读取{objects}的"
                f"{_join_chinese([_FIELD_LABELS[item] for item in task.fields])}"
            )
        elif task.type in {"appreciate", "compare"}:
            verb = "赏析" if task.type == "appreciate" else "对比"
            angles = [
                *(_ASPECT_LABELS[item] for item in task.aspects),
                *task.custom_aspects,
            ]
            sentence = f"{verb}{objects}"
            if angles:
                sentence += f"的{_join_chinese(angles)}"
        else:
            sentence = (
                f"核对{objects}的"
                f"{_join_chinese([_FIELD_LABELS[item] for item in task.fields])}"
            )
        sentences.append(f"{sentence}。")
    return "".join(sentences)


def render_resolved_request(request: ResolvedRequest) -> str:
    """用稳定 target_id 确定性渲染会话当前完整请求。"""
    if not isinstance(request, ResolvedRequest):
        raise TypeError("request 必须是 ResolvedRequest")
    targets = {target.target_id: target for target in request.targets}
    sentences: list[str] = []
    for task in request.tasks:
        descriptions = [
            _render_target(targets[target_id])
            for target_id in task.target_ids
        ]
        objects = _join_chinese(descriptions)
        if task.type == "search":
            sentence = f"查找{objects}"
        elif task.type == "read":
            sentence = (
                f"读取{objects}的"
                f"{_join_chinese([_FIELD_LABELS[item] for item in task.fields])}"
            )
        elif task.type in {"appreciate", "compare"}:
            verb = "赏析" if task.type == "appreciate" else "对比"
            angles = [
                *(_ASPECT_LABELS[item] for item in task.aspects),
                *task.custom_aspects,
            ]
            sentence = f"{verb}{objects}"
            if angles:
                sentence += f"的{_join_chinese(angles)}"
        else:
            sentence = (
                f"核对{objects}的"
                f"{_join_chinese([_FIELD_LABELS[item] for item in task.fields])}"
            )
        target_ids = "、".join(str(item) for item in task.target_ids)
        sentences.append(f"{sentence}（target_id：{target_ids}）。")
    return "".join(sentences)


def _render_target(target: RequestTarget | ResolvedTarget) -> str:
    owner = "·".join(
        value
        for value in (target.dynasty, target.author)
        if value is not None
    )
    if target.title is not None:
        description = f"{owner}的《{target.title}》" if owner else f"《{target.title}》"
    elif owner:
        description = f"{owner}的作品"
    else:
        description = "作品"
    if target.themes:
        description += f"（主题：{'、'.join(target.themes)}）"
    return description


def _join_chinese(values: list[str]) -> str:
    if len(values) <= 1:
        return "".join(values)
    return "、".join(values[:-1]) + f"与{values[-1]}"

