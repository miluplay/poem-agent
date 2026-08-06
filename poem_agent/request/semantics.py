"""用户原话与跨轮完整请求的语义守卫。"""

from __future__ import annotations

from .. import store
from .models import (
    ConsolidatedRequest,
    RequestTarget,
    ResolvedRequest,
    ResolvedTarget,
    _ANALYSIS_TASK_TYPES,
    _BOOK_TITLE_PATTERN,
    _DYNASTY_ALIASES,
    _EXPLICIT_ASPECT_SIGNALS,
    _EXPLICIT_TASK_SIGNALS,
)

def request_semantics_error(
    user_query: str,
    previous: ResolvedRequest | None,
    request: ConsolidatedRequest,
) -> str | None:
    """在提交 Session/Pool 前检查高置信度任务语义与 follow-up 继承。"""
    if not isinstance(user_query, str):
        raise TypeError("user_query 必须是字符串")
    if previous is not None and not isinstance(previous, ResolvedRequest):
        raise TypeError("previous 必须是 ResolvedRequest 或 None")
    if not isinstance(request, ConsolidatedRequest):
        raise TypeError("request 必须是 ConsolidatedRequest")

    target_error = _target_constraints_error(user_query, previous, request)
    if target_error is not None:
        return target_error

    explicit_type = _explicit_task_type(user_query)
    new_types = {task.type for task in request.tasks}
    if explicit_type is not None and explicit_type not in new_types:
        return (
            "任务语义守卫拒绝提交：当前用户原话明确要求 "
            f"task type={explicit_type}，新完整请求却只有 "
            f"{sorted(new_types)}。tasks 表示最终用户任务，不能用 search "
            "代替赏析或对比。请重交完整 targets + tasks。"
        )

    explicit_aspects = {
        aspect
        for aspect, signals in _EXPLICIT_ASPECT_SIGNALS
        if any(signal in user_query for signal in signals)
    }
    if explicit_aspects:
        relevant_type = (
            explicit_type
            if explicit_type in _ANALYSIS_TASK_TYPES
            else None
        )
        relevant_tasks = [
            task
            for task in request.tasks
            if task.type in _ANALYSIS_TASK_TYPES
            and (relevant_type is None or task.type == relevant_type)
        ]
        covered = {
            aspect for task in relevant_tasks for aspect in task.aspects
        }
        missing = sorted(explicit_aspects - covered)
        if missing:
            return (
                "任务语义守卫拒绝提交：用户明确指定分析角度 "
                f"{missing}，对应 appreciate/compare task 必须在 aspects "
                "中保留这些值。请重交完整请求。"
            )

    if previous is None or explicit_type is not None:
        return None

    previous_primary = _primary_tasks(previous.tasks)
    for old_task in previous_primary:
        compatible = [
            task for task in request.tasks if task.type == old_task.type
        ]
        if not compatible:
            return (
                "任务语义守卫拒绝提交：当前 follow-up 没有明确切换最终任务，"
                f"必须继承 task type={old_task.type}。target 可以改变，但 tasks "
                "不能漂移为内部检索步骤。"
            )
        if old_task.type in _ANALYSIS_TASK_TYPES:
            match = any(
                set(old_task.aspects).issubset(task.aspects)
                and set(old_task.custom_aspects).issubset(task.custom_aspects)
                for task in compatible
            )
            if not match:
                return (
                    "任务语义守卫拒绝提交：模糊 follow-up 必须继承已确认的 "
                    f"{old_task.type} 分析角度 aspects={list(old_task.aspects)}、"
                    f"custom_aspects={list(old_task.custom_aspects)}。"
                )
        elif old_task.type in {"read", "verify"} and not any(
            set(old_task.fields).issubset(task.fields) for task in compatible
        ):
            return (
                "任务语义守卫拒绝提交：模糊 follow-up 必须继承 "
                f"{old_task.type} fields={list(old_task.fields)}。"
            )
    return None


def _target_constraints_error(
    user_query: str,
    previous: ResolvedRequest | None,
    request: ConsolidatedRequest,
) -> str | None:
    """拒绝模型把未确认的诗题或朝代补成新的检索硬条件。"""
    explicit_titles = {
        store._normalize_title(match) or match.strip()
        for match in _BOOK_TITLE_PATTERN.findall(user_query)
        if match.strip()
    }
    explicit_dynasties = {
        canonical
        for canonical, aliases in _DYNASTY_ALIASES.items()
        if any(alias in user_query for alias in aliases)
    }
    previous_targets = previous.targets if previous is not None else ()
    exact_previous = {
        (
            target.author,
            target.dynasty,
            target.title,
            target.themes,
        )
        for target in previous_targets
    }
    allow_partial_inheritance = (
        previous is not None
        and len(request.targets) <= len(previous_targets)
    )

    for index, target in enumerate(request.targets, start=1):
        identity = (
            target.author,
            target.dynasty,
            target.title,
            target.themes,
        )
        if identity in exact_previous:
            continue
        source = (
            _partially_inherited_target(
                user_query,
                explicit_titles,
                explicit_dynasties,
                target,
                previous_targets,
            )
            if allow_partial_inheritance
            else None
        )

        title_inherited = (
            source is not None and target.title == source.title
        )
        if (
            target.title is not None
            and target.title not in explicit_titles
            and target.title not in user_query
            and not title_inherited
        ):
            return (
                "target 语义守卫拒绝提交："
                f"targets[{index}].title={target.title!r} 是用户未确认的新增"
                "硬条件。用户原话未明确指定该诗题；请将 title 设为 null，"
                "或仅原样继承当前 resolved request 中对应旧 target 的 title。"
            )

        dynasty_inherited = (
            source is not None and target.dynasty == source.dynasty
        )
        if (
            target.dynasty is not None
            and target.dynasty not in explicit_dynasties
            and not dynasty_inherited
        ):
            return (
                "target 语义守卫拒绝提交："
                f"targets[{index}].dynasty={target.dynasty!r} 是用户未确认的"
                "新增硬条件。请将 dynasty 设为 null，或原样继承当前 "
                "resolved request 中对应旧 target 的 dynasty。"
            )
    return None


def _partially_inherited_target(
    user_query: str,
    explicit_titles: set[str],
    explicit_dynasties: set[str],
    target: RequestTarget,
    previous_targets: tuple[ResolvedTarget, ...],
) -> ResolvedTarget | None:
    """只在用户明确表达所有变化时，把其余字段视为旧 target 原样继承。"""
    for source in previous_targets:
        if (
            target.author != source.author
            and not (
                target.author is not None
                and target.author in user_query
            )
        ):
            continue
        if (
            target.title != source.title
            and target.title not in explicit_titles
            and not (
                target.title is not None
                and target.title in user_query
            )
        ):
            continue
        if (
            target.dynasty != source.dynasty
            and target.dynasty not in explicit_dynasties
        ):
            continue
        if target.themes != source.themes and not all(
            theme in user_query for theme in target.themes
        ):
            continue
        return source
    return None


def _explicit_task_type(user_query: str) -> str | None:
    for task_type, signals in _EXPLICIT_TASK_SIGNALS:
        if any(signal in user_query for signal in signals):
            return task_type
    return None


def _primary_tasks(tasks: tuple[ResolvedTask, ...]) -> tuple[ResolvedTask, ...]:
    """取现有请求中最高语义优先级的任务；并列任务一并继承。"""
    priority = {
        "compare": 0,
        "appreciate": 1,
        "read": 2,
        "verify": 3,
        "search": 4,
    }
    best = min(priority[task.type] for task in tasks)
    return tuple(task for task in tasks if priority[task.type] == best)


