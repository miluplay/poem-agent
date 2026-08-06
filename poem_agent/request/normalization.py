"""完整请求的结构与字段规范化。"""

from __future__ import annotations

import unicodedata

from .. import store
from .models import (
    ConsolidatedRequest,
    ConsolidatedRequestProtocolError,
    NormalizedTargetFields,
    ProtocolErrorFactory,
    RequestTarget,
    RequestTask,
    TARGET_FIELDS,
    _ASPECTS,
    _READ_FIELDS,
    _REQUEST_FIELDS,
    _TARGET_REF_PATTERN,
    _TASK_TYPES,
    _VERIFY_FIELDS,
)

def normalize_target_fields(
    raw,
    *,
    location: str,
    error_factory: ProtocolErrorFactory,
    require_all_fields: bool,
) -> NormalizedTargetFields:
    """规范化两套协议共用的 author/dynasty/title/themes 字段。"""
    if not isinstance(raw, dict):
        raise error_factory(f"{location} 必须是对象")
    fields = set(raw)
    unknown = fields - TARGET_FIELDS
    if unknown:
        raise error_factory(
            f"{location} 包含未知字段: {'、'.join(sorted(unknown))}"
        )
    if require_all_fields:
        missing = TARGET_FIELDS - fields
        if missing:
            raise error_factory(
                f"{location} 缺少字段: {'、'.join(sorted(missing))}"
            )

    author = _optional_string(
        f"{location}.author", raw.get("author"), error_factory
    )
    dynasty = _optional_string(
        f"{location}.dynasty", raw.get("dynasty"), error_factory
    )
    raw_title = _optional_string(
        f"{location}.title", raw.get("title"), error_factory
    )
    title = (store._normalize_title(raw_title) or None) if raw_title else None
    themes = _normalize_string_list(
        raw.get("themes", []),
        location=f"{location}.themes",
        error_factory=error_factory,
        allowed=None,
        nonempty_items=True,
    )
    if author is None and dynasty is None and title is None and not themes:
        raise error_factory(
            f"{location} 至少需要 author、dynasty、title 或一个 theme"
        )
    return NormalizedTargetFields(author, dynasty, title, themes)


def normalize_consolidated_request(raw) -> ConsolidatedRequest:
    if not isinstance(raw, dict):
        raise ConsolidatedRequestProtocolError("总请求必须是对象")
    _require_exact_fields(raw, _REQUEST_FIELDS, "总请求")
    raw_targets = raw["targets"]
    raw_tasks = raw["tasks"]
    if not isinstance(raw_targets, list):
        raise ConsolidatedRequestProtocolError("targets 必须是列表")
    if not isinstance(raw_tasks, list):
        raise ConsolidatedRequestProtocolError("tasks 必须是列表")

    targets: list[RequestTarget] = []
    seen_refs: set[str] = set()
    identity_to_ref: dict[tuple, str] = {}
    ref_aliases: dict[str, str] = {}
    for index, item in enumerate(raw_targets, start=1):
        location = f"targets[{index}]"
        if not isinstance(item, dict):
            raise ConsolidatedRequestProtocolError(f"{location} 必须是对象")
        _require_exact_fields(
            item, TARGET_FIELDS | {"target_ref"}, location
        )
        target_ref = item["target_ref"]
        if not isinstance(target_ref, str) or not _TARGET_REF_PATTERN.fullmatch(
            target_ref
        ):
            raise ConsolidatedRequestProtocolError(
                f"{location}.target_ref 格式无效"
            )
        if target_ref in seen_refs:
            raise ConsolidatedRequestProtocolError(
                f"{location}.target_ref 在本次请求中必须唯一"
            )
        seen_refs.add(target_ref)
        normalized = normalize_target_fields(
            {name: item[name] for name in TARGET_FIELDS},
            location=location,
            error_factory=ConsolidatedRequestProtocolError,
            require_all_fields=True,
        )
        identity = normalized.identity()
        canonical_ref = identity_to_ref.get(identity)
        if canonical_ref is not None:
            ref_aliases[target_ref] = canonical_ref
            continue
        identity_to_ref[identity] = target_ref
        ref_aliases[target_ref] = target_ref
        targets.append(RequestTarget(target_ref, *identity))

    if not 1 <= len(targets) <= 6:
        raise ConsolidatedRequestProtocolError(
            "targets 规范化去重后必须有 1–6 个，不能静默截断"
        )

    tasks: list[RequestTask] = []
    seen_tasks: set[RequestTask] = set()
    for index, item in enumerate(raw_tasks, start=1):
        task = _normalize_task(item, index=index, aliases=ref_aliases)
        if task not in seen_tasks:
            seen_tasks.add(task)
            tasks.append(task)
    if not 1 <= len(tasks) <= 8:
        raise ConsolidatedRequestProtocolError(
            "tasks 规范化去重后必须有 1–8 个，不能静默截断"
        )

    used_refs = {
        target_ref for task in tasks for target_ref in task.target_refs
    }
    orphaned = [
        target.target_ref
        for target in targets
        if target.target_ref not in used_refs
    ]
    if orphaned:
        raise ConsolidatedRequestProtocolError(
            f"targets 中存在未被 task 引用的对象: {'、'.join(orphaned)}"
        )
    return ConsolidatedRequest(tuple(targets), tuple(tasks))
def _normalize_task(
    raw, *, index: int, aliases: dict[str, str]
) -> RequestTask:
    location = f"tasks[{index}]"
    if not isinstance(raw, dict):
        raise ConsolidatedRequestProtocolError(f"{location} 必须是对象")
    task_type = raw.get("type")
    if not isinstance(task_type, str) or task_type not in _TASK_TYPES:
        raise ConsolidatedRequestProtocolError(
            f"{location}.type 必须是 search/read/appreciate/compare/verify 之一"
        )
    fields = {
        "search": frozenset({"type", "target_refs"}),
        "read": frozenset({"type", "target_refs", "fields"}),
        "appreciate": frozenset(
            {"type", "target_refs", "aspects", "custom_aspects"}
        ),
        "compare": frozenset(
            {"type", "target_refs", "aspects", "custom_aspects"}
        ),
        "verify": frozenset({"type", "target_refs", "fields"}),
    }[task_type]
    _require_exact_fields(raw, fields, location)
    target_refs = _normalize_target_refs(
        raw["target_refs"], location=location, aliases=aliases
    )

    if task_type == "search":
        return RequestTask(task_type, target_refs)
    if task_type == "read":
        normalized_fields = _normalize_string_list(
            raw["fields"],
            location=f"{location}.fields",
            error_factory=ConsolidatedRequestProtocolError,
            allowed=_READ_FIELDS,
            nonempty_items=True,
        )
        if not normalized_fields:
            raise ConsolidatedRequestProtocolError(
                f"{location}.fields 必须是非空列表"
            )
        return RequestTask(task_type, target_refs, fields=normalized_fields)
    if task_type == "verify":
        normalized_fields = _normalize_string_list(
            raw["fields"],
            location=f"{location}.fields",
            error_factory=ConsolidatedRequestProtocolError,
            allowed=_VERIFY_FIELDS,
            nonempty_items=True,
        )
        if not normalized_fields:
            raise ConsolidatedRequestProtocolError(
                f"{location}.fields 必须是非空列表"
            )
        return RequestTask(task_type, target_refs, fields=normalized_fields)

    aspects = _normalize_string_list(
        raw["aspects"],
        location=f"{location}.aspects",
        error_factory=ConsolidatedRequestProtocolError,
        allowed=_ASPECTS,
        nonempty_items=True,
    )
    custom_aspects = _normalize_custom_aspects(
        raw["custom_aspects"], location=f"{location}.custom_aspects"
    )
    if task_type == "compare" and len(target_refs) < 2:
        raise ConsolidatedRequestProtocolError(
            f"{location}.target_refs 规范化后至少需要两个不同对象"
        )
    return RequestTask(
        task_type,
        target_refs,
        aspects=aspects,
        custom_aspects=custom_aspects,
    )


def _normalize_target_refs(
    raw, *, location: str, aliases: dict[str, str]
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConsolidatedRequestProtocolError(
            f"{location}.target_refs 必须是非空字符串列表"
        )
    refs: list[str] = []
    for ref_index, target_ref in enumerate(raw, start=1):
        if not isinstance(target_ref, str) or not target_ref.strip():
            raise ConsolidatedRequestProtocolError(
                f"{location}.target_refs[{ref_index}] 必须是非空字符串"
            )
        if target_ref not in aliases:
            raise ConsolidatedRequestProtocolError(
                f"{location}.target_refs[{ref_index}] 引用了未知 target_ref: "
                f"{target_ref}"
            )
        canonical_ref = aliases[target_ref]
        if canonical_ref not in refs:
            refs.append(canonical_ref)
    return tuple(refs)


def _normalize_custom_aspects(raw, *, location: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ConsolidatedRequestProtocolError(f"{location} 必须是字符串列表")
    values: list[str] = []
    for index, value in enumerate(raw, start=1):
        item_location = f"{location}[{index}]"
        if not isinstance(value, str):
            raise ConsolidatedRequestProtocolError(
                f"{item_location} 必须是字符串"
            )
        normalized = value.strip()
        if not 1 <= len(normalized) <= 20:
            raise ConsolidatedRequestProtocolError(
                f"{item_location} 去除两端空白后必须为 1–20 个字符"
            )
        if "\n" in normalized or "\r" in normalized or any(
            unicodedata.category(character).startswith("C")
            for character in normalized
        ):
            raise ConsolidatedRequestProtocolError(
                f"{item_location} 不允许换行或控制字符"
            )
        if normalized not in values:
            values.append(normalized)
    if len(values) > 2:
        raise ConsolidatedRequestProtocolError(
            f"{location} 规范化去重后最多 2 项"
        )
    return tuple(values)


def _normalize_string_list(
    raw,
    *,
    location: str,
    error_factory: ProtocolErrorFactory,
    allowed: frozenset[str] | None,
    nonempty_items: bool,
) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise error_factory(f"{location} 必须是字符串列表")
    values: list[str] = []
    for index, value in enumerate(raw, start=1):
        if not isinstance(value, str) or (nonempty_items and not value.strip()):
            raise error_factory(
                f"{location}[{index}] 必须是非空字符串"
            )
        normalized = value.strip()
        if allowed is not None and normalized not in allowed:
            raise error_factory(
                f"{location}[{index}] 包含不允许的值: {normalized}"
            )
        if normalized not in values:
            values.append(normalized)
    return tuple(values)


def _optional_string(
    location: str, value, error_factory: ProtocolErrorFactory
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise error_factory(f"{location} 必须是字符串或 None")
    return value.strip() or None


def _require_exact_fields(
    raw: dict, expected: frozenset[str], location: str
) -> None:
    unknown = set(raw) - expected
    if unknown:
        raise ConsolidatedRequestProtocolError(
            f"{location} 包含未知字段: {'、'.join(sorted(unknown))}"
        )
    missing = expected - set(raw)
    if missing:
        raise ConsolidatedRequestProtocolError(
            f"{location} 缺少字段: {'、'.join(sorted(missing))}"
        )


