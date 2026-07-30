"""完整用户请求的结构化协议、规范化和确定性渲染。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable

from . import store


TARGET_FIELDS = frozenset({"author", "dynasty", "title", "themes"})
_REQUEST_FIELDS = frozenset({"targets", "tasks"})
_TARGET_REF_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
_TASK_TYPES = frozenset({"search", "read", "appreciate", "compare", "verify"})
_READ_FIELDS = frozenset({"content", "annotations"})
_VERIFY_FIELDS = frozenset({"author", "dynasty", "title"})
_ASPECTS = frozenset(
    {"theme", "emotion", "imagery", "technique", "structure", "diction"}
)
_ASPECT_LABELS = {
    "theme": "主题",
    "emotion": "情感",
    "imagery": "意象",
    "technique": "手法",
    "structure": "结构",
    "diction": "字词",
}
_FIELD_LABELS = {
    "content": "原文",
    "annotations": "注释",
    "author": "作者",
    "dynasty": "朝代",
    "title": "标题",
}
_EXPLICIT_TASK_SIGNALS = (
    ("compare", ("对比", "比较", "异同", "区别")),
    (
        "appreciate",
        ("赏析", "分析", "解读", "意象", "情感", "手法", "结构", "字词", "炼字"),
    ),
    ("read", ("原文", "全文", "注释", "读一下")),
    ("verify", ("核对", "验证", "是不是", "是否", "谁写", "作者", "朝代")),
    ("search", ("查找", "找", "列举", "有哪些", "推荐")),
)
_EXPLICIT_ASPECT_SIGNALS = (
    ("theme", ("主题",)),
    ("emotion", ("情感",)),
    ("imagery", ("意象",)),
    ("technique", ("手法",)),
    ("structure", ("结构",)),
    ("diction", ("字词", "炼字")),
)
_ANALYSIS_TASK_TYPES = frozenset({"appreciate", "compare"})
_BOOK_TITLE_PATTERN = re.compile(r"《([^《》]+)》")
_DYNASTY_ALIASES = {
    "先秦": ("先秦",),
    "两汉": ("两汉", "汉代", "汉朝"),
    "魏晋": ("魏晋",),
    "南北朝": ("南北朝",),
    "隋代": ("隋代", "隋朝"),
    "唐代": ("唐代", "唐朝"),
    "五代": ("五代",),
    "宋代": ("宋代", "宋朝"),
    "金朝": ("金朝", "金代"),
    "元代": ("元代", "元朝"),
    "明代": ("明代", "明朝"),
    "清代": ("清代", "清朝"),
    "近代": ("近代",),
    "现代": ("现代",),
}


class ConsolidatedRequestProtocolError(ValueError):
    """模型提交的内容不符合完整总请求协议。"""


@dataclass(frozen=True)
class NormalizedTargetFields:
    """Candidate Pool 与总请求共用的 target 字段规范化结果。"""

    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def identity(self) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
        return self.author, self.dynasty, self.title, self.themes


@dataclass(frozen=True)
class RequestTarget:
    target_ref: str
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "themes", tuple(self.themes))

    def snapshot(self) -> dict:
        return {
            "target_ref": self.target_ref,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
        }


@dataclass(frozen=True)
class RequestTask:
    type: str
    target_refs: tuple[str, ...]
    fields: tuple[str, ...] = ()
    aspects: tuple[str, ...] = ()
    custom_aspects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_refs", tuple(self.target_refs))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "aspects", tuple(self.aspects))
        object.__setattr__(self, "custom_aspects", tuple(self.custom_aspects))

    def snapshot(self) -> dict:
        value: dict[str, object] = {
            "type": self.type,
            "target_refs": list(self.target_refs),
        }
        if self.type in {"read", "verify"}:
            value["fields"] = list(self.fields)
        elif self.type in {"appreciate", "compare"}:
            value["aspects"] = list(self.aspects)
            value["custom_aspects"] = list(self.custom_aspects)
        return value


@dataclass(frozen=True)
class ConsolidatedRequest:
    targets: tuple[RequestTarget, ...]
    tasks: tuple[RequestTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "tasks", tuple(self.tasks))

    def snapshot(self) -> dict:
        return {
            "targets": [target.snapshot() for target in self.targets],
            "tasks": [task.snapshot() for task in self.tasks],
        }


@dataclass(frozen=True)
class ResolvedTarget:
    """用会话级稳定 ID 表达的 canonical target。"""

    target_id: int
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "themes", tuple(self.themes))

    def snapshot(self) -> dict:
        return {
            "target_id": self.target_id,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
        }


@dataclass(frozen=True)
class ResolvedTask:
    """临时 target_ref 已被替换为持久 target_id 的任务。"""

    type: str
    target_ids: tuple[int, ...]
    fields: tuple[str, ...] = ()
    aspects: tuple[str, ...] = ()
    custom_aspects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", tuple(self.target_ids))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "aspects", tuple(self.aspects))
        object.__setattr__(self, "custom_aspects", tuple(self.custom_aspects))

    def snapshot(self) -> dict:
        value: dict[str, object] = {
            "type": self.type,
            "target_ids": list(self.target_ids),
        }
        if self.type in {"read", "verify"}:
            value["fields"] = list(self.fields)
        elif self.type in {"appreciate", "compare"}:
            value["aspects"] = list(self.aspects)
            value["custom_aspects"] = list(self.custom_aspects)
        return value


@dataclass(frozen=True)
class ResolvedRequest:
    """可由 AgentSession 长期持有的完整请求快照。"""

    targets: tuple[ResolvedTarget, ...]
    tasks: tuple[ResolvedTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "tasks", tuple(self.tasks))

    def snapshot(self) -> dict:
        return {
            "targets": [target.snapshot() for target in self.targets],
            "tasks": [task.snapshot() for task in self.tasks],
        }


def resolve_consolidated_request(
    request: ConsolidatedRequest,
    target_ids_by_ref: dict[str, int],
) -> ResolvedRequest:
    """把一次提交内的 refs 确定性映射为会话级 target IDs。"""
    if not isinstance(request, ConsolidatedRequest):
        raise TypeError("request 必须是 ConsolidatedRequest")
    expected_refs = {target.target_ref for target in request.targets}
    if set(target_ids_by_ref) != expected_refs:
        raise ValueError("target_ids_by_ref 必须精确覆盖请求中的 canonical refs")
    targets = tuple(
        ResolvedTarget(
            target_ids_by_ref[target.target_ref],
            target.author,
            target.dynasty,
            target.title,
            target.themes,
        )
        for target in request.targets
    )
    tasks = tuple(
        ResolvedTask(
            task.type,
            tuple(target_ids_by_ref[target_ref] for target_ref in task.target_refs),
            fields=task.fields,
            aspects=task.aspects,
            custom_aspects=task.custom_aspects,
        )
        for task in request.tasks
    )
    return ResolvedRequest(targets, tasks)


ProtocolErrorFactory = Callable[[str], ValueError]


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
