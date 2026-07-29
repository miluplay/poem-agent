"""一次 Agent 运行内持有的 Candidate Pool 两阶段状态。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Callable

from . import store
from .retrieval import retrieve_all_poems


THEME_SEPARATOR = "；"
VISIBLE_CANDIDATE_LIMIT = 5
_TARGET_FIELDS = frozenset({"author", "dynasty", "title", "themes"})


class CandidatePoolProtocolError(ValueError):
    """模型提交的内容不符合 Candidate Pool 协议。"""


@dataclass(frozen=True)
class Target:
    target_id: int
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def snapshot(self) -> dict:
        return {
            "target_id": self.target_id,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
        }


@dataclass
class QueryTask:
    target_id: int
    kind: str
    query: dict
    status: str = "pending"
    candidate_ids: list[str] = field(default_factory=list)
    attempt_count: int = 0
    recovery_status: str | None = None


SearchFunction = Callable[..., list[dict]]


class CandidatePool:
    """保存筛选池、详情池、搜索画像和参考量画像。"""

    def __init__(
        self,
        targets: list[Target],
        *,
        search_fn: SearchFunction | None = None,
        visible_limit: int = VISIBLE_CANDIDATE_LIMIT,
        baseline: dict | None = None,
    ) -> None:
        self.targets = tuple(targets)
        self.visible_limit = visible_limit
        self._search_fn = search_fn or retrieve_all_poems
        self.main_tasks: dict[int, QueryTask] = {}
        self.diagnostic_tasks: dict[int, QueryTask] = {}
        self.candidates: dict[str, dict] = {}
        self.target_candidate_ids = {target.target_id: [] for target in targets}
        self.candidate_sources: dict[str, dict[int, set[str]]] = {}
        self.loaded_details: dict[str, dict] = {}
        self.failed_candidate_ids: set[str] = set()
        self.detail_unavailable_target_ids: set[int] = set()
        self._recovered_target_ids: set[int] = set()
        self.target_results: list[dict] = []
        self.profile: dict = {}
        self.verdict = ""
        self.reference_baseline = deepcopy(
            baseline if baseline is not None else store.reference_count_baseline()
        )
        self.reference_stats: dict = {}
        self.reference_verdict = ""
        self._compile_tasks()
        self._execute()
        self._refresh()

    @classmethod
    def initialize(
        cls,
        raw_targets,
        *,
        search_fn: SearchFunction | None = None,
        visible_limit: int = VISIBLE_CANDIDATE_LIMIT,
        baseline: dict | None = None,
    ) -> "CandidatePool":
        return cls(
            normalize_targets(raw_targets),
            search_fn=search_fn,
            visible_limit=visible_limit,
            baseline=baseline,
        )

    def _compile_tasks(self) -> None:
        for target in self.targets:
            self.main_tasks[target.target_id] = QueryTask(
                target.target_id, "main", _main_query(target)
            )
            diagnostic = _diagnostic_query(target)
            if diagnostic is not None:
                self.diagnostic_tasks[target.target_id] = QueryTask(
                    target.target_id, "diagnostic", diagnostic
                )

    def _execute(self) -> None:
        for target in self.targets:
            main_task = self.main_tasks[target.target_id]
            main_results = self._run_task(main_task)
            diagnostic_results: list[dict] = []
            diagnostic_task = self.diagnostic_tasks.get(target.target_id)
            if diagnostic_task is not None:
                if main_results:
                    diagnostic_task.status = "skipped"
                else:
                    diagnostic_results = self._run_task(diagnostic_task)
            self._associate(target.target_id, main_results, "main")
            self._associate(
                target.target_id, diagnostic_results, "diagnostic"
            )

    def _run_task(self, task: QueryTask) -> list[dict]:
        task.attempt_count += 1
        try:
            results = self._search_fn(**task.query)
        except Exception:
            task.status = "failed"
            raise
        task.status = "completed"
        result_ids = [item["poem_id"] for item in results]
        if task.attempt_count == 1:
            task.candidate_ids = result_ids
        else:
            # 恢复重筛是增量候选来源，不能改写初始化时的搜索事实与排序。
            for poem_id in result_ids:
                if poem_id not in task.candidate_ids:
                    task.candidate_ids.append(poem_id)
        for item in results:
            self.candidates.setdefault(item["poem_id"], dict(item))
        return results

    def _associate(
        self, target_id: int, results: list[dict], source_kind: str
    ) -> None:
        associated = self.target_candidate_ids[target_id]
        for item in results:
            poem_id = item["poem_id"]
            if poem_id not in associated:
                associated.append(poem_id)
            self.candidate_sources.setdefault(poem_id, {}).setdefault(
                target_id, set()
            ).add(source_kind)

    def _refresh(self) -> None:
        self.target_results = [
            self._target_result(target) for target in self.targets
        ]
        author_dist: dict[str, int] = {}
        for candidate in self.candidates.values():
            author = candidate.get("author")
            if isinstance(author, str):
                author_dist[author] = author_dist.get(author, 0) + 1
        self.profile = {
            "size": len(self.candidates),
            "author_dist": author_dist,
            "target_results": self.target_results,
            "theme_coverage": None,
        }
        self.verdict = _build_verdict(self.target_results)
        self.reference_stats = self._build_reference_stats()
        self.reference_verdict = _build_reference_verdict(
            self.available_target_coverage(),
            self.reference_stats["overall"],
            bool(self.detail_unavailable_target_ids),
        )

    def _target_result(self, target: Target) -> dict:
        target_id = target.target_id
        main_ids = self.main_tasks[target_id].candidate_ids
        diagnostic_task = self.diagnostic_tasks.get(target_id)
        diagnostic_ids = (
            diagnostic_task.candidate_ids if diagnostic_task is not None else []
        )
        status, basis = _target_status(
            target, main_ids, diagnostic_ids, self.candidates
        )
        candidate_ids = self.target_candidate_ids[target_id]
        loaded_ids = [
            poem_id for poem_id in candidate_ids
            if poem_id in self.loaded_details
        ]
        failed_ids = [
            poem_id for poem_id in candidate_ids
            if poem_id in self.failed_candidate_ids
        ]
        remaining_ids = [
            poem_id for poem_id in candidate_ids
            if poem_id not in self.loaded_details
            and poem_id not in self.failed_candidate_ids
        ]
        return {
            "target_id": target_id,
            "target": target.snapshot(),
            "status": status,
            "retrieval": "found" if candidate_ids else "empty",
            "candidate_count": len(candidate_ids),
            "visible_candidate_ids": remaining_ids[: self.visible_limit],
            "loaded_candidate_ids": loaded_ids,
            "loaded_candidate_count": len(loaded_ids),
            "failed_candidate_ids": failed_ids,
            "remaining_candidate_count": len(remaining_ids),
            "detail_access_status": (
                "unavailable"
                if target_id in self.detail_unavailable_target_ids
                else "available"
            ),
            "basis": basis,
            "theme_coverage": None,
        }

    def visible_candidate_ids(self) -> list[str]:
        """返回所有 target 当前窗口的合法 ID，按 target/候选顺序去重。"""
        visible: list[str] = []
        for result in self.target_results:
            for poem_id in result["visible_candidate_ids"]:
                if poem_id not in visible:
                    visible.append(poem_id)
        return visible

    def target_ids_for(self, poem_id: str) -> list[int]:
        return sorted(self.candidate_sources.get(poem_id, {}))

    def is_loaded(self, poem_id: str) -> bool:
        return poem_id in self.loaded_details

    def add_detail(
        self, detail: dict, session_poems: dict[int, str]
    ) -> int:
        """校验完整详情后，原子提交详情、会话编号与全部派生画像。"""
        copied = _validate_and_copy_detail(detail)
        poem_id = copied["poem_id"]
        if poem_id in self.loaded_details:
            for number, known_id in session_poems.items():
                if known_id == poem_id:
                    return number
            raise RuntimeError("详情池与会话编号状态不一致")
        if poem_id not in self.visible_candidate_ids():
            raise CandidatePoolProtocolError("poem_id 不在当前可见未读窗口")

        target_ids = self.target_ids_for(poem_id)
        source_kinds = sorted(
            {
                kind
                for kinds in self.candidate_sources[poem_id].values()
                for kind in kinds
            },
            key=("main", "diagnostic").index,
        )
        item = {
            **copied,
            "target_ids": target_ids,
            "source_kinds": source_kinds,
        }
        poem_number = next(
            (
                number
                for number, known_id in session_poems.items()
                if known_id == poem_id
            ),
            len(session_poems) + 1,
        )
        # 所有可能失败的验证和派生数据构造都已完成，下面一次提交。
        self.loaded_details[poem_id] = item
        session_poems[poem_number] = poem_id
        self._refresh()
        return poem_number

    def recover_failed_detail(self, poem_id: str) -> dict:
        """隔离失败候选，并对其关联 target 各执行至多一次受控重筛。"""
        self.failed_candidate_ids.add(poem_id)
        recovered: list[int] = []
        for target_id in self.target_ids_for(poem_id):
            if target_id in self._recovered_target_ids:
                continue
            self._recovered_target_ids.add(target_id)
            recovered.append(target_id)
            source_kinds = self.candidate_sources[poem_id][target_id]
            task = (
                self.diagnostic_tasks.get(target_id)
                if "diagnostic" in source_kinds
                else self.main_tasks[target_id]
            )
            if task is None:
                continue
            results = self._run_task(task)
            task.recovery_status = "completed"
            self._associate(target_id, results, task.kind)

        self._refresh()
        for target_id in self.target_ids_for(poem_id):
            result = self.target_results[target_id - 1]
            if (
                result["loaded_candidate_count"] == 0
                and not result["visible_candidate_ids"]
            ):
                self.detail_unavailable_target_ids.add(target_id)
                task = (
                    self.diagnostic_tasks.get(target_id)
                    if "diagnostic"
                    in self.candidate_sources[poem_id][target_id]
                    else self.main_tasks[target_id]
                )
                if task is not None:
                    task.recovery_status = "unavailable"
        self._refresh()
        return {
            "failed_poem_id": poem_id,
            "recovered_target_ids": recovered,
            "candidate_pool": self.model_snapshot(),
        }

    def available_target_coverage(self) -> dict:
        eligible: list[int] = []
        unavailable: list[int] = []
        loaded: list[int] = []
        for result in self.target_results:
            target_id = result["target_id"]
            has_usable_candidate = bool(
                result["loaded_candidate_count"]
                or result["remaining_candidate_count"]
            )
            if not has_usable_candidate:
                unavailable.append(target_id)
                continue
            eligible.append(target_id)
            if result["loaded_candidate_count"] > 0:
                loaded.append(target_id)
        unloaded = [target_id for target_id in eligible if target_id not in loaded]
        if not loaded:
            status = "none_loaded"
        elif unloaded:
            status = "partially_covered"
        else:
            status = "all_covered"
        return {
            "status": status,
            "eligible_target_ids": eligible,
            "loaded_target_ids": loaded,
            "unloaded_target_ids": unloaded,
            "unavailable_target_ids": unavailable,
            "loaded_target_ratio": (
                len(loaded) / len(eligible) if eligible else None
            ),
        }

    def _build_reference_stats(self) -> dict:
        by_poem = []
        for item in self.loaded_details.values():
            appr_count = len(item["appreciation"])
            anno_count = len(item["annotations"])
            by_poem.append(
                {
                    "poem_id": item["poem_id"],
                    "target_ids": list(item["target_ids"]),
                    "appreciation": _poem_dimension(
                        appr_count,
                        self.reference_baseline[
                            "appreciation_threshold"
                        ],
                    ),
                    "annotations": _poem_dimension(
                        anno_count,
                        self.reference_baseline["annotation_threshold"],
                    ),
                }
            )

        by_target = []
        for target in self.targets:
            poem_rows = [
                row
                for row in by_poem
                if target.target_id in row["target_ids"]
            ]
            by_target.append(
                {
                    "target_id": target.target_id,
                    "loaded_poem_ids": [
                        row["poem_id"] for row in poem_rows
                    ],
                    "appreciation": _aggregate_poem_dimension(
                        poem_rows, "appreciation"
                    ),
                    "annotations": _aggregate_poem_dimension(
                        poem_rows, "annotations"
                    ),
                }
            )
        overall = {
            "poem_count": len(by_poem),
            "appreciation": _overall_dimension(
                by_poem, by_target, "appreciation"
            ),
            "annotations": _overall_dimension(
                by_poem, by_target, "annotations"
            ),
        }
        return {
            "baseline": deepcopy(self.reference_baseline),
            "by_poem": by_poem,
            "by_target": by_target,
            "overall": overall,
        }

    def model_snapshot(self) -> dict:
        snapshot = self.public_snapshot()
        for result in snapshot["profile"]["target_results"]:
            result["visible_candidates"] = [
                deepcopy(self.candidates[poem_id])
                for poem_id in result["visible_candidate_ids"]
            ]
        return snapshot

    def public_snapshot(self) -> dict:
        return {
            "targets": [target.snapshot() for target in self.targets],
            "profile": {
                "size": self.profile["size"],
                "author_dist": dict(self.profile["author_dist"]),
                "target_results": deepcopy(self.target_results),
                "theme_coverage": None,
            },
            "verdict": self.verdict,
            "detail_pool": {
                "size": len(self.loaded_details),
                "items": [
                    {
                        key: deepcopy(item[key])
                        for key in (
                            "poem_id",
                            "title",
                            "author",
                            "dynasty",
                            "target_ids",
                            "source_kinds",
                        )
                    }
                    for item in self.loaded_details.values()
                ],
                "available_target_coverage": (
                    self.available_target_coverage()
                ),
            },
            "reference_stats": deepcopy(self.reference_stats),
            "reference_verdict": self.reference_verdict,
        }


def _validate_and_copy_detail(detail: dict) -> dict:
    if not isinstance(detail, dict) or "error" in detail:
        raise ValueError("只有成功的详情结果可以进入详情池")
    required_strings = ("poem_id", "title", "author", "dynasty", "content")
    for field_name in required_strings:
        if not isinstance(detail.get(field_name), str):
            raise ValueError(f"详情字段 {field_name} 必须是字符串")
    for field_name in ("appreciation", "annotations"):
        blocks = detail.get(field_name)
        if not isinstance(blocks, list) or any(
            not isinstance(block, dict) for block in blocks
        ):
            raise ValueError(f"详情字段 {field_name} 必须是对象列表")
    return deepcopy(detail)


def _poem_dimension(count: int, threshold: int) -> dict:
    return {
        "count": count,
        "label": "sufficient" if count >= threshold else "limited",
    }


def _summary(values: list[int]) -> dict:
    if not values:
        return {"total": 0, "min": None, "max": None, "median": None, "mean": None}
    return {
        "total": sum(values),
        "min": min(values),
        "max": max(values),
        "median": median(values),
        "mean": mean(values),
    }


def _aggregate_poem_dimension(rows: list[dict], dimension: str) -> dict:
    values = [row[dimension]["count"] for row in rows]
    sufficient_ids = [
        row["poem_id"]
        for row in rows
        if row[dimension]["label"] == "sufficient"
    ]
    limited_ids = [
        row["poem_id"]
        for row in rows
        if row[dimension]["label"] == "limited"
    ]
    count = len(rows)
    ratio = len(sufficient_ids) / count if count else None
    return {
        **_summary(values),
        "sufficient_count": len(sufficient_ids),
        "limited_count": len(limited_ids),
        "sufficient_ratio": ratio,
        "limited_poem_ids": limited_ids,
        "label": (
            "not_evaluated"
            if ratio is None
            else "sufficient" if ratio > 0.6 else "limited"
        ),
    }


def _overall_dimension(
    poem_rows: list[dict], target_rows: list[dict], dimension: str
) -> dict:
    evaluated = [
        row for row in target_rows
        if row[dimension]["label"] != "not_evaluated"
    ]
    sufficient = [
        row["target_id"]
        for row in evaluated
        if row[dimension]["label"] == "sufficient"
    ]
    limited = [
        row["target_id"]
        for row in evaluated
        if row[dimension]["label"] == "limited"
    ]
    ratio = len(sufficient) / len(evaluated) if evaluated else None
    return {
        **_summary([row[dimension]["count"] for row in poem_rows]),
        "sufficient_target_count": len(sufficient),
        "limited_target_count": len(limited),
        "sufficient_target_ratio": ratio,
        "limited_target_ids": limited,
        "label": (
            "not_evaluated"
            if ratio is None
            else "sufficient" if ratio > 0.6 else "limited"
        ),
    }


def _build_reference_verdict(
    coverage: dict, overall: dict, has_unavailable_detail: bool
) -> str:
    if overall["poem_count"] == 0:
        if has_unavailable_detail:
            return "候选详情不可用，参考量无法评估。"
        return "尚未读取作品详情，参考量未评估。"
    appr = overall["appreciation"]["label"]
    anno = overall["annotations"]["label"]
    description = {
        ("sufficient", "sufficient"): "赏析与注释参考量均充足",
        ("sufficient", "limited"): "赏析参考量充足，注释参考量较少",
        ("limited", "sufficient"): "赏析参考量较少，注释参考量充足",
        ("limited", "limited"): "赏析与注释参考量均较少",
    }[(appr, anno)]
    if (
        coverage["status"] == "all_covered"
        and not has_unavailable_detail
    ):
        return f"全部可用 targets 已取得详情；{description}。"
    suffix = (
        "；另有候选详情不可用"
        if has_unavailable_detail
        else ""
    )
    if coverage["status"] == "all_covered":
        return (
            f"全部剩余可用 targets 已取得详情；"
            f"{description}{suffix}。"
        )
    return f"仅部分可用 targets 已取得详情；已读部分{description}{suffix}。"


def normalize_targets(raw_targets) -> list[Target]:
    if not isinstance(raw_targets, list):
        raise CandidatePoolProtocolError("targets 必须是列表")
    normalized = []
    seen: set[tuple] = set()
    for index, raw in enumerate(raw_targets, start=1):
        if not isinstance(raw, dict):
            raise CandidatePoolProtocolError(f"targets[{index}] 必须是对象")
        unknown = set(raw) - _TARGET_FIELDS
        if unknown:
            raise CandidatePoolProtocolError(
                f"targets[{index}] 包含未知字段: {'、'.join(sorted(unknown))}"
            )
        author = _optional_string(f"targets[{index}].author", raw.get("author"))
        dynasty = _optional_string(f"targets[{index}].dynasty", raw.get("dynasty"))
        raw_title = _optional_string(f"targets[{index}].title", raw.get("title"))
        title = (store._normalize_title(raw_title) or None) if raw_title else None
        themes = _normalize_themes(raw.get("themes", []), target_index=index)
        if author is None and dynasty is None and title is None and not themes:
            raise CandidatePoolProtocolError(
                f"targets[{index}] 至少需要 author、dynasty、title 或一个 theme"
            )
        value = (author, dynasty, title, themes)
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if not 1 <= len(normalized) <= 4:
        raise CandidatePoolProtocolError(
            "targets 规范化去重后必须有 1–4 个，不能静默截断"
        )
    return [
        Target(index, *values)
        for index, values in enumerate(normalized, start=1)
    ]


def _optional_string(name: str, value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CandidatePoolProtocolError(f"{name} 必须是字符串或 None")
    return value.strip() or None


def _normalize_themes(raw_themes, *, target_index: int) -> tuple[str, ...]:
    if not isinstance(raw_themes, list):
        raise CandidatePoolProtocolError(
            f"targets[{target_index}].themes 必须是字符串列表"
        )
    themes: list[str] = []
    for theme_index, theme in enumerate(raw_themes, start=1):
        if not isinstance(theme, str) or not theme.strip():
            raise CandidatePoolProtocolError(
                f"targets[{target_index}].themes[{theme_index}] 必须是非空字符串"
            )
        normalized = theme.strip()
        if normalized not in themes:
            themes.append(normalized)
    return tuple(themes)


def _main_query(target: Target) -> dict:
    return {
        "query": THEME_SEPARATOR.join(target.themes) or None,
        "author": target.author,
        "dynasty": target.dynasty,
        "title": target.title,
    }


def _diagnostic_query(target: Target) -> dict | None:
    query = THEME_SEPARATOR.join(target.themes) or None
    if target.title is not None and (
        target.author is not None or target.dynasty is not None
    ):
        return {
            "query": query, "author": None, "dynasty": None,
            "title": target.title,
        }
    if (
        target.title is None
        and target.author is not None
        and target.dynasty is not None
    ):
        return {
            "query": query, "author": target.author,
            "dynasty": None, "title": None,
        }
    return None


def _target_status(
    target: Target,
    main_ids: list[str],
    diagnostic_ids: list[str],
    candidates: dict[str, dict],
) -> tuple[str, dict | None]:
    if not any((target.author, target.dynasty, target.title)):
        return "not_applicable", None
    if main_ids:
        strict_ids = [
            poem_id for poem_id in main_ids
            if _strictly_matches(target, candidates[poem_id])
        ]
        if strict_ids:
            return "matched", None
        if target.title is not None:
            return "partial_match", {
                "requested_title": target.title,
                "partial_title_candidate_ids": list(main_ids),
            }
        return "missing", None
    if diagnostic_ids:
        conflicts = []
        for poem_id in diagnostic_ids:
            item = candidates[poem_id]
            differences = {}
            for name in ("author", "dynasty"):
                expected = getattr(target, name)
                actual = item.get(name)
                if expected is not None and actual != expected:
                    differences[name] = {
                        "expected": expected, "actual": actual
                    }
            conflicts.append({"poem_id": poem_id, "differences": differences})
        return "conflict", {"diagnostic_conflicts": conflicts}
    return "missing", None


def _strictly_matches(target: Target, candidate: dict) -> bool:
    return (
        (target.author is None or candidate.get("author") == target.author)
        and (target.dynasty is None or candidate.get("dynasty") == target.dynasty)
        and (
            target.title is None
            or (
                isinstance(candidate.get("title"), str)
                and store._normalize_title(candidate["title"]) == target.title
            )
        )
    )


def _build_verdict(target_results: list[dict]) -> str:
    statuses = [item["status"] for item in target_results]
    if "conflict" in statuses:
        return "请求不符：条件诊断发现一个或多个 target 的作者或朝代与语料不一致。"
    if "partial_match" in statuses and "missing" in statuses:
        return "部分满足：存在标题部分匹配 target，另有 target 未命中。"
    if "partial_match" in statuses:
        return "标题部分匹配：至少一个 target 仅找到标题部分匹配候选。"
    if all(status == "missing" for status in statuses):
        return "未命中：所有 target 在允许的检索与诊断路径中均无结果。"
    if "missing" in statuses:
        return "部分满足：部分 target 已命中，另有 target 未命中。"
    if all(status == "not_applicable" for status in statuses):
        if any(item["retrieval"] == "found" for item in target_results):
            return "已取得主题排序候选，但主题覆盖待评估。"
        return "未命中：主题检索未取得候选，主题覆盖仍待评估。"
    if "not_applicable" in statuses:
        return "部分满足：结构化 target 已命中，主题覆盖仍待评估。"
    return "全部命中：所有 target 的结构化条件均得到严格匹配。"
