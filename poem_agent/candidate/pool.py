"""Candidate Pool 的唯一可变状态聚合根。"""

from __future__ import annotations

from copy import deepcopy

from .. import store
from ..request import (
    ConsolidatedRequest,
    ResolvedRequest,
    resolve_consolidated_request,
)
from .models import (
    FROZEN_TARGET_LIMIT,
    VISIBLE_CANDIDATE_LIMIT,
    CandidatePoolProtocolError,
    QueryTask,
    SearchFunction,
    Target,
    _target_identity,
    normalize_targets,
)
from .profile import (
    _aggregate_poem_dimension,
    _build_reference_verdict,
    _overall_dimension,
    _poem_dimension,
)
from .queries import _build_verdict, _diagnostic_query, _main_query, _target_status


def _default_search(**query) -> list[dict]:
    """经兼容入口延迟取得默认检索函数，保留调用方的注入点。"""
    from ..candidate_pool import retrieve_all_poems

    return retrieve_all_poems(**query)


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
        self._targets_by_id = {target.target_id: target for target in targets}
        self._active_target_ids = [target.target_id for target in targets]
        self._frozen_target_ids: list[int] = []
        self._next_target_id = max(self._targets_by_id, default=0) + 1
        self.visible_limit = visible_limit
        self._search_fn = search_fn or _default_search
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

    @property
    def targets(self) -> tuple[Target, ...]:
        """兼容既有消费者的 active-only target 视图。"""
        return tuple(
            self._targets_by_id[target_id]
            for target_id in self._active_target_ids
        )

    @property
    def frozen_targets(self) -> tuple[Target, ...]:
        return tuple(
            self._targets_by_id[target_id]
            for target_id in self._frozen_target_ids
        )

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

    @classmethod
    def initialize_request(
        cls,
        request: ConsolidatedRequest,
        *,
        search_fn: SearchFunction | None = None,
        visible_limit: int = VISIBLE_CANDIDATE_LIMIT,
        baseline: dict | None = None,
    ) -> tuple["CandidatePool", ResolvedRequest]:
        """从严格总请求初始化池，并同时返回不含临时 refs 的请求。"""
        if not isinstance(request, ConsolidatedRequest):
            raise TypeError("request 必须是 ConsolidatedRequest")
        targets = [
            Target(
                index,
                target.author,
                target.dynasty,
                target.title,
                target.themes,
            )
            for index, target in enumerate(request.targets, start=1)
        ]
        pool = cls(
            targets,
            search_fn=search_fn,
            visible_limit=visible_limit,
            baseline=baseline,
        )
        target_ids_by_ref = {
            request_target.target_ref: target.target_id
            for request_target, target in zip(request.targets, targets)
        }
        return pool, resolve_consolidated_request(request, target_ids_by_ref)

    def update_request(self, request: ConsolidatedRequest) -> ResolvedRequest:
        """按最新完整请求原子更新 active/frozen targets。"""
        if not isinstance(request, ConsolidatedRequest):
            raise TypeError("request 必须是 ConsolidatedRequest")

        identity_to_id = {
            _target_identity(target): target_id
            for target_id, target in self._targets_by_id.items()
        }
        desired_ids: list[int] = []
        target_ids_by_ref: dict[str, int] = {}
        new_targets: list[Target] = []
        next_target_id = self._next_target_id
        for request_target in request.targets:
            identity = (
                request_target.author,
                request_target.dynasty,
                request_target.title,
                request_target.themes,
            )
            target_id = identity_to_id.get(identity)
            if target_id is None:
                target_id = next_target_id
                next_target_id += 1
                target = Target(target_id, *identity)
                new_targets.append(target)
                identity_to_id[identity] = target_id
            desired_ids.append(target_id)
            target_ids_by_ref[request_target.target_ref] = target_id

        # 新 target 的全部检索先进入局部暂存区；任何异常都不会碰池状态或 ID。
        staged = [self._stage_target(target) for target in new_targets]
        resolved = resolve_consolidated_request(request, target_ids_by_ref)

        old_active = list(self._active_target_ids)
        frozen_ids = [
            target_id
            for target_id in self._frozen_target_ids
            if target_id not in desired_ids
        ]
        frozen_ids.extend(
            target_id
            for target_id in old_active
            if target_id not in desired_ids
        )
        evicted_ids = frozen_ids[:-FROZEN_TARGET_LIMIT]
        frozen_ids = frozen_ids[-FROZEN_TARGET_LIMIT:]

        rollback = self._mutable_state_snapshot()
        try:
            for target, main_task, diagnostic_task, results in staged:
                target_id = target.target_id
                self._targets_by_id[target_id] = target
                self.main_tasks[target_id] = main_task
                if diagnostic_task is not None:
                    self.diagnostic_tasks[target_id] = diagnostic_task
                self.target_candidate_ids[target_id] = []
                for items, source_kind in results:
                    for item in items:
                        self.candidates.setdefault(item["poem_id"], dict(item))
                    self._associate(target_id, items, source_kind)
            self._active_target_ids = desired_ids
            self._frozen_target_ids = frozen_ids
            self._next_target_id = next_target_id
            for target_id in evicted_ids:
                self._evict_target(target_id)
            self._refresh()
        except Exception:
            self._restore_mutable_state(rollback)
            raise
        return resolved

    def _stage_target(
        self, target: Target
    ) -> tuple[
        Target,
        QueryTask,
        QueryTask | None,
        list[tuple[list[dict], str]],
    ]:
        main_task = QueryTask(target.target_id, "main", _main_query(target))
        main_results = self._run_staged_task(main_task)
        diagnostic_task = None
        diagnostic_results: list[dict] = []
        diagnostic_query = _diagnostic_query(target)
        if diagnostic_query is not None:
            diagnostic_task = QueryTask(
                target.target_id, "diagnostic", diagnostic_query
            )
            if main_results:
                diagnostic_task.status = "skipped"
            else:
                diagnostic_results = self._run_staged_task(diagnostic_task)
        return (
            target,
            main_task,
            diagnostic_task,
            [(main_results, "main"), (diagnostic_results, "diagnostic")],
        )

    def _run_staged_task(self, task: QueryTask) -> list[dict]:
        task.attempt_count += 1
        try:
            results = self._search_fn(**task.query)
            task.candidate_ids = [item["poem_id"] for item in results]
        except Exception:
            task.status = "failed"
            raise
        task.status = "completed"
        return results

    def _mutable_state_snapshot(self) -> dict:
        names = (
            "_targets_by_id",
            "_active_target_ids",
            "_frozen_target_ids",
            "_next_target_id",
            "main_tasks",
            "diagnostic_tasks",
            "candidates",
            "target_candidate_ids",
            "candidate_sources",
            "loaded_details",
            "failed_candidate_ids",
            "detail_unavailable_target_ids",
            "_recovered_target_ids",
            "target_results",
            "profile",
            "verdict",
            "reference_stats",
            "reference_verdict",
        )
        return {name: deepcopy(getattr(self, name)) for name in names}

    def _restore_mutable_state(self, snapshot: dict) -> None:
        for name, value in snapshot.items():
            setattr(self, name, value)

    def _evict_target(self, target_id: int) -> None:
        self._targets_by_id.pop(target_id, None)
        self.main_tasks.pop(target_id, None)
        self.diagnostic_tasks.pop(target_id, None)
        candidate_ids = self.target_candidate_ids.pop(target_id, [])
        self.detail_unavailable_target_ids.discard(target_id)
        self._recovered_target_ids.discard(target_id)
        for poem_id in candidate_ids:
            sources = self.candidate_sources.get(poem_id)
            if sources is not None:
                sources.pop(target_id, None)
                if not sources:
                    self.candidate_sources.pop(poem_id, None)
                    if poem_id not in self.loaded_details:
                        self.candidates.pop(poem_id, None)
                        self.failed_candidate_ids.discard(poem_id)

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
        for poem_id in self._active_candidate_ids():
            candidate = self.candidates[poem_id]
            author = candidate.get("author")
            if isinstance(author, str):
                author_dist[author] = author_dist.get(author, 0) + 1
        self.profile = {
            "size": len(self._active_candidate_ids()),
            "author_dist": author_dist,
            "target_results": self.target_results,
            "theme_coverage": None,
        }
        self.verdict = _build_verdict(self.target_results)
        self.reference_stats = self._build_reference_stats()
        self.reference_verdict = _build_reference_verdict(
            self.available_target_coverage(),
            self.reference_stats["overall"],
            bool(
                self.detail_unavailable_target_ids
                & set(self._active_target_ids)
            ),
        )

    def _active_candidate_ids(self) -> list[str]:
        poem_ids: list[str] = []
        for target_id in self._active_target_ids:
            for poem_id in self.target_candidate_ids.get(target_id, []):
                if poem_id not in poem_ids:
                    poem_ids.append(poem_id)
        return poem_ids

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
        sources = self.candidate_sources.get(poem_id, {})
        return [
            target_id
            for target_id in self._active_target_ids
            if target_id in sources
        ]

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

        item = {
            **copied,
            "_loaded_target_ids": self.target_ids_for(poem_id),
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
        active_target_ids = self.target_ids_for(poem_id)
        if not active_target_ids or (
            poem_id not in self.visible_candidate_ids()
            and poem_id not in self.failed_candidate_ids
        ):
            raise CandidatePoolProtocolError(
                "poem_id 不属于当前 active target 的可恢复候选"
            )
        self.failed_candidate_ids.add(poem_id)
        recovered: list[int] = []
        for target_id in active_target_ids:
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
        results_by_id = {
            result["target_id"]: result for result in self.target_results
        }
        for target_id in active_target_ids:
            result = results_by_id[target_id]
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
            target_ids = self.target_ids_for(item["poem_id"])
            if not target_ids:
                continue
            appr_count = len(item["appreciation"])
            anno_count = len(item["annotations"])
            by_poem.append(
                {
                    "poem_id": item["poem_id"],
                    "target_ids": target_ids,
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
        snapshot.pop("frozen_targets", None)
        for result in snapshot["profile"]["target_results"]:
            result["visible_candidates"] = [
                deepcopy(self.candidates[poem_id])
                for poem_id in result["visible_candidate_ids"]
            ]
        return snapshot

    def public_snapshot(self) -> dict:
        detail_items = []
        for item in self.loaded_details.values():
            target_ids = self.target_ids_for(item["poem_id"])
            if not target_ids:
                continue
            source_kinds = sorted(
                {
                    kind
                    for target_id in target_ids
                    for kind in self.candidate_sources[item["poem_id"]][target_id]
                },
                key=("main", "diagnostic").index,
            )
            detail_items.append(
                {
                    "poem_id": item["poem_id"],
                    "title": item["title"],
                    "author": item["author"],
                    "dynasty": item["dynasty"],
                    "target_ids": target_ids,
                    "source_kinds": source_kinds,
                }
            )
        return {
            "targets": [target.snapshot() for target in self.targets],
            "frozen_targets": [
                {**target.snapshot(), "state": "frozen"}
                for target in self.frozen_targets
            ],
            "profile": {
                "size": self.profile["size"],
                "author_dist": dict(self.profile["author_dist"]),
                "target_results": deepcopy(self.target_results),
                "theme_coverage": None,
            },
            "verdict": self.verdict,
            "detail_pool": {
                "size": len(detail_items),
                "items": detail_items,
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

