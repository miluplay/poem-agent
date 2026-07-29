"""最终回答的任务相关分析支撑计算。

本模块只读取 Candidate Pool 和已经完成绑定的 evidence。它不判断文学观点
真假，也不修改搜索、详情池或参考量画像。
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_pool import CandidatePool


LEVELS = (
    "not_applicable",
    "sufficient",
    "partial",
    "insufficient",
)
_LEVEL_RANK = {"insufficient": 0, "partial": 1, "sufficient": 2}

NOT_APPLICABLE_VERDICT = "本次请求不涉及内容分析。"
SUFFICIENT_VERDICT = "现有详情与合法依据覆盖本次分析范围。"
SUFFICIENT_LIMITED_VERDICT = (
    "现有合法依据覆盖本次分析范围；部分作品参考量低于语料常见水平。"
)
PARTIAL_VERDICT = (
    "现有详情与合法依据只能覆盖部分分析范围，未覆盖对象已在回答中说明。"
)
INSUFFICIENT_VERDICT = (
    "当前没有足够的可用详情与合法依据支持所要求的主体分析。"
)
FORCE_FALLBACK_VERDICT = "步数耗尽，未能形成可验证的分析支撑评估。"


class AnalysisAssessmentProtocolError(ValueError):
    """模型提交的 finish analysis_assessment 不符合协议。"""


@dataclass(frozen=True)
class SupportEvaluation:
    analysis_support: dict
    maximum: str
    target_states: dict[int, str]
    evidence_poem_ids: tuple[str, ...]
    reference_limited: bool


def validate_analysis_assessment(
    action_input,
    candidate_pool: CandidatePool | None,
) -> dict:
    """校验正常 finish 的精确 action_input，并稳定规范化 target 顺序。"""
    if not isinstance(action_input, dict):
        raise AnalysisAssessmentProtocolError("action_input 必须是对象")
    fields = set(action_input)
    if fields != {"answer", "analysis_assessment"}:
        unknown = sorted(fields - {"answer", "analysis_assessment"})
        missing = sorted({"answer", "analysis_assessment"} - fields)
        details = []
        if missing:
            details.append(f"缺少字段: {', '.join(missing)}")
        if unknown:
            details.append(f"包含未知字段: {', '.join(unknown)}")
        raise AnalysisAssessmentProtocolError(
            "finish.action_input 只能且必须包含 answer、analysis_assessment"
            + (f"（{'；'.join(details)}）" if details else "")
        )
    if not isinstance(action_input["answer"], str):
        raise AnalysisAssessmentProtocolError("answer 必须是字符串")

    assessment = action_input["analysis_assessment"]
    if not isinstance(assessment, dict):
        raise AnalysisAssessmentProtocolError("analysis_assessment 必须是对象")
    fields = set(assessment)
    if fields != {"level", "target_ids"}:
        unknown = sorted(fields - {"level", "target_ids"})
        missing = sorted({"level", "target_ids"} - fields)
        details = []
        if missing:
            details.append(f"缺少字段: {', '.join(missing)}")
        if unknown:
            details.append(f"包含未知字段: {', '.join(unknown)}")
        raise AnalysisAssessmentProtocolError(
            "analysis_assessment 只能且必须包含 level、target_ids"
            + (f"（{'；'.join(details)}）" if details else "")
        )

    level = assessment["level"]
    if level not in LEVELS:
        raise AnalysisAssessmentProtocolError(
            "level 必须是 not_applicable、sufficient、partial 或 insufficient"
        )
    target_ids = assessment["target_ids"]
    if not isinstance(target_ids, list):
        raise AnalysisAssessmentProtocolError("target_ids 必须是列表")
    if any(
        isinstance(target_id, bool) or not isinstance(target_id, int)
        for target_id in target_ids
    ):
        raise AnalysisAssessmentProtocolError("target_ids 元素必须是非布尔整数")
    if len(set(target_ids)) != len(target_ids):
        raise AnalysisAssessmentProtocolError("target_ids 不得重复")

    known_ids = (
        [target.target_id for target in candidate_pool.targets]
        if candidate_pool is not None
        else []
    )
    unknown_ids = [target_id for target_id in target_ids if target_id not in known_ids]
    if unknown_ids:
        raise AnalysisAssessmentProtocolError(
            f"target_ids 包含不存在的 ID: {unknown_ids}"
        )
    if level == "not_applicable":
        if target_ids:
            raise AnalysisAssessmentProtocolError(
                "not_applicable 时 target_ids 必须为空"
            )
    else:
        if candidate_pool is None:
            raise AnalysisAssessmentProtocolError(
                f"{level} 要求 Candidate Pool 已初始化"
            )
        if not target_ids:
            raise AnalysisAssessmentProtocolError(
                f"{level} 时 target_ids 至少包含一个真实 target ID"
            )

    ordered_ids = [target_id for target_id in known_ids if target_id in target_ids]
    return {
        "answer": action_input["answer"],
        "analysis_assessment": {
            "level": level,
            "target_ids": ordered_ids,
        },
    }


def evaluate_analysis_support(
    assessment: dict,
    candidate_pool: CandidatePool | None,
    evidence: list[dict],
    *,
    degraded: bool = False,
) -> SupportEvaluation:
    """用合法 evidence 和池状态给模型申报设置客观上限。"""
    proposed = assessment["level"]
    target_ids = list(assessment["target_ids"])
    if proposed == "not_applicable":
        support = {
            "level": "not_applicable",
            "target_ids": [],
            "verdict": NOT_APPLICABLE_VERDICT,
        }
        return SupportEvaluation(support, "not_applicable", {}, (), False)

    valid_poem_ids = _valid_evidence_poem_ids(evidence)
    target_states: dict[int, str] = {}
    supported_target_count = 0
    for target_id in target_ids:
        result = _target_result(candidate_pool, target_id)
        target_poems = {
            poem_id
            for poem_id in valid_poem_ids
            if target_id in candidate_pool.target_ids_for(poem_id)
            and candidate_pool.is_loaded(poem_id)
        }
        if not target_poems or result["status"] == "missing":
            state = "unsupported"
        elif (
            result["status"] in {"partial_match", "conflict"}
            or result["detail_access_status"] == "unavailable"
        ):
            state = "partially_supported"
        else:
            # matched，或仅主题 not_applicable，均可由真实详情和依据支撑。
            state = "fully_supported"
        target_states[target_id] = state
        if target_poems:
            supported_target_count += 1

    if supported_target_count == 0:
        maximum = "insufficient"
    elif all(state == "fully_supported" for state in target_states.values()):
        maximum = "sufficient"
    else:
        maximum = "partial"
    if degraded and maximum == "sufficient":
        maximum = "partial"

    final_level = (
        proposed
        if _LEVEL_RANK[proposed] <= _LEVEL_RANK[maximum]
        else maximum
    )
    reference_limited = _has_reference_limit(
        candidate_pool, set(valid_poem_ids)
    )
    verdict = _verdict(final_level, reference_limited)
    support = {
        "level": final_level,
        "target_ids": target_ids,
        "verdict": verdict,
    }
    return SupportEvaluation(
        support,
        maximum,
        target_states,
        tuple(valid_poem_ids),
        reference_limited,
    )


def force_fallback_analysis_support() -> dict:
    return {
        "level": "insufficient",
        "target_ids": [],
        "verdict": FORCE_FALLBACK_VERDICT,
    }


def append_required_support_notice(answer: str, analysis_support: dict) -> str:
    """为 partial/insufficient 提供确定、可见且不重复的系统说明。"""
    if analysis_support["level"] not in {"partial", "insufficient"}:
        return answer
    line = f"分析支撑说明：{analysis_support['verdict']}"
    stripped = answer.rstrip()
    if line in stripped.splitlines():
        return stripped
    return f"{stripped}{chr(10) if stripped else ''}{line}"


def required_support_notice(analysis_support: dict) -> str | None:
    if analysis_support["level"] not in {"partial", "insufficient"}:
        return None
    return f"分析支撑说明：{analysis_support['verdict']}"


def _valid_evidence_poem_ids(evidence: list[dict]) -> list[str]:
    poem_ids: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        poem_id = item.get("poem_id")
        if (
            item.get("dangling") is not True
            and isinstance(poem_id, str)
            and poem_id not in poem_ids
        ):
            poem_ids.append(poem_id)
    return poem_ids


def _target_result(candidate_pool: CandidatePool, target_id: int) -> dict:
    return next(
        result
        for result in candidate_pool.target_results
        if result["target_id"] == target_id
    )


def _has_reference_limit(
    candidate_pool: CandidatePool, evidence_poem_ids: set[str]
) -> bool:
    for row in candidate_pool.reference_stats["by_poem"]:
        if row["poem_id"] not in evidence_poem_ids:
            continue
        if (
            row["appreciation"]["label"] == "limited"
            or row["annotations"]["label"] == "limited"
        ):
            return True
    return False


def _verdict(level: str, reference_limited: bool) -> str:
    if level == "sufficient":
        return (
            SUFFICIENT_LIMITED_VERDICT
            if reference_limited
            else SUFFICIENT_VERDICT
        )
    if level == "partial":
        return PARTIAL_VERDICT
    return INSUFFICIENT_VERDICT
