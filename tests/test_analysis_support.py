import json
import unittest
from unittest.mock import patch

from poem_agent.agent import force_finish, run_agent
from poem_agent.analysis_support import (
    AnalysisAssessmentProtocolError,
    FORCE_FALLBACK_VERDICT,
    PARTIAL_VERDICT,
    SUFFICIENT_LIMITED_VERDICT,
    append_required_support_notice,
    evaluate_analysis_support,
    validate_analysis_assessment,
)
from poem_agent.candidate_pool import CandidatePool


BASELINE = {
    "method": "corpus_lower_quartile",
    "appreciation_threshold": 4,
    "annotation_threshold": 5,
    "aggregate_ratio_threshold": 0.6,
    "comparison": "strictly_greater",
}


def candidate(poem_id, title="甲诗", author="甲"):
    return {
        "poem_id": poem_id,
        "title": title,
        "author": author,
        "dynasty": "唐",
        "score": None,
    }


def detail(poem_id, title="甲诗", *, appr=4, anno=5):
    return {
        "poem_id": poem_id,
        "title": title,
        "author": "甲",
        "dynasty": "唐",
        "content": "正文",
        "appreciation": [
            {"evidence_id": f"{poem_id}#appr-{index}", "text": "赏析"}
            for index in range(appr)
        ],
        "annotations": [
            {"evidence_id": f"{poem_id}#anno-{index}", "text": "注释"}
            for index in range(anno)
        ],
    }


def evidence(poem_id):
    return [
        {
            "evidence_id": f"{poem_id}#appr-0",
            "poem_id": poem_id,
            "text": "赏析",
            "title": "甲诗",
        }
    ]


def assessment(level, target_ids):
    return {"level": level, "target_ids": target_ids}


def finish(answer, level="not_applicable", target_ids=None):
    return {
        "action": "finish",
        "action_input": {
            "answer": answer,
            "analysis_assessment": assessment(level, target_ids or []),
        },
    }


def request_input(*, author=None, title=None, task_type="appreciate"):
    task = {"type": task_type, "target_refs": ["t1"]}
    if task_type in {"appreciate", "compare"}:
        task.update({"aspects": [], "custom_aspects": []})
    return {
        "targets": [{
            "target_ref": "t1",
            "author": author,
            "dynasty": None,
            "title": title,
            "themes": [],
        }],
        "tasks": [task],
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        return (
            response
            if isinstance(response, str)
            else json.dumps(response, ensure_ascii=False)
        )


class AssessmentProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pool = CandidatePool.initialize(
            [{"author": "甲"}, {"author": "乙"}],
            search_fn=lambda **query: [
                candidate(
                    "p1" if query["author"] == "甲" else "p2",
                    author=query["author"],
                )
            ],
            baseline=BASELINE,
        )

    def test_exact_fields_and_stable_target_order(self):
        payload = validate_analysis_assessment(
            {
                "answer": "完整回答文本。",
                "analysis_assessment": {
                    "level": "partial",
                    "target_ids": [2, 1],
                },
            },
            self.pool,
        )
        self.assertEqual(
            set(payload), {"answer", "analysis_assessment"}
        )
        self.assertEqual(
            set(payload["analysis_assessment"]), {"level", "target_ids"}
        )
        self.assertEqual(
            payload["analysis_assessment"]["target_ids"], [1, 2]
        )

    def test_rejects_missing_unknown_level_and_bad_ids(self):
        invalid = [
            {"answer": "回答"},
            {
                "answer": "回答",
                "extra": 1,
                "analysis_assessment": assessment("partial", [1]),
            },
            {
                "answer": "回答",
                "analysis_assessment": {
                    **assessment("partial", [1]),
                    "poem_ids": ["p1"],
                },
            },
            {
                "answer": "回答",
                "analysis_assessment": assessment("high", [1]),
            },
            {
                "answer": "回答",
                "analysis_assessment": assessment("partial", [True]),
            },
            {
                "answer": "回答",
                "analysis_assessment": assessment("partial", [1, 1]),
            },
            {
                "answer": "回答",
                "analysis_assessment": assessment("partial", [3]),
            },
            {
                "answer": "回答",
                "analysis_assessment": assessment("not_applicable", [1]),
            },
            {
                "answer": "回答",
                "analysis_assessment": assessment("partial", []),
            },
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(AnalysisAssessmentProtocolError):
                    validate_analysis_assessment(payload, self.pool)

    def test_poolless_conversation_is_only_not_applicable(self):
        valid = validate_analysis_assessment(
            {
                "answer": "这是一段普通交流回答。",
                "analysis_assessment": assessment("not_applicable", []),
            },
            None,
        )
        self.assertEqual(
            valid["analysis_assessment"], assessment("not_applicable", [])
        )
        with self.assertRaises(AnalysisAssessmentProtocolError):
            validate_analysis_assessment(
                {
                    "answer": "这是一段分析回答。",
                    "analysis_assessment": assessment("sufficient", [1]),
                },
                None,
            )

    def test_invalid_finish_uses_recovery_then_model_can_correct(self):
        llm = FakeLLM(
            [
                {"action": "finish", "action_input": {"answer": "旧格式回答"}},
                finish("修正后的完整元信息回答。"),
            ]
        )
        result = run_agent("你好", llm)
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("finish 协议错误", llm.prompts[1])
        self.assertEqual(result["analysis_support"]["level"], "not_applicable")
        self.assertFalse(result["degraded"])


class ObjectiveSupportTests(unittest.TestCase):
    def make_pool(self, targets, search_fn):
        return CandidatePool.initialize(
            targets, search_fn=search_fn, baseline=BASELINE
        )

    def load(self, pool, poem_id, **kwargs):
        pool.add_detail(detail(poem_id, **kwargs), {})

    def test_matched_evidence_is_sufficient_and_limited_is_only_soft(self):
        pool = self.make_pool(
            [{"author": "甲"}], lambda **_: [candidate("p1")]
        )
        self.load(pool, "p1", appr=1, anno=1)
        result = evaluate_analysis_support(
            assessment("sufficient", [1]), pool, evidence("p1")
        )
        self.assertEqual(result.maximum, "sufficient")
        self.assertEqual(result.analysis_support["level"], "sufficient")
        self.assertEqual(
            result.analysis_support["verdict"],
            SUFFICIENT_LIMITED_VERDICT,
        )
        self.assertTrue(result.reference_limited)

    def test_shared_poem_evidence_supports_both_targets_once(self):
        shared = candidate("shared")
        pool = self.make_pool(
            [{"author": "甲"}, {"title": "甲诗"}],
            lambda **_: [shared],
        )
        self.load(pool, "shared")
        result = evaluate_analysis_support(
            assessment("sufficient", [1, 2]), pool, evidence("shared")
        )
        self.assertEqual(
            result.target_states, {1: "fully_supported", 2: "fully_supported"}
        )
        self.assertEqual(result.evidence_poem_ids, ("shared",))
        self.assertEqual(result.maximum, "sufficient")

    def test_partial_match_and_conflict_cap_at_partial(self):
        cases = [
            (
                [{"title": "甲"}],
                lambda **_: [candidate("p1", title="甲诗")],
            ),
            (
                [{"author": "乙", "title": "甲诗"}],
                lambda **query: (
                    [] if query["author"] == "乙" else [candidate("p1")]
                ),
            ),
        ]
        for targets, search in cases:
            with self.subTest(targets=targets):
                pool = self.make_pool(targets, search)
                self.load(pool, "p1")
                result = evaluate_analysis_support(
                    assessment("sufficient", [1]), pool, evidence("p1")
                )
                self.assertEqual(result.maximum, "partial")
                self.assertEqual(
                    result.target_states[1], "partially_supported"
                )
                self.assertEqual(result.analysis_support["level"], "partial")

    def test_missing_or_uncovered_target_makes_partial_when_another_is_supported(self):
        def search(**query):
            return [candidate("p1")] if query["author"] == "甲" else []

        pool = self.make_pool([{"author": "甲"}, {"author": "乙"}], search)
        self.load(pool, "p1")
        result = evaluate_analysis_support(
            assessment("sufficient", [1, 2]), pool, evidence("p1")
        )
        self.assertEqual(
            result.target_states, {1: "fully_supported", 2: "unsupported"}
        )
        self.assertEqual(result.maximum, "partial")

    def test_detail_unavailable_target_cannot_supply_support(self):
        pool = self.make_pool(
            [{"author": "甲"}], lambda **_: [candidate("bad")]
        )
        pool.recover_failed_detail("bad")
        self.assertEqual(
            pool.target_results[0]["detail_access_status"], "unavailable"
        )
        result = evaluate_analysis_support(
            assessment("sufficient", [1]), pool, []
        )
        self.assertEqual(result.target_states[1], "unsupported")
        self.assertEqual(result.maximum, "insufficient")

    def test_no_legal_evidence_is_insufficient_and_model_is_never_raised(self):
        pool = self.make_pool(
            [{"author": "甲"}], lambda **_: [candidate("p1")]
        )
        self.load(pool, "p1")
        no_evidence = evaluate_analysis_support(
            assessment("sufficient", [1]), pool, []
        )
        self.assertEqual(no_evidence.maximum, "insufficient")
        self.assertEqual(
            no_evidence.analysis_support["level"], "insufficient"
        )
        lower = evaluate_analysis_support(
            assessment("partial", [1]), pool, evidence("p1")
        )
        self.assertEqual(lower.maximum, "sufficient")
        self.assertEqual(lower.analysis_support["level"], "partial")

    def test_theme_only_can_be_supported_and_degraded_caps_level(self):
        pool = self.make_pool(
            [{"themes": ["月亮"]}], lambda **_: [candidate("p1")]
        )
        self.load(pool, "p1")
        normal = evaluate_analysis_support(
            assessment("sufficient", [1]), pool, evidence("p1")
        )
        self.assertEqual(normal.target_states[1], "fully_supported")
        self.assertEqual(normal.analysis_support["level"], "sufficient")
        degraded = evaluate_analysis_support(
            assessment("sufficient", [1]),
            pool,
            evidence("p1"),
            degraded=True,
        )
        self.assertEqual(degraded.analysis_support["level"], "partial")

    def test_required_notice_is_deterministic_and_not_duplicated(self):
        support = {
            "level": "partial",
            "target_ids": [1],
            "verdict": PARTIAL_VERDICT,
        }
        once = append_required_support_notice("已有范围受限的回答。", support)
        twice = append_required_support_notice(once, support)
        self.assertEqual(once, twice)
        self.assertEqual(once.count("分析支撑说明："), 1)


class UnifiedFinalCheckTests(unittest.TestCase):
    def test_incomplete_dangling_and_overclaim_share_one_regeneration(self):
        poem = detail("p1", title="甲诗")
        partial_line = f"分析支撑说明：{PARTIAL_VERDICT}"
        llm = FakeLLM(
            [
                {
                    "action": "initialize_candidate_pool",
                    "action_input": request_input(title="甲"),
                },
                {
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "p1"},
                },
                finish("[诗1-appr-99]", "sufficient", [1]),
                finish(
                    f"从现有资料看，这是修正后的有限分析 [诗1-appr-0]\n{partial_line}",
                    "partial",
                    [1],
                ),
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[candidate("p1", title="甲诗")],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": lambda poem_id: poem},
                clear=True,
            ),
        ):
            result = run_agent("分析标题为甲的诗", llm)

        self.assertEqual(len(llm.prompts), 4)
        feedback = llm.prompts[-1]
        self.assertIn("疑似截断", feedback)
        self.assertIn("悬空引用", feedback)
        self.assertIn("客观上限", feedback)
        self.assertIn("target 1", feedback)
        self.assertEqual(result["analysis_support"]["level"], "partial")
        self.assertEqual(result["evidence"][0]["poem_id"], "p1")
        self.assertFalse(result["degraded"])

    def test_second_overclaim_is_clamped_without_another_call(self):
        poem = detail("p1", title="甲诗")
        llm = FakeLLM(
            [
                {
                    "action": "initialize_candidate_pool",
                    "action_input": request_input(title="甲"),
                },
                {
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "p1"},
                },
                finish(
                    "这是带真实依据但申报过高的完整分析 [诗1-appr-0]",
                    "sufficient",
                    [1],
                ),
                finish(
                    "这是第二次仍然申报过高的完整分析 [诗1-appr-0]",
                    "sufficient",
                    [1],
                ),
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[candidate("p1", title="甲诗")],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": lambda poem_id: poem},
                clear=True,
            ),
        ):
            result = run_agent("分析标题为甲的诗", llm)

        self.assertEqual(len(llm.prompts), 4)
        self.assertEqual(result["analysis_support"]["level"], "partial")
        self.assertIn("分析支撑说明：", result["answer"])
        self.assertFalse(result["degraded"])

    def test_system_notice_does_not_require_model_regeneration(self):
        poem = detail("p1")
        invalid_second = finish(
            "第二次回答正文和引用有效 [诗1-appr-0]",
            "partial",
            [1],
        )
        invalid_second["action_input"]["analysis_assessment"]["extra"] = True
        llm = FakeLLM(
            [
                {
                    "action": "initialize_candidate_pool",
                    "action_input": request_input(author="甲"),
                },
                {
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "p1"},
                },
                finish(
                    "第一次采用保守等级但没有固定说明 [诗1-appr-0]",
                    "partial",
                    [1],
                ),
                invalid_second,
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[candidate("p1")],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": lambda poem_id: poem},
                clear=True,
            ),
        ):
            result = run_agent("分析甲诗", llm)

        self.assertEqual(len(llm.prompts), 3)
        self.assertIn("第一次采用保守等级", result["answer"])
        self.assertEqual(result["analysis_support"]["level"], "partial")
        self.assertEqual(result["answer"].count("分析支撑说明："), 1)
        self.assertFalse(result["degraded"])

    def test_force_finish_accepts_wrapper_and_has_stable_public_shape(self):
        llm = FakeLLM(
            [finish("这是包装形式的强制收尾元信息回答。")]
        )
        result = force_finish("列举信息", llm, [], {})
        self.assertEqual(len(llm.prompts), 1)
        self.assertEqual(
            set(result["analysis_support"]),
            {"level", "target_ids", "verdict"},
        )
        self.assertEqual(result["analysis_support"]["level"], "not_applicable")

    def test_force_finish_accepts_compact_structure(self):
        llm = FakeLLM(
            [
                {
                    "answer": "这是紧凑形式的强制收尾元信息回答。",
                    "analysis_assessment": assessment(
                        "not_applicable", []
                    ),
                }
            ]
        )
        result = force_finish("列举信息", llm, [], {})
        self.assertEqual(len(llm.prompts), 1)
        self.assertEqual(result["analysis_support"]["level"], "not_applicable")

    def test_second_dangling_response_degrades_without_third_call(self):
        poem = detail("p1")
        llm = FakeLLM(
            [
                {
                    "action": "initialize_candidate_pool",
                    "action_input": request_input(author="甲"),
                },
                {
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "p1"},
                },
                finish(
                    "第一次回答仍使用悬空引用 [诗1-appr-99]",
                    "sufficient",
                    [1],
                ),
                finish(
                    "第二次回答仍使用悬空引用 [诗1-appr-98]",
                    "sufficient",
                    [1],
                ),
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[candidate("p1")],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": lambda poem_id: poem},
                clear=True,
            ),
        ):
            result = run_agent("分析甲诗", llm)

        self.assertEqual(len(llm.prompts), 4)
        self.assertTrue(result["degraded"])
        self.assertTrue(result["evidence"][0]["dangling"])
        self.assertEqual(
            result["analysis_support"]["level"], "insufficient"
        )
        self.assertIn("部分解读未能匹配到可靠出处", result["answer"])
        self.assertIn("分析支撑说明：", result["answer"])

    def test_force_finish_twice_invalid_uses_conservative_fallback(self):
        llm = FakeLLM(
            [
                {"answer": "第一次完整回答，但 assessment 缺失。"},
                {"answer": "第二次完整回答，assessment 仍然缺失。"},
            ]
        )
        result = force_finish("请分析", llm, [], {})
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(
            result["analysis_support"],
            {
                "level": "insufficient",
                "target_ids": [],
                "verdict": FORCE_FALLBACK_VERDICT,
            },
        )
        self.assertFalse(result["degraded"])
        self.assertIn(FORCE_FALLBACK_VERDICT, result["answer"])


if __name__ == "__main__":
    unittest.main()
