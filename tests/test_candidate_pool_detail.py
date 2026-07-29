import json
import unittest
from unittest.mock import Mock, patch

import poem_agent.agent as agent_module
from poem_agent import store
from poem_agent.agent import run_agent
from poem_agent.agent.prompts import SYSTEM_INSTRUCTION
from poem_agent.candidate_pool import CandidatePool


BASELINE = {
    "method": "corpus_lower_quartile",
    "appreciation_threshold": 4,
    "annotation_threshold": 5,
    "aggregate_ratio_threshold": 0.6,
    "comparison": "strictly_greater",
}


def candidate(poem_id, author="甲", title=None):
    return {
        "poem_id": poem_id,
        "title": title or poem_id,
        "author": author,
        "dynasty": "唐",
        "score": None,
    }


def detail(poem_id, *, appr=4, anno=5, author="甲", title=None):
    return {
        "poem_id": poem_id,
        "title": title or poem_id,
        "author": author,
        "dynasty": "唐",
        "content": "正文不计入参考量",
        "appreciation": [
            {"evidence_id": f"{poem_id}#appr-{i}", "text": "析"}
            for i in range(appr)
        ],
        "annotations": [
            {"evidence_id": f"{poem_id}#anno-{i}", "text": "注"}
            for i in range(anno)
        ],
    }


def init(targets):
    return {
        "action": "initialize_candidate_pool",
        "action_input": {"targets": targets},
    }


def get(poem_id):
    return {
        "action": "get_poem_detail",
        "action_input": {"poem_id": poem_id},
    }


def finish(answer="这是完整且可返回的测试回答。"):
    return {
        "action": "finish",
        "action_input": {
            "answer": answer,
            "analysis_assessment": {
                "level": "not_applicable",
                "target_ids": [],
            },
        },
    }


def forced(answer):
    return {
        "answer": answer,
        "analysis_assessment": {
            "level": "not_applicable",
            "target_ids": [],
        },
    }


class FakeLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        decision = next(self.decisions)
        if isinstance(decision, str):
            return decision
        return json.dumps(decision, ensure_ascii=False)


class DetailPoolStateTests(unittest.TestCase):
    def pool(self, ids, targets=None, search_fn=None):
        items = [candidate(poem_id) for poem_id in ids]
        return CandidatePool.initialize(
            targets or [{"author": "甲"}],
            search_fn=search_fn or (lambda **_: items),
            baseline=BASELINE,
        )

    def test_real_corpus_baseline_and_invalid_inputs(self):
        baseline = store.reference_count_baseline()
        self.assertEqual(baseline["appreciation_threshold"], 4)
        self.assertEqual(baseline["annotation_threshold"], 5)
        for invalid in (
            [],
            [{}],
            ["bad"],
            [{"appreciation": ["bad"], "annotations": []}],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    store.reference_count_baseline(invalid)

    def test_success_is_unique_and_rolls_visible_window(self):
        pool = self.pool([f"p{i}" for i in range(6)])
        sessions = {}
        search_verdict = pool.verdict
        self.assertEqual(pool.visible_candidate_ids(), [f"p{i}" for i in range(5)])

        self.assertEqual(pool.add_detail(detail("p0"), sessions), 1)
        result = pool.target_results[0]
        self.assertEqual(result["candidate_count"], 6)
        self.assertEqual(result["loaded_candidate_ids"], ["p0"])
        self.assertEqual(
            result["visible_candidate_ids"], ["p1", "p2", "p3", "p4", "p5"]
        )
        self.assertEqual(pool.verdict, search_verdict)
        self.assertEqual(pool.public_snapshot()["detail_pool"]["size"], 1)
        self.assertNotIn(
            "content", pool.public_snapshot()["detail_pool"]["items"][0]
        )
        self.assertEqual(pool.add_detail(detail("p0"), sessions), 1)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(pool.reference_stats["overall"]["poem_count"], 1)

    def test_shared_poem_covers_two_targets_and_counts_overall_once(self):
        def search(**_):
            return [candidate("shared")]

        pool = self.pool(
            [],
            targets=[{"author": "甲"}, {"dynasty": "唐"}],
            search_fn=search,
        )
        pool.add_detail(detail("shared"), {})
        snapshot = pool.public_snapshot()
        item = snapshot["detail_pool"]["items"][0]
        self.assertEqual(item["target_ids"], [1, 2])
        self.assertEqual(item["source_kinds"], ["main"])
        self.assertEqual(
            snapshot["detail_pool"]["available_target_coverage"]["status"],
            "all_covered",
        )
        self.assertEqual(snapshot["reference_stats"]["overall"]["poem_count"], 1)
        self.assertEqual(len(snapshot["reference_stats"]["by_target"]), 2)

    def test_conflict_diagnostic_source_does_not_change_search_status(self):
        def search(**query):
            if query["author"] == "李白":
                return []
            return [candidate("spring", author="杜甫", title="春望")]

        pool = CandidatePool.initialize(
            [{"author": "李白", "title": "春望"}],
            search_fn=search,
            baseline=BASELINE,
        )
        pool.add_detail(
            detail("spring", author="杜甫", title="春望"), {}
        )
        self.assertEqual(pool.target_results[0]["status"], "conflict")
        self.assertEqual(
            pool.public_snapshot()["detail_pool"]["items"][0]["source_kinds"],
            ["diagnostic"],
        )
        self.assertTrue(pool.verdict.startswith("请求不符"))

    def test_coverage_excludes_missing_target(self):
        def search(**query):
            return [candidate("p")] if query["author"] == "甲" else []

        pool = self.pool(
            [],
            targets=[{"author": "甲"}, {"author": "无"}],
            search_fn=search,
        )
        coverage = pool.available_target_coverage()
        self.assertEqual(coverage["status"], "none_loaded")
        self.assertEqual(coverage["eligible_target_ids"], [1])
        self.assertEqual(coverage["unavailable_target_ids"], [2])
        pool.add_detail(detail("p"), {})
        self.assertEqual(
            pool.available_target_coverage()["status"], "all_covered"
        )

    def test_reference_thresholds_and_strict_point_six_ratio(self):
        pool = self.pool([f"p{i}" for i in range(5)])
        sessions = {}
        for i in range(5):
            pool.add_detail(
                detail(
                    f"p{i}",
                    appr=4 if i < 3 else 3,
                    anno=5 if i < 3 else 4,
                ),
                sessions,
            )
        stats = pool.reference_stats
        target = stats["by_target"][0]
        self.assertEqual(target["appreciation"]["sufficient_ratio"], 0.6)
        self.assertEqual(target["appreciation"]["label"], "limited")
        self.assertEqual(stats["by_poem"][0]["appreciation"]["label"], "sufficient")
        self.assertEqual(stats["by_poem"][0]["annotations"]["label"], "sufficient")
        self.assertEqual(stats["by_poem"][3]["appreciation"]["label"], "limited")
        self.assertEqual(target["appreciation"]["total"], 18)
        self.assertEqual(target["appreciation"]["median"], 4)
        self.assertEqual(target["appreciation"]["mean"], 3.6)

    def test_no_details_are_not_evaluated_and_have_fixed_verdict(self):
        pool = self.pool(["p"])
        stats = pool.reference_stats
        self.assertEqual(
            stats["by_target"][0]["appreciation"]["label"], "not_evaluated"
        )
        self.assertEqual(
            stats["overall"]["annotations"]["label"], "not_evaluated"
        )
        self.assertEqual(
            pool.reference_verdict, "尚未读取作品详情，参考量未评估。"
        )

    def test_failed_detail_isolated_and_recovery_runs_once(self):
        calls = 0

        def search(**_):
            nonlocal calls
            calls += 1
            return [candidate("bad"), candidate("replacement")]

        pool = self.pool([], search_fn=search)
        recovery = pool.recover_failed_detail("bad")
        self.assertEqual(calls, 2)
        self.assertEqual(recovery["recovered_target_ids"], [1])
        self.assertEqual(
            pool.target_results[0]["visible_candidate_ids"], ["replacement"]
        )
        pool.recover_failed_detail("bad")
        self.assertEqual(calls, 2)
        self.assertEqual(pool.target_results[0]["candidate_count"], 2)

    def test_no_alternative_marks_detail_unavailable_without_search_missing(self):
        pool = self.pool(["bad"], search_fn=lambda **_: [candidate("bad")])
        original_verdict = pool.verdict
        pool.recover_failed_detail("bad")
        result = pool.target_results[0]
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["detail_access_status"], "unavailable")
        self.assertEqual(pool.verdict, original_verdict)
        self.assertEqual(
            pool.available_target_coverage()["unavailable_target_ids"], [1]
        )
        self.assertEqual(
            pool.reference_verdict, "候选详情不可用，参考量无法评估。"
        )


class DetailPoolAgentTests(unittest.TestCase):
    def test_mixed_loaded_and_unavailable_detail_is_fully_disclosed(self):
        def search(**query):
            return [
                candidate(
                    "good" if query["author"] == "甲" else "bad",
                    author=query["author"],
                )
            ]

        llm = FakeLLM(
            [
                init([{"author": "甲"}, {"author": "乙"}]),
                get("good"),
                get("bad"),
                finish("已取得一首详情，另一首资料当前不可用。"),
            ]
        )
        tool = Mock(
            side_effect=[
                detail("good", author="甲"),
                {"error": "not_found", "poem_id": "bad"},
                {"error": "not_found", "poem_id": "bad"},
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                side_effect=search,
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": tool},
                clear=True,
            ),
        ):
            result = run_agent("比较甲乙两首诗", llm)

        pool = result["candidate_pool"]
        coverage = pool["detail_pool"]["available_target_coverage"]
        self.assertEqual(coverage["status"], "all_covered")
        self.assertEqual(coverage["loaded_target_ids"], [1])
        self.assertEqual(coverage["unavailable_target_ids"], [2])
        self.assertEqual(
            [item["status"] for item in pool["profile"]["target_results"]],
            ["matched", "matched"],
        )
        self.assertTrue(pool["verdict"].startswith("全部命中"))
        self.assertEqual(
            pool["reference_verdict"],
            "全部剩余可用 targets 已取得详情；"
            "赏析与注释参考量均充足；另有候选详情不可用。",
        )
        self.assertFalse(result["degraded"])

    def test_prompt_distinguishes_search_missing_from_detail_failure(self):
        self.assertIn("筛选未命中", SYSTEM_INSTRUCTION)
        self.assertIn("本次详情读取异常", SYSTEM_INSTRUCTION)
        self.assertIn("不得宣称作品本身不存在或不在语料", SYSTEM_INSTRUCTION)
        self.assertIn(
            "初次 get_poem_detail 返回 not_found 会由系统内部自动重试",
            SYSTEM_INSTRUCTION,
        )
        self.assertNotIn(
            "get_poem_detail 返回 not_found 时，说明作品不在",
            SYSTEM_INSTRUCTION,
        )

    def test_uninitialized_detail_is_rejected_without_tool_call(self):
        llm = FakeLLM(
            [get("p"), init([{"author": "甲"}]), finish()]
        )
        tool = Mock()
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[candidate("p")],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": tool},
                clear=True,
            ),
        ):
            run_agent("测试", llm)
        tool.assert_not_called()
        self.assertIn("尚未初始化", llm.prompts[1])

    def test_hidden_id_rejected_then_legal_detail_succeeds(self):
        candidates = [candidate(f"p{i}") for i in range(6)]
        llm = FakeLLM(
            [init([{"author": "甲"}]), get("p5"), get("p0"), finish()]
        )
        tool = Mock(return_value=detail("p0"))
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=candidates,
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": tool},
                clear=True,
            ),
        ):
            result = run_agent("测试", llm)
        tool.assert_called_once_with(poem_id="p0")
        self.assertIn("当前合法 visible_candidate_ids", llm.prompts[2])
        self.assertEqual(result["candidate_pool"]["detail_pool"]["size"], 1)
        self.assertIn(
            "p5",
            result["candidate_pool"]["profile"]["target_results"][0][
                "visible_candidate_ids"
            ],
        )

    def test_not_found_retries_then_recovers_without_extra_llm_round(self):
        candidates = [candidate("bad"), candidate("good")]
        llm = FakeLLM(
            [init([{"author": "甲"}]), get("bad"), get("good"), finish()]
        )
        tool = Mock(
            side_effect=[
                {"error": "not_found", "poem_id": "bad"},
                {"error": "not_found", "poem_id": "bad"},
                detail("good"),
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=candidates,
            ) as search,
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": tool},
                clear=True,
            ),
        ):
            result = run_agent("测试", llm)
        self.assertEqual(tool.call_count, 3)
        self.assertEqual(search.call_count, 2)
        self.assertEqual(len(llm.prompts), 4)
        self.assertEqual(result["candidate_pool"]["detail_pool"]["size"], 1)
        target = result["candidate_pool"]["profile"]["target_results"][0]
        self.assertEqual(target["failed_candidate_ids"], ["bad"])

    def test_tool_exception_is_retried_once_then_propagates(self):
        llm = FakeLLM([init([{"author": "甲"}]), get("p")])
        tool = Mock(side_effect=RuntimeError("detail broken"))
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[candidate("p")],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": tool},
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "detail broken"):
                run_agent("测试", llm)
        self.assertEqual(tool.call_count, 2)

    def test_two_recovery_errors_force_finish(self):
        llm = FakeLLM(
            [
                init([{"author": "甲"}]),
                get("outside"),
                get("outside-again"),
                forced("强制收尾后的完整测试回答。"),
            ]
        )
        with patch(
            "poem_agent.candidate_pool.retrieve_all_poems",
            return_value=[candidate("p")],
        ):
            result = run_agent("测试", llm)
        self.assertEqual(len(llm.prompts), 4)
        self.assertIn("reference_verdict", llm.prompts[-1])
        self.assertFalse(result["degraded"])

    def test_one_recovery_keeps_all_six_productive_steps(self):
        authors = ["甲", "乙", "丙", "丁"]
        ids = ["p1", "p2", "p3", "p4"]

        def search(**query):
            index = authors.index(query["author"])
            return [candidate(ids[index], author=authors[index])]

        finish_answer = "四首作品详情均已取得，这是正常 finish 返回的回答。"
        llm = FakeLLM(
            [
                init([{"author": author} for author in authors]),
                get("outside"),
                *[get(poem_id) for poem_id in ids],
                finish(finish_answer),
            ]
        )
        tool = Mock(
            side_effect=[
                detail(poem_id, author=author)
                for poem_id, author in zip(ids, authors)
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                side_effect=search,
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": tool},
                clear=True,
            ),
        ):
            result = run_agent("读取四首诗", llm)

        self.assertEqual(len(llm.prompts), 7)
        self.assertEqual(tool.call_count, 4)
        self.assertEqual(result["answer"], finish_answer)
        self.assertEqual(result["candidate_pool"]["detail_pool"]["size"], 4)
        self.assertEqual(
            result["candidate_pool"]["detail_pool"][
                "available_target_coverage"
            ]["status"],
            "all_covered",
        )

    def test_six_productive_steps_then_force_finish_without_extra_action(self):
        ids = [f"p{i}" for i in range(5)]
        forced_answer = "正常额度耗尽后的强制收尾回答。"
        llm = FakeLLM(
            [
                init([{"author": "甲"}]),
                *[get(poem_id) for poem_id in ids],
                forced(forced_answer),
            ]
        )
        tool = Mock(side_effect=[detail(poem_id) for poem_id in ids])
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[candidate(poem_id) for poem_id in ids],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": tool},
                clear=True,
            ),
        ):
            result = run_agent("读取五首诗", llm)

        # 前六次是 Agent 动作；第七次直接使用强制收尾 Prompt，不再请求动作。
        self.assertEqual(len(llm.prompts), 7)
        self.assertIn("步数耗尽后的最终要求", llm.prompts[-1])
        self.assertEqual(tool.call_count, 5)
        self.assertEqual(result["answer"], forced_answer)
        self.assertEqual(result["candidate_pool"]["detail_pool"]["size"], 5)

    def test_total_step_guard_has_no_off_by_one(self):
        unknown = {"action": "unknown", "action_input": {}}
        forced_answer = "八轮动作硬上限后的强制收尾回答。"
        llm = FakeLLM([*[unknown for _ in range(8)], forced(forced_answer)])

        # 单独放宽两个较小熔断，只隔离验证总轮次 guard 本身。
        with (
            patch.object(agent_module, "MAX_PRODUCTIVE_STEPS", 99),
            patch.object(agent_module, "MAX_RECOVERY_STEPS", 99),
        ):
            result = run_agent("测试总硬上限", llm)

        self.assertEqual(len(llm.prompts), 9)
        self.assertNotIn("步数耗尽后的最终要求", llm.prompts[7])
        self.assertIn("步数耗尽后的最终要求", llm.prompts[8])
        self.assertEqual(result["answer"], forced_answer)


if __name__ == "__main__":
    unittest.main()
