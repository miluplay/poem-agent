import json
import unittest
from unittest.mock import patch

from poem_agent.agent import run_agent
from poem_agent.candidate_pool import (
    CandidatePool,
    CandidatePoolProtocolError,
    THEME_SEPARATOR,
    normalize_targets,
)


def candidate(
    poem_id: str,
    title: str,
    author: str = "作者",
    dynasty: str = "唐",
    score=None,
) -> dict:
    return {
        "poem_id": poem_id,
        "title": title,
        "author": author,
        "dynasty": dynasty,
        "score": score,
    }


def init_decision(targets, *, task_type="search") -> dict:
    normalized = [
        {
            "target_ref": f"t{index}",
            "author": target.get("author"),
            "dynasty": target.get("dynasty"),
            "title": target.get("title"),
            "themes": target.get("themes", []),
        }
        for index, target in enumerate(targets, start=1)
    ]
    return {
        "thought": "一次提交全部目标",
        "action": "initialize_candidate_pool",
        "action_input": {
            "targets": normalized,
            "tasks": [{
                "type": task_type,
                "target_refs": [item["target_ref"] for item in normalized],
                **(
                    {"aspects": [], "custom_aspects": []}
                    if task_type in {"appreciate", "compare"}
                    else {}
                ),
            }],
        },
    }


def finish_decision(answer="这是基于当前候选池生成的完整回答。") -> dict:
    return {
        "thought": "完成",
        "action": "finish",
        "action_input": {
            "answer": answer,
            "analysis_assessment": {
                "level": "not_applicable",
                "target_ids": [],
            },
        },
    }


class FakeLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(next(self.decisions), ensure_ascii=False)


class TargetValidationTests(unittest.TestCase):
    def test_types_unknown_fields_empty_and_theme_elements(self):
        invalid_cases = [
            None,
            {},
            ["not-an-object"],
            [{"author": 1}],
            [{"dynasty": []}],
            [{"title": True}],
            [{"themes": "月亮"}],
            [{"themes": ["月亮", " "]}],
            [{"themes": [1]}],
            [{"unknown": "x", "themes": ["月亮"]}],
            [{"author": " ", "dynasty": None, "title": " ", "themes": []}],
        ]
        for raw in invalid_cases:
            with self.subTest(raw=raw):
                with self.assertRaises(CandidatePoolProtocolError):
                    normalize_targets(raw)

    def test_normalization_deduplication_and_stable_target_ids(self):
        targets = normalize_targets(
            [
                {
                    "author": " 李白 ",
                    "dynasty": " ",
                    "title": "《 静 夜 思 》",
                    "themes": [" 月亮 ", "思乡", "月亮"],
                },
                {
                    "author": "李白",
                    "dynasty": None,
                    "title": "静夜思",
                    "themes": ["月亮", "思乡"],
                },
                {"author": "杜甫"},
            ]
        )
        self.assertEqual([target.target_id for target in targets], [1, 2])
        self.assertEqual(targets[0].author, "李白")
        self.assertIsNone(targets[0].dynasty)
        self.assertEqual(targets[0].title, "静夜思")
        self.assertEqual(targets[0].themes, ("月亮", "思乡"))
        self.assertEqual(targets[1].author, "杜甫")

    def test_one_to_four_after_deduplication_and_overflow_rejected(self):
        self.assertEqual(len(normalize_targets([{"author": "李白"}])), 1)
        self.assertEqual(
            len(normalize_targets([{"author": str(i)} for i in range(4)])),
            4,
        )
        with self.assertRaisesRegex(CandidatePoolProtocolError, "1–4"):
            normalize_targets([])
        with self.assertRaisesRegex(CandidatePoolProtocolError, "1–4"):
            normalize_targets([{"author": str(i)} for i in range(5)])

    def test_two_poems_keep_author_title_pairing(self):
        targets = normalize_targets(
            [
                {"author": "李白", "title": "静夜思"},
                {"author": "杜甫", "title": "春望"},
            ]
        )
        self.assertEqual(
            [(item.author, item.title) for item in targets],
            [("李白", "静夜思"), ("杜甫", "春望")],
        )


class CandidatePoolStateTests(unittest.TestCase):
    def test_themes_compile_once_and_separate_targets_make_separate_queries(self):
        calls = []

        def search(**query):
            calls.append(query)
            return []

        CandidatePool.initialize(
            [
                {"author": "李白", "themes": ["月亮", "思乡"]},
                {"author": "杜甫", "themes": ["月亮"]},
            ],
            search_fn=search,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["query"], f"月亮{THEME_SEPARATOR}思乡")
        self.assertEqual(calls[1]["query"], "月亮")

    def test_all_main_queries_execute_and_global_dedup_keeps_associations(self):
        calls = []
        shared = candidate("shared", "共诗", "甲")

        def search(**query):
            calls.append(query)
            return [shared, candidate(f"p{len(calls)}", "各诗", query["author"])]

        pool = CandidatePool.initialize(
            [{"author": "甲"}, {"author": "乙"}],
            search_fn=search,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(pool.profile["size"], 3)
        self.assertIn("shared", pool.target_candidate_ids[1])
        self.assertIn("shared", pool.target_candidate_ids[2])
        self.assertTrue(
            all(task.status == "completed" for task in pool.main_tasks.values())
        )

    def test_full_statistics_are_separate_from_default_visible_window(self):
        results = [candidate(f"p{i}", f"诗{i}", "甲") for i in range(7)]
        pool = CandidatePool.initialize(
            [{"author": "甲"}], search_fn=lambda **_: results
        )
        target_result = pool.profile["target_results"][0]
        self.assertEqual(pool.profile["size"], 7)
        self.assertEqual(pool.profile["author_dist"], {"甲": 7})
        self.assertEqual(target_result["candidate_count"], 7)
        self.assertEqual(target_result["visible_candidate_ids"], [f"p{i}" for i in range(5)])
        public_result = pool.public_snapshot()["profile"]["target_results"][0]
        model_result = pool.model_snapshot()["profile"]["target_results"][0]
        self.assertEqual(len(public_result["visible_candidate_ids"]), 5)
        self.assertNotIn("visible_candidates", public_result)
        self.assertEqual(
            [item["poem_id"] for item in model_result["visible_candidates"]],
            [f"p{i}" for i in range(5)],
        )

    def test_exact_partial_conflict_missing_and_theme_only_states(self):
        def search(**query):
            if query["title"] == "静夜思" and query["author"] == "李白":
                return [candidate("quiet", "静夜思", "李白")]
            if query["title"] == "月夜":
                return [candidate("partial", "月夜忆舍弟", query["author"] or "杜甫")]
            if query["title"] == "春望" and query["author"] == "李白":
                return []
            if query["title"] == "春望":
                return [candidate("spring", "春望", "杜甫")]
            if query["author"] == "李日":
                return []
            if query["query"] == "月亮":
                return [candidate("moon", "月诗", "甲", score=0.5)]
            return []

        cases = [
            (
                [{"author": "李白", "title": "静夜思"}],
                "matched",
                "全部命中",
            ),
            (
                [{"author": "杜甫", "title": "月夜"}],
                "partial_match",
                "标题部分匹配",
            ),
            (
                [{"author": "李白", "title": "春望"}],
                "conflict",
                "请求不符",
            ),
            (
                [{"author": "李日", "themes": ["月亮"]}],
                "missing",
                "未命中",
            ),
            (
                [{"themes": ["月亮"]}],
                "not_applicable",
                "已取得主题排序候选",
            ),
        ]
        for targets, expected, verdict_prefix in cases:
            with self.subTest(expected=expected):
                pool = CandidatePool.initialize(targets, search_fn=search)
                result = pool.profile["target_results"][0]
                self.assertEqual(result["status"], expected)
                self.assertIsNone(result["theme_coverage"])
                self.assertTrue(pool.verdict.startswith(verdict_prefix))

    def test_diagnostic_runs_only_after_empty_main(self):
        calls = []

        def search(**query):
            calls.append(query)
            if query["author"] == "李白":
                return []
            return [candidate("spring", "春望", "杜甫")]

        pool = CandidatePool.initialize(
            [{"author": "李白", "title": "春望"}],
            search_fn=search,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(pool.diagnostic_tasks[1].status, "completed")
        self.assertEqual(
            pool.profile["target_results"][0]["retrieval"], "found"
        )
        self.assertTrue(
            pool.profile["target_results"][0]["basis"][
                "diagnostic_conflicts"
            ]
        )

        calls.clear()
        matched = CandidatePool.initialize(
            [{"author": "杜甫", "title": "春望"}],
            search_fn=lambda **query: calls.append(query)
            or [candidate("spring", "春望", "杜甫")],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(matched.diagnostic_tasks[1].status, "skipped")

    def test_mixed_profile_and_fixed_verdict(self):
        def search(**query):
            if query["author"] == "李白":
                return [candidate("quiet", "静夜思", "李白")]
            return []

        pool = CandidatePool.initialize(
            [{"author": "李白"}, {"author": "李日"}],
            search_fn=search,
        )
        self.assertEqual(pool.profile["size"], 1)
        self.assertEqual(pool.profile["author_dist"], {"李白": 1})
        self.assertEqual(
            [item["status"] for item in pool.profile["target_results"]],
            ["matched", "missing"],
        )
        self.assertEqual(
            pool.verdict, "部分满足：部分 target 已命中，另有 target 未命中。"
        )

    def test_infrastructure_exception_propagates(self):
        def broken(**_):
            raise RuntimeError("chroma broken")

        with self.assertRaisesRegex(RuntimeError, "chroma broken"):
            CandidatePool.initialize([{"themes": ["月亮"]}], search_fn=broken)

    def test_author_dynasty_diagnostic_keeps_author_and_detects_conflict(self):
        calls = []

        def search(**query):
            calls.append(query)
            if query["dynasty"] == "宋":
                return []
            return [candidate("quiet", "静夜思", "李白", "唐")]

        pool = CandidatePool.initialize(
            [{"author": "李白", "dynasty": "宋"}],
            search_fn=search,
        )
        self.assertEqual(
            calls[1],
            {
                "query": None,
                "author": "李白",
                "dynasty": None,
                "title": None,
            },
        )
        self.assertEqual(pool.target_results[0]["status"], "conflict")


class CandidatePoolAgentTests(unittest.TestCase):
    def test_infrastructure_exception_terminates_agent_run(self):
        llm = FakeLLM([init_decision([{"themes": ["月亮"]}])])
        with patch(
            "poem_agent.candidate_pool.retrieve_all_poems",
            side_effect=RuntimeError("embedding unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
                run_agent("找月亮诗", llm)

    def test_protocol_error_can_be_corrected_and_snapshot_persists(self):
        llm = FakeLLM(
            [
                init_decision([{"author": 1}]),
                init_decision([{"author": "李白"}]),
                finish_decision(),
            ]
        )
        with patch(
            "poem_agent.candidate_pool.retrieve_all_poems",
            return_value=[candidate("quiet", "静夜思", "李白")],
        ):
            result = run_agent("找李白的诗", llm)

        self.assertIn("Candidate Pool 协议错误", llm.prompts[1])
        self.assertIn("当前 Candidate Pool", llm.prompts[2])
        self.assertEqual(
            result["candidate_pool"]["profile"]["target_results"][0]["status"],
            "matched",
        )
        self.assertNotIn("confidence", result)

    def test_repeated_successful_initialization_is_rejected_and_pool_unchanged(self):
        llm = FakeLLM(
            [
                init_decision([{"author": "李白"}]),
                init_decision([{"author": "杜甫"}]),
                finish_decision(),
            ]
        )
        with patch(
            "poem_agent.candidate_pool.retrieve_all_poems",
            return_value=[candidate("quiet", "静夜思", "李白")],
        ) as search:
            result = run_agent("找李白的诗", llm)

        search.assert_called_once()
        self.assertIn("本轮请求动作已经成功", llm.prompts[2])
        self.assertIn("禁止再次扩写、替换或增加 target", llm.prompts[2])
        self.assertEqual(
            result["candidate_pool"]["targets"][0]["author"], "李白"
        )

    def test_two_targets_initialize_in_one_step_then_get_details(self):
        quiet = candidate("quiet", "静夜思", "李白")
        spring = candidate("spring", "春望", "杜甫")

        def search(**query):
            return [quiet] if query["author"] == "李白" else [spring]

        details = {
            "quiet": {
                **quiet,
                "content": "床前明月光",
                "appreciation": [],
                "annotations": [],
            },
            "spring": {
                **spring,
                "content": "国破山河在",
                "appreciation": [],
                "annotations": [],
            },
        }
        llm = FakeLLM(
            [
                init_decision(
                    [
                        {"author": "李白", "title": "静夜思"},
                        {"author": "杜甫", "title": "春望"},
                    ],
                    task_type="compare",
                ),
                {
                    "thought": "取第一首详情",
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "quiet"},
                },
                {
                    "thought": "取第二首详情",
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "spring"},
                },
                finish_decision("《静夜思》和《春望》的资料均已取得。"),
            ]
        )
        with (
            patch("poem_agent.candidate_pool.retrieve_all_poems", side_effect=search),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": lambda poem_id: details[poem_id]},
                clear=True,
            ),
        ):
            result = run_agent("比较《静夜思》和《春望》", llm)

        self.assertIn('"target_id": 1', llm.prompts[1])
        self.assertIn('"target_id": 2', llm.prompts[1])
        self.assertIn("【诗1】", llm.prompts[2])
        self.assertIn("【诗2】", llm.prompts[3])
        self.assertEqual(len(result["candidate_pool"]["targets"]), 2)

    def test_missing_named_author_and_wrong_premise_are_explicit(self):
        def search(**query):
            if query["author"] == "李白" and query["title"] == "春望":
                return []
            if query["title"] == "春望":
                return [candidate("spring", "春望", "杜甫")]
            if query["author"] == "李白":
                return [candidate("moon", "月诗", "李白", score=0.6)]
            return []

        missing_pool = CandidatePool.initialize(
            [
                {"author": "李白", "themes": ["月亮"]},
                {"author": "李日", "themes": ["月亮"]},
            ],
            search_fn=search,
        )
        self.assertEqual(
            [item["status"] for item in missing_pool.target_results],
            ["matched", "missing"],
        )

        conflict_pool = CandidatePool.initialize(
            [{"author": "李白", "title": "春望"}], search_fn=search
        )
        result = conflict_pool.target_results[0]
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(
            result["basis"]["diagnostic_conflicts"][0]["differences"]["author"],
            {"expected": "李白", "actual": "杜甫"},
        )


if __name__ == "__main__":
    unittest.main()
