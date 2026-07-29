import io
import json
import unittest
from contextlib import redirect_stdout

from poem_agent.agent import force_finish, run_agent
from poem_agent.trust import (
    answer_integrity_gate,
    is_answer_suspiciously_incomplete,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return next(self.responses)


def finish_response(answer):
    return json.dumps(
        {
            "thought": "已有足够资料",
            "action": "finish",
            "action_input": {
                "answer": answer,
                "analysis_assessment": {
                    "level": "not_applicable",
                    "target_ids": [],
                },
            },
        },
        ensure_ascii=False,
    )


class AnswerIntegrityTest(unittest.TestCase):
    def test_incomplete_shape_detection(self):
        self.assertTrue(is_answer_suspiciously_incomplete(""))
        self.assertTrue(is_answer_suspiciously_incomplete(" \n "))
        self.assertTrue(
            is_answer_suspiciously_incomplete("  [诗2-appr-0] 后续内容")
        )
        self.assertTrue(is_answer_suspiciously_incomplete("一二三四五六七八九"))
        self.assertFalse(is_answer_suspiciously_incomplete("一二三四五六七八九十"))

    def test_normal_finish_retries_with_unified_feedback(self):
        llm = FakeLLM(
            [
                finish_response("[诗2-appr-0]"),
                finish_response("这是重试后生成的完整回答正文。"),
            ]
        )

        result = run_agent("请赏析", llm)

        self.assertEqual(result["answer"], "这是重试后生成的完整回答正文。")
        self.assertFalse(result["degraded"])
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("统一终检反馈", llm.prompts[1])

    def test_force_finish_retries_with_the_same_prompt(self):
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "answer": "太短",
                        "analysis_assessment": {
                            "level": "not_applicable",
                            "target_ids": [],
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "answer": "这是重试后生成的完整 force finish 回答。",
                        "analysis_assessment": {
                            "level": "not_applicable",
                            "target_ids": [],
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )

        result = force_finish("请赏析", llm, [], {})

        self.assertEqual(
            result["answer"], "这是重试后生成的完整 force finish 回答。"
        )
        self.assertFalse(result["degraded"])
        self.assertIn("统一终检反馈", llm.prompts[1])

    def test_second_failure_degrades_and_lists_collected_titles(self):
        trajectory = [
            {
                "action": "get_poem_detail",
                "input": {"poem_id": "p1"},
                "observation": {
                    "poem_id": "p1",
                    "title": "蜀道难",
                    "dynasty": "唐代",
                    "author": "李白",
                },
            },
            {
                "action": "get_poem_detail",
                "input": {"poem_id": "p2"},
                "observation": {
                    "poem_id": "p2",
                    "title": "静夜思",
                    "dynasty": "唐代",
                    "author": "李白",
                },
            },
        ]
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "answer": "",
                        "analysis_assessment": {
                            "level": "not_applicable",
                            "target_ids": [],
                        },
                    }
                ),
                json.dumps(
                    {
                        "answer": "[诗1-appr-0]",
                        "analysis_assessment": {
                            "level": "not_applicable",
                            "target_ids": [],
                        },
                    }
                ),
            ]
        )

        result = force_finish("请赏析", llm, trajectory, {1: "p1", 2: "p2"})

        self.assertTrue(result["degraded"])
        self.assertEqual(
            result["answer"],
            "生成回答时出现异常,已获取的资料涉及:《蜀道难》、《静夜思》。\n"
            "请重试,或追问具体某一首诗。",
        )
        self.assertEqual(result["evidence"], [])

    def test_verbose_reports_retry_and_degradation(self):
        output = io.StringIO()
        with redirect_stdout(output):
            answer, degraded = answer_integrity_gate(
                "",
                lambda: "仍短",
                [],
                verbose=True,
            )

        self.assertTrue(degraded)
        self.assertIn("生成回答时出现异常", answer)
        self.assertIn("[完整性检查] 检测到答案疑似截断,重试", output.getvalue())
        self.assertIn("[完整性检查] 检测到答案疑似截断,降级", output.getvalue())


if __name__ == "__main__":
    unittest.main()
