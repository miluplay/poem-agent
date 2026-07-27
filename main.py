"""poem-agent 命令行入口。

用法:
    python main.py "赏析《蜀道难》"
    python main.py                    # 进入交互模式
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

from poem_agent.agent import run_agent


class LLMError(RuntimeError):
    """LLM 请求失败。"""


class DeepSeekLLM:
    """满足 agent 所需 ``generate(prompt)`` 契约的 DeepSeek 客户端。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "thinking": {"type": "disabled"},
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            detail = ""
            if exc.response is not None:
                detail = f": {exc.response.text[:500]}"
            raise LLMError(f"LLM 请求失败（{exc}{detail}）") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("LLM 返回了无法识别的数据") from exc


def load_dotenv(path: Path = Path(".env")) -> None:
    """加载简单的 KEY=VALUE 配置，不覆盖终端中已设置的环境变量。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def build_llm(args: argparse.Namespace) -> DeepSeekLLM:
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise LLMError(
            "缺少 DEEPSEEK_API_KEY。请在 .env 中配置，"
            "例如 DEEPSEEK_API_KEY=你的密钥"
        )
    return DeepSeekLLM(
        api_key=api_key,
        model=args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url=args.base_url
        or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=args.timeout,
    )


def print_result(result: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n{result.get('answer', '（无回答）')}")
    evidence = result.get("evidence") or []
    if evidence:
        print("\n引用依据：")
        for item in evidence:
            evidence_id = item.get("evidence_id", "?")
            if item.get("dangling"):
                print(f"- [{evidence_id}] 无法绑定到语料，引用可能有误")
                continue
            title = item.get("title", "未知作品")
            print(f"- [{evidence_id}] 《{title}》：{item.get('text', '')}")


def ask(
    query: str, llm: DeepSeekLLM, *, as_json: bool, verbose: bool = False
) -> None:
    query = query.strip()
    if not query:
        return
    result = run_agent(query, llm, verbose=verbose)
    print_result(result, as_json=as_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可溯源的古诗文智能助手")
    parser.add_argument("query", nargs="*", help="要提问的内容；不填则进入交互模式")
    parser.add_argument("--model", help="模型名（默认读取 DEEPSEEK_MODEL）")
    parser.add_argument("--base-url", help="DeepSeek API 地址")
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=60, help="请求超时秒数")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="显示 Agent 每一步的思考、工具调用和观察摘要",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    try:
        llm = build_llm(args)
        if args.query:
            ask(" ".join(args.query), llm, as_json=args.json, verbose=args.verbose)
            return 0

        print("poem-agent 已启动。输入问题开始，输入 exit / quit / 退出 结束。")
        while True:
            try:
                query = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                return 0
            if query.lower() in {"exit", "quit"} or query == "退出":
                print("再见。")
                return 0
            try:
                ask(query, llm, as_json=args.json, verbose=args.verbose)
            except LLMError as exc:
                print(f"\n错误：{exc}", file=sys.stderr)
    except LLMError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
