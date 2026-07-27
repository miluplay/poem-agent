# poem-agent

一个"敢被追问"的古诗文智能助手。核心不是接一个大模型做赏析,而是让每一句解读都能溯源到可核对的原文与权威注释;能自主规划多步任务(检索、作者关联、跨文本对比),遇到库外或错误前提时优雅纠错、标注不确定、必要时拒答。

## 核心特性

- **可溯源**:每句解读绑定 `evidence_id`,指向具体的注释/赏析块,可核对出处。
- **防幻觉**:无据不答 —— 检索不到或置信度低时,标注不确定或拒答,不编造。
- **多步规划**:手写 agent 循环,支持检索 → 作者关联 → 跨文本对比等多步任务。

## 数据

- 来源:[chinese-gushiwen](https://github.com/aopao/chinese-gushiwen)(MIT),经筛选与结构化处理。(感谢🙏)
- 规模:精选 913 首,每首含正文、注释、译文、赏析、标签,赏析单段 ≤ 500 字。
- 处理:`build_poems.py` 从原始数据筛选并转换为统一的 PoemDetail 契约(见 `data/poems.json`)。
- 原始数据(`guwen/`)未包含在仓库中,请从上述来源下载后运行 `build_poems.py` 生成。

## 架构

- **数据层**:结构化诗词库(PoemDetail 契约)。
- **检索层**:混合检索 = 语义(embedding)+ 标签过滤。
- **Agent 编排层**:手写规划循环(工具调度、终止判断、降级)。
- **可信度层**:引用绑定 + 置信度判断 + 降级/拒答。

## 首次运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
python main.py "赏析《蜀道难》"
```

不带问题运行 `python main.py` 会进入交互模式；输入 `exit`、`quit` 或
`退出` 即可结束。默认调用 DeepSeek 的 `deepseek-v4-flash`；可在 `.env`
中通过 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL` 调整。

## 工具(对模型暴露)

| 工具 | 作用 |
| --- | --- |
| `search_poems` | 语义检索,返回候选 + 相似度分 |
| `filter_by_tag` | 按标签/作者/朝代结构化过滤 |
| `get_poem_detail` | 取单首
