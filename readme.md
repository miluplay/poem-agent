# poem-agent

一个"敢被追问"的古诗文智能助手。核心不是接一个大模型做赏析,而是让每一句解读都能溯源到可核对的原文与权威注释;能自主规划多步任务(检索、作者关联、跨文本对比),遇到库外或错误前提时优雅纠错、标注不确定、必要时拒答。

## 核心特性

- **可溯源**:每句解读绑定 `evidence_id`,指向具体的注释/赏析块,可核对出处。悬空引用(引用了未检索到的内容)会被检测。
- **防幻觉**:无据不答 —— 检索不到或置信度低时,标注不确定或拒答,不编造。
- **多步规划**:手写 Agent 循环(不套框架),可控地处理工具调度、终止判断、错误自我修正与降级。

## 架构

- **数据层**:结构化诗词库(PoemDetail 契约),913 首,每首含正文、注释、译文、赏析、标签。
- **检索层**:混合检索 = 标题匹配 + 语义检索 + 标签软硬过滤。
  - 语义:用 `bge-large-zh-v1.5` 对**正文**和**赏析(逐段)**分别 embedding,查询时两路取 max 融合。
  - 标签:作者/朝代走硬过滤,意象/题材/情感走软打分。
  - 融合:`score = 0.6 × 语义 + 0.4 × 标签`,精确标题命中置顶。
  - 向量库:Chroma,本地持久化。
- **Agent 编排层**:手写规划循环(工具调度、终止判断、降级)。
- **可信度层**:引用绑定 + 悬空引用检测 + 降级/拒答。

## 数据

- 来源:[chinese-gushiwen](https://github.com/aopao/chinese-gushiwen)(MIT),经筛选与结构化处理。(感谢🙏)
- 规模:精选 913 首(唐诗 402 / 宋词 101 / 文言文 59 等),每首含正文、注释、译文、赏析、标签,赏析单段 ≤ 500 字。
- 处理:`build_poems.py` 从原始数据筛选并转换为统一的 PoemDetail 契约(见 `data/poems.json`)。
- 原始数据(`guwen/`)未包含在仓库中,如需重建 `poems.json`,请从上述来源下载后运行 `build_poems.py`。

## 首次运行

```bash
# 1. 环境与依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API key
cp .env.example .env
# 编辑 .env,填入 DEEPSEEK_API_KEY

# 3. 构建向量索引(首次必须,会下载 bge-large-zh-v1.5,约 1.3GB)
python build_index.py

# 4. 运行
python main.py "赏析《蜀道难》"
```

> 索引数据(`chroma/`)不在仓库中,首次运行前必须先执行 `build_index.py` 生成,否则检索会失败。

不带问题运行 `python main.py` 会进入交互模式;输入 `exit`、`quit` 或 `退出` 即可结束。默认调用 DeepSeek 的 `deepseek-v4-flash`;可在 `.env` 中通过 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL` 调整。Agent 架构与具体 LLM 解耦,任何 OpenAI 兼容接口均可接入。

调试或演示 Agent 轨迹时，可加 `-v` / `--verbose`，也可设置环境变量
`POEM_AGENT_VERBOSE=1`。默认关闭，开启后会逐步显示 thought、工具调用参数和
观测摘要，最终答案与引用依据的输出格式保持不变。

## 工具

对模型暴露的工具:

| 工具 | 作用 | 状态 |
| --- | --- | --- |
| `search_poems` | 混合检索:标题匹配 + 语义(正文/赏析双向量 max 融合)+ 标签软硬过滤 | ✅ 已实现 |
| `get_poem_detail` | 取单首的正文/注释/译文/赏析 | ✅ 已实现 |
| `get_author_works` | 取某作者的作品列表 | 🚧 规划中 |

系统内部(不对模型暴露):`verify_claim` —— finish 后对关键论断做事后校验,不一致则回退或降级。

## 状态

开发中。已完成:数据管道、手写 Agent 循环(诗→赏析,含引用绑定与无据不答)、混合检索索引。进行中:混合检索接入循环、多步规划(A 线)、降级机制(B 线)。

---

*数据仅供学习交流使用。*
