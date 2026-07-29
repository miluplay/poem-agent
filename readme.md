# poem-agent

一个可检索、可追溯、敢于说“不知道”的古诗文智能助手。

它不是让大模型直接凭记忆赏析诗词，而是先从本地语料库检索作品，再读取正文、注释与赏析，最后基于真实证据生成回答。回答中的解读引用可以绑定回具体语料块；遇到库外作品、错误前提、低相关结果或无效引用时，系统会纠正、降低措辞强度、重试或诚实拒答。

## 已实现能力

- **统一检索**：显式作者/朝代/标题硬过滤、候选池内正文/赏析双路语义检索与标签软打分。
- **多步 Agent**：手写 `thought → action → observation` 循环，不依赖 Agent 框架，最多执行 6 步。
- **多诗任务**：可以连续检索、读取多首作品，并在同一回答中比较或综合分析。
- **可核对引用**：解读使用 `[诗N-appr-x]` 或 `[诗N-note-x]`，最终映射到完整 `evidence_id` 和原始证据文本。
- **错误前提纠正**：用户给出的标题、作者或朝代与检索结果冲突时，基于已取回的详情纠正。
- **防空转与兜底**：拦截完全重复的工具调用；连续无有效结果或达到步数上限时停止继续检索。
- **可信度控制**：按每首作品的最高检索分数计算 `normal`、`low_conf` 或 `no_hit`，低置信度回答使用审慎措辞。
- **答案质量闸门**：检测空答案或疑似截断，自动重试一次；悬空引用会携带具体反馈重新生成，仍失败则显式降级。
- **CLI 与 JSON 输出**：支持单次问答、交互模式、完整 JSON 结果以及可视化 Agent 轨迹。

## 工作流程

```text
用户问题
   │
   ▼
手写 Agent 循环
   ├── search_poems ──► 混合检索 ──► Chroma + poems.json
   ├── get_poem_detail ► 正文 / 注释 / 赏析
   └── finish
          │
          ▼
完整性检查 ──► 引用绑定 ──► 悬空引用修正 ──► 置信度与降级信息
          │
          ▼
     最终回答 + evidence
```

### 检索策略

1. 用户自然语言由 Agent 在检索前解析；`search_poems` 不从 `query` 猜取作者、朝代或标题。显式 `author`、`dynasty`、`title` 是精确硬条件，多个维度取交集，单次调用不会静默放宽。
2. 提供 `query` 时，`BAAI/bge-large-zh-v1.5` 在候选池内分别检索诗文正文与逐段赏析，同一作品取两路最高余弦相似度。
3. 意象、题材、情感等标签参与软打分，候选按 `0.6 × semantic_score + 0.4 × tag_score` 排序。
4. 标题复用 Unicode、空白与书名号归一化：存在精确标题时排除部分命中；完全没有精确命中时，才使用“输入标题是库内标题子串”的部分匹配。
5. 不提供 `query` 时，不加载 embedding 或查询 Chroma；候选按 `poems.json` 原始顺序返回，且 `score` 全为 `null`。提供 `query` 时，`score` 才表示语义与标签融合的软排序分。
6. 组合硬条件为空时，Agent 可诊断性放宽一次：有标题就保留标题并移除作者、朝代；无标题但作者与朝代并存时保留作者并移除朝代。放宽命中只用于发现冲突，Agent 必须读取详情后依据真实元信息纠正，不能宣称原请求已完整满足。

向量存储使用本地持久化的 Chroma，正文和赏析分别写入 `poem_content`、`poem_appreciation` 两个 collection。

### 可信度与引用

Agent 只有在调用 `get_poem_detail` 后才能使用作品详情。每首取回的作品会获得稳定的会话编号，例如：

```text
[诗1-appr-0]  第一首诗的第 0 个赏析证据块
[诗2-note-3]  第二首诗的第 3 条注释
```

回答完成后，系统会把这些短引用绑定到 `poems.json` 中的完整证据 ID，并返回对应原文。非法格式会被清理；引用了未取详情的作品或不存在的段号时，系统会要求模型修正一次，仍无法修正则保留风险提示并标记 `degraded: true`。

检索分数同时被转换为逐诗依据强度：

| 等级 | 分数 | 行为 |
| --- | ---: | --- |
| `normal` | `score >= 0.60` | 正常陈述 |
| `low_conf` | `0.35 <= score < 0.60` | 明确使用审慎措辞 |
| `no_hit` | `score < 0.35` 或没有有效命中 | 一般不应据此解读 |

## 项目结构

```text
.
├── main.py                      # CLI、环境变量加载与 DeepSeek 客户端
├── data/
│   └── poems.json               # 结构化古诗文语料
├── scripts/
│   └── build_index.py           # 构建正文/赏析 Chroma 索引
├── poem_agent/
│   ├── agent/
│   │   ├── __init__.py          # Agent 主循环、解析、防空转与强制收尾
│   │   ├── prompts.py           # 系统指令和 Prompt 构造
│   │   ├── observation.py       # 工具观察摘要与会话诗编号
│   │   └── display.py           # verbose 模式展示
│   ├── tools/
│   │   ├── search.py            # search_poems 工具契约
│   │   └── detail.py            # get_poem_detail 工具契约
│   ├── retrieval/
│   │   └── engine.py            # 标题、语义、标签混合检索
│   ├── store.py                 # poems.json 加载与基础查询
│   ├── trust.py                 # 完整性、引用、置信度和降级检查
│   └── utils.py
├── .env.example
└── requirements.txt
```

工具层只负责参数校验和返回契约，检索算法位于 `retrieval/`，数据访问位于 `store.py`。目前暴露给模型的工具为：

| 工具 | 作用 |
| --- | --- |
| `search_poems(query=None, author=None, dynasty=None, title=None, top_k=5)` | 结构化硬过滤后按可选语义意图排序；至少一个条件非空，`top_k` 为 1–20 |
| `get_poem_detail(poem_id)` | 读取指定作品的正文、注释、赏析和来源 |

`search_poems` 的轻量候选字段固定为 `poem_id`、`title`、`author`、`dynasty`、`score`。`author` 和 `dynasty` 使用去除两端空白后的精确匹配；`query` 仅用于主题、意象、情感、场景等软排序意图。

## 数据

- 来源：[chinese-gushiwen](https://github.com/aopao/chinese-gushiwen)（MIT），经筛选和结构化处理。
- 规模：913 篇作品、5,164 个赏析块、10,401 条注释。
- 朝代分布包括唐代 402 篇、宋代 247 篇、先秦 105 篇等。
- 每条记录都遵循统一的 `PoemDetail` 结构，包含 `poem_id`、标题、作者、朝代、正文、注释、译文、赏析、标签、音频地址和来源。
- 语料位于 `data/poems.json`；生成后的 `chroma/` 索引不提交到 Git。

数据仅供学习交流使用。

## 快速开始

项目使用了 `str | Path` 等现代类型语法，建议使用 Python 3.10 或更高版本。

```bash
# 1. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 DeepSeek
cp .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY

# 4. 首次运行前构建本地向量索引
# 首次执行会下载 BAAI/bge-large-zh-v1.5
python scripts/build_index.py

# 5. 提问
python main.py "赏析《蜀道难》"
```

> `chroma/` 不包含在仓库中。首次运行、修改语料或更换 embedding 模型后，需要重新执行索引脚本。

默认配置：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

也可以在命令行用 `--model` 和 `--base-url` 临时覆盖模型与服务地址。当前客户端调用 `/chat/completions` 接口，并使用 DeepSeek 的请求参数。

### 常用命令

```bash
# 交互模式；输入 exit、quit 或 退出 结束
python main.py

# 查看 Agent 每一步的决策、工具调用、观察摘要和置信度
python main.py -v "比较《静夜思》和《春望》的情感"

# verbose 也可以通过环境变量开启
POEM_AGENT_VERBOSE=1 python main.py "赏析《蜀道难》"

# 输出 answer、evidence、confidence、degraded 等完整 JSON
python main.py --json "赏析《蜀道难》"

# 调整请求超时
python main.py --timeout 120 "赏析《蜀道难》"
```

查看所有 CLI 选项：

```bash
python main.py --help
python scripts/build_index.py --help
```

索引脚本支持自定义数据路径、索引目录、模型、运行设备及批大小：

```bash
python scripts/build_index.py \
  --data data/poems.json \
  --chroma-dir chroma \
  --device cpu \
  --batch-size 32 \
  --write-batch-size 256
```

## 返回结果

默认终端输出回答正文，并在末尾列出本次实际使用的引用依据。使用 `--json` 时，返回结构的核心字段如下：

```json
{
  "answer": "基于语料生成并带有 [诗1-appr-0] 引用的回答",
  "evidence": [
    {
      "evidence_id": "作品ID#appr-0",
      "text": "对应的原始赏析文本",
      "poem_id": "作品ID",
      "title": "作品标题",
      "poem_number": 1
    }
  ],
  "confidence": {
    "confidence_table": {
      "1": {
        "poem_id": "作品ID",
        "title": "作品标题",
        "score": 1.0,
        "level": "normal"
      }
    },
    "overall_level": "normal"
  },
  "degraded": false
}
```

`degraded` 是布尔值：正常结果为 `false`；答案完整性或引用问题在自动重试后仍未解决时为 `true`。

## 当前边界

- 知识范围受 `data/poems.json` 限制，库外作品不会凭模型记忆补全。
- 语义检索依赖本地 Chroma 索引和 BGE 模型；缺少索引时无法完成普通语义检索。
- 置信度反映检索匹配强度，不等同于对回答事实正确率的统计保证。
- 引用检查可以验证引用是否真实存在并绑定原文，但不能替代人工判断某段证据是否足以支持全部表述。
