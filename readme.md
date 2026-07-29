# poem-agent

一个可检索、可追溯、敢于说“不知道”的古诗文智能助手。

它不是让大模型直接凭记忆赏析诗词，而是先把用户请求解析成保留对象关系的 targets，在一个 Agent 步骤内初始化 Candidate Pool，再读取正文、注释与赏析并基于真实证据回答。回答中的解读引用可以绑定回具体语料块；遇到库外作品、错误前提或无效引用时，系统会标明缺失、诊断冲突、重试或诚实拒答。

## 已实现能力

- **统一检索**：显式作者/朝代/标题硬过滤、候选池内正文/赏析双路语义检索与标签软打分。
- **有状态 Candidate Pool**：一次提交 1–4 个 targets，批量执行初始查询与必要诊断，并在整个 Agent 运行中持续保存候选和画像。
- **客观搜索状态**：逐 target 计算 `matched`、`partial_match`、`conflict`、`missing` 或 `not_applicable`，由系统生成固定 verdict。
- **多步 Agent**：手写 `thought → action → observation` 循环，不依赖 Agent 框架，最多执行 6 步。
- **多诗任务**：作者、朝代、标题和主题的对象关系保存在同一 target，可在一次初始化中处理多首作品，再分别读取详情。
- **可核对引用**：解读使用 `[诗N-appr-x]` 或 `[诗N-note-x]`，最终映射到完整 `evidence_id` 和原始证据文本。
- **错误前提纠正**：用户给出的标题、作者或朝代与检索结果冲突时，基于已取回的详情纠正。
- **防空转与兜底**：拦截完全重复的工具调用；连续无有效结果或达到步数上限时停止继续检索。
- **答案质量闸门**：检测空答案或疑似截断，自动重试一次；悬空引用会携带具体反馈重新生成，仍失败则显式降级。
- **CLI 与 JSON 输出**：支持单次问答、交互模式、完整 JSON 结果以及可视化 Agent 轨迹。

## 工作流程

```text
用户问题
   │
   ▼
手写 Agent 循环
   ├── initialize_candidate_pool
   │      └── 批量主查询 / 条件诊断 ──► Chroma + poems.json
   ├── get_poem_detail ► 正文 / 注释 / 赏析
   └── finish
          │
          ▼
完整性检查 ──► 引用绑定 ──► 悬空引用修正 ──► 降级信息
          │
          ▼
最终回答 + evidence + Candidate Pool 精简快照
```

### 检索策略

1. Agent 先把用户自然语言一次解析为 1–4 个 targets。属于同一对象的 `author`、`dynasty`、`title` 和 `themes` 保持在同一 target；系统不会对作者列表和标题列表做笛卡尔积或按位置猜配对。
2. 每个 target 编译一个主查询。同一 target 的 themes 去重后以固定分隔符连接成一次语义 `query`；所有主查询在一次 `initialize_candidate_pool` 动作内执行。
3. 公开 Python 接口 `search_poems` 不从 `query` 猜取作者、朝代或标题。显式 `author`、`dynasty`、`title` 是精确硬条件，多个维度取交集，单次调用不会静默放宽。
4. 提供 `query` 时，`BAAI/bge-large-zh-v1.5` 在硬过滤候选内分别检索正文与逐段赏析，同一作品取两路最高余弦相似度；标签参与软打分，候选按 `0.6 × semantic_score + 0.4 × tag_score` 排序。
5. 标题复用 Unicode、空白与书名号归一化：存在精确标题时排除部分命中；完全没有精确命中时才使用部分标题匹配。
6. 不提供 `query` 时，不加载 embedding 或查询 Chroma；候选按 `poems.json` 原始顺序返回，且 `score` 为 `null`。提供 `query` 时，`score` 只表示同一 query 下的排序信号。
7. 组合硬条件的主查询为空时，系统最多执行一个预编译诊断：有标题就只保留标题；无标题但作者与朝代并存时只保留作者。诊断结果只用于形成 `conflict` 依据。
8. 池内部保存每个查询的完整候选并按真实 `poem_id` 全局去重，同时保留每个 target 的候选关联。Prompt 默认展示每个 target 的前 5 个轻量候选，最终 JSON 只保留这些候选的 ID；`size`、`author_dist` 和 `candidate_count` 仍按完整候选计算。

向量存储使用本地持久化的 Chroma，正文和赏析分别写入 `poem_content`、`poem_appreciation` 两个 collection。

### Candidate Pool 画像与引用

阶段 1 的 profile 包含全池 `size`、`author_dist`、逐项可追溯的 `target_results` 和固定为 `null` 的 `theme_coverage`。系统根据 target 状态生成固定 verdict，类型包括全部命中、部分满足、标题部分匹配、请求不符、未命中，以及“已取得主题排序候选，但主题覆盖待评估”。

Agent 只有在调用 `get_poem_detail` 后才能使用作品详情。每首取回的作品会获得稳定的会话编号，例如：

```text
[诗1-appr-0]  第一首诗的第 0 个赏析证据块
[诗2-note-3]  第二首诗的第 3 条注释
```

回答完成后，系统会把这些短引用绑定到 `poems.json` 中的完整证据 ID，并返回对应原文。非法格式会被清理；引用了未取详情的作品或不存在的段号时，系统会要求模型修正一次，仍无法修正则保留风险提示并标记 `degraded: true`。

旧的逐诗检索分数等级已移除。Candidate Pool 的 target 状态只描述结构化条件和检索结果能够客观证明的满足情况；阶段 1 不计算主题覆盖率，也不判断内容分析的最终可信性。

## 项目结构

```text
.
├── main.py                      # CLI、环境变量加载与 DeepSeek 客户端
├── data/
│   └── poems.json               # 结构化古诗文语料
├── scripts/
│   └── build_index.py           # 构建正文/赏析 Chroma 索引
├── poem_agent/
│   ├── candidate_pool.py            # targets、查询任务、完整候选、画像与 verdict
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
│   ├── trust.py                 # 完整性、引用绑定、悬空修正和降级检查
│   └── utils.py
├── .env.example
└── requirements.txt
```

工具层只负责参数校验和返回契约，检索算法位于 `retrieval/`，数据访问位于 `store.py`。目前暴露给模型的动作是：

| 动作 | 作用 |
| --- | --- |
| `initialize_candidate_pool(targets)` | 一次提交 1–4 个 targets；主循环在一个步骤内完成初始化 |
| `get_poem_detail(poem_id)` | 读取指定作品的正文、注释、赏析和来源 |

`search_poems(query=None, author=None, dynasty=None, title=None, top_k=5)` 仍是稳定的公开 Python 接口，可供 Candidate Pool 内部逻辑和直接调用者复用；它不再作为主 Agent 手动逐轮构建初始候选的动作。轻量候选字段仍固定为 `poem_id`、`title`、`author`、`dynasty`、`score`，`top_k` 合法范围仍为 1–20。

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

# 查看 Agent 每一步的决策、池初始化、工具调用和观察摘要
python main.py -v "比较《静夜思》和《春望》的情感"

# verbose 也可以通过环境变量开启
POEM_AGENT_VERBOSE=1 python main.py "赏析《蜀道难》"

# 输出 answer、evidence、candidate_pool、degraded 等完整 JSON
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
  "candidate_pool": {
    "targets": [
      {
        "target_id": 1,
        "author": "李白",
        "dynasty": null,
        "title": "静夜思",
        "themes": []
      }
    ],
    "profile": {
      "size": 1,
      "author_dist": {"李白": 1},
      "target_results": [
        {
          "target_id": 1,
          "target": {
            "target_id": 1,
            "author": "李白",
            "dynasty": null,
            "title": "静夜思",
            "themes": []
          },
          "status": "matched",
          "retrieval": "found",
          "candidate_count": 1,
          "visible_candidate_ids": ["作品ID"],
          "basis": null,
          "theme_coverage": null
        }
      ],
      "theme_coverage": null
    },
    "verdict": "全部命中：所有 target 的结构化条件均得到严格匹配。"
  },
  "degraded": false
}
```

不需要诗词检索而直接结束的兼容路径会返回 `candidate_pool: null`。`degraded` 是布尔值：正常结果为 `false`；答案完整性或引用问题在自动重试后仍未解决时为 `true`。

## 当前边界

- 知识范围受 `data/poems.json` 限制，库外作品不会凭模型记忆补全。
- 语义检索依赖本地 Chroma 索引和 BGE 模型；缺少索引时无法完成普通语义检索。
- 阶段 1 的 `theme_coverage` 固定为 `null`，不宣称主题已完整满足；参考量统计和最终分析可信性属于后续阶段。
- 引用检查可以验证引用是否真实存在并绑定原文，但不能替代人工判断某段证据是否足以支持全部表述。
