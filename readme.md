# poem-agent

一个可检索、可追溯、敢于说“不知道”的古诗文智能助手。

它不是让大模型直接凭记忆赏析诗词，而是先把用户请求解析成保留对象关系的 targets，在一个 Agent 步骤内初始化 Candidate Pool，再读取正文、注释与赏析并基于真实证据回答。回答中的解读引用可以绑定回具体语料块；遇到库外作品、错误前提或无效引用时，系统会标明缺失、诊断冲突、重试或诚实拒答。

## 版本状态

当前版本为 **v0.6**。Candidate Pool 的三个阶段已经完成：批量 targets 与搜索画像、筛选池与详情池及参考量画像、任务相关分析支撑与统一终检。完整测试共 82 项，当前全部通过。

## 已实现能力

- **统一检索**：显式作者/朝代/标题硬过滤、候选池内正文/赏析双路语义检索与标签软打分。
- **有状态 Candidate Pool**：一次提交 1–4 个 targets，批量执行初始查询与必要诊断，并在整个 Agent 运行中持续保存候选和画像。
- **筛选池与详情池**：完整轻量候选、排序和 target 关联永久保留；成功读取的唯一作品原子进入详情池，已读或隔离候选会推动默认 5 项窗口滚动。
- **客观搜索状态**：逐 target 计算 `matched`、`partial_match`、`conflict`、`missing` 或 `not_applicable`，由系统生成固定 verdict。
- **客观参考量画像**：按全语料赏析/注释数量的下四分位数（当前分别为 4/5），提供逐诗、逐 target、全池统计和独立 `reference_verdict`。
- **任务相关分析支撑**：模型只申报分析等级和 target 范围；系统从最终合法 evidence 反查实际作品，结合 Candidate Pool 计算不可突破的客观上限，稳定返回 `analysis_support`。
- **多步 Agent**：手写 `thought → action → observation` 循环，不依赖 Agent 框架；正常决策最多 6 步、恢复决策最多 2 步，总决策最多 8 步。
- **多诗任务**：作者、朝代、标题和主题的对象关系保存在同一 target，可在一次初始化中处理多首作品，再分别读取详情。
- **可核对引用**：解读使用 `[诗N-appr-x]` 或 `[诗N-note-x]`，最终映射到完整 `evidence_id` 和原始证据文本。
- **错误前提纠正**：用户给出的标题、作者或朝代与检索结果冲突时，基于已取回的详情纠正。
- **防空转与兜底**：拦截完全重复的工具调用；连续无有效结果或达到步数上限时停止继续检索。
- **统一最终检查**：一次收集空白/截断、悬空引用和分析支撑过度申报，最多合并重生成一次；仍有完整性或引用问题才设置 `degraded`。
- **CLI 与 JSON 输出**：支持单次问答、交互模式、完整 JSON 结果以及可视化 Agent 轨迹。

## 工作流程

```text
用户问题
   │
   ▼
手写 Agent 循环
   ├── initialize_candidate_pool
   │      └── 批量主查询 / 条件诊断 ──► Chroma + poems.json
   ├── get_poem_detail ► 可见 ID 校验 / 详情池 / 窗口滚动 / 参考画像
   │      └── not_found 重试 / 失败隔离 / 受控 target 重筛
   └── finish
          │
          ▼
统一终检：完整性 + 引用绑定 + analysis_support 客观上限
          └── 有问题时最多合并重生成一次
          │
          ▼
最终回答 + evidence + Candidate Pool 精简快照 + analysis_support
```

### 检索策略

1. Agent 先把用户自然语言一次解析为 1–4 个 targets。属于同一对象的 `author`、`dynasty`、`title` 和 `themes` 保持在同一 target；系统不会对作者列表和标题列表做笛卡尔积或按位置猜配对。
2. 每个 target 编译一个主查询。同一 target 的 themes 去重后以固定分隔符连接成一次语义 `query`；所有主查询在一次 `initialize_candidate_pool` 动作内执行。
3. 公开 Python 接口 `search_poems` 不从 `query` 猜取作者、朝代或标题。显式 `author`、`dynasty`、`title` 是精确硬条件，多个维度取交集，单次调用不会静默放宽。
4. 提供 `query` 时，`BAAI/bge-large-zh-v1.5` 在硬过滤候选内分别检索正文与逐段赏析，同一作品取两路最高余弦相似度；标签参与软打分，候选按 `0.6 × semantic_score + 0.4 × tag_score` 排序。
5. 标题复用 Unicode、空白与书名号归一化：存在精确标题时排除部分命中；完全没有精确命中时才使用部分标题匹配。
6. 不提供 `query` 时，不加载 embedding 或查询 Chroma；候选按 `poems.json` 原始顺序返回，且 `score` 为 `null`。提供 `query` 时，`score` 只表示同一 query 下的排序信号。
7. 组合硬条件的主查询为空时，系统最多执行一个预编译诊断：有标题就只保留标题；无标题但作者与朝代并存时只保留作者。诊断结果只用于形成 `conflict` 依据。
8. 池内部保存每个查询的完整候选并按真实 `poem_id` 全局去重，同时保留每个 target 的候选关联。Prompt 默认展示每个 target 排名最靠前的 5 个未读、未隔离轻量候选；详情成功或失败隔离后自动滚动补位。`size`、`author_dist` 和 `candidate_count` 始终按完整原始候选计算。

向量存储使用本地持久化的 Chroma，正文和赏析分别写入 `poem_content`、`poem_appreciation` 两个 collection。

### Candidate Pool 画像与引用

筛选池的 profile 包含全池 `size`、`author_dist`、逐项可追溯的 `target_results` 和固定为 `null` 的 `theme_coverage`。系统根据 target 状态生成固定搜索 verdict，类型包括全部命中、部分满足、标题部分匹配、请求不符、未命中，以及“已取得主题排序候选，但主题覆盖待评估”。

Agent 只能对当前 `visible_candidate_ids` 调用 `get_poem_detail`。成功详情以真实 `poem_id` 唯一保存，关联全部 targets 和主查询/诊断来源，并获得稳定的会话编号，例如：

```text
[诗1-appr-0]  第一首诗的第 0 个赏析证据块
[诗2-note-3]  第二首诗的第 3 条注释
```

回答完成后，系统会把这些短引用绑定到 `poems.json` 中的完整证据 ID，并返回对应原文。非法格式会被清理；引用了未取详情的作品或不存在的段号时，系统会要求模型修正一次，仍无法修正则保留风险提示并标记 `degraded: true`。

详情池分别统计赏析块和注释条目，正文不计数。逐诗以对应下四分位数为充足边界；target 与全池都严格使用 `sufficient_ratio > 0.6`，且池级以 targets 为同级单位汇总。`reference_verdict` 描述已读详情覆盖和参考数量，不覆盖搜索 verdict，也不代表观点正确性或最终分析可信性；参考量较少不会设置 `degraded`。

当前基线为赏析 4 个块、注释 5 条。两个阈值分别来自各自语料分布，表示“在同类资料中是否偏少”，不能横向比较，也不是赏析与注释的权重或可信度分数。赏析和注释会分别统计、分别披露；是否足以支持用户要求的具体分析，仍由最终 evidence、target 覆盖和 `analysis_support` 判断。

合法详情 `not_found` 会对同一 ID 自动重试一次；连续失败后隔离该 ID，并对每个关联 target 至多受控重筛一次。工具异常同样只重试一次，仍异常则直接向上抛出。

正常 `finish` 的 `action_input` 只允许 `answer` 和 `analysis_assessment`；后者只允许 `level` 与 `target_ids`。等级固定为 `not_applicable`、`sufficient`、`partial`、`insufficient`。系统不相信模型提交作品 ID，而是从最终非悬空 evidence 的真实 `poem_id` 反查详情池及 target 关联。`matched` 且有详情和合法依据可达到 sufficient；`partial_match`、`conflict`、未覆盖或详情不可用会限制上限；完全没有合法分析依据时为 insufficient。仅主题 target 不会因搜索层的 `not_applicable` 自动失去支撑资格。

参考量 `limited` 是透明披露的软因素，不会单独机械下调 sufficient。模型可以主动申报更保守的等级，系统只会下调过度申报、不会主动上调。partial/insufficient 的固定分析支撑说明会确定性出现在回答中。分析支撑不足本身不设置 `degraded`；该布尔值仍只表示答案完整性或引用安全降级。

四类结论保持独立：

| 结论 | 表达的事实 |
| --- | --- |
| `candidate_pool.verdict` | 筛选结果是否满足用户提交的 targets |
| `candidate_pool.reference_verdict` | 已读详情覆盖和赏析/注释参考数量 |
| 顶层 `analysis_support` | 最终合法依据对具体分析任务的支撑范围 |
| 顶层 `degraded` | 答案完整性或引用安全机制是否降级 |

## 项目结构

```text
.
├── main.py                      # CLI、环境变量加载与 DeepSeek 客户端
├── data/
│   └── poems.json               # 结构化古诗文语料
├── scripts/
│   └── build_index.py           # 构建正文/赏析 Chroma 索引
├── poem_agent/
│   ├── candidate_pool.py            # 筛选池、详情池、覆盖与两类客观画像
│   ├── analysis_support.py           # finish 协议、逐 target 支撑与客观上限
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
│   ├── trust.py                 # 完整性与引用绑定底层函数、降级提示
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

# 输出 answer、evidence、candidate_pool、analysis_support、degraded 等完整 JSON
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
  "answer": "《静夜思》的作者是李白。",
  "evidence": [],
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
    "verdict": "全部命中：所有 target 的结构化条件均得到严格匹配。",
    "detail_pool": {
      "size": 0,
      "items": [],
      "available_target_coverage": {
        "status": "none_loaded",
        "eligible_target_ids": [1],
        "loaded_target_ids": [],
        "unloaded_target_ids": [1],
        "unavailable_target_ids": [],
        "loaded_target_ratio": 0.0
      }
    },
    "reference_stats": {
      "baseline": {
        "method": "corpus_lower_quartile",
        "appreciation_threshold": 4,
        "annotation_threshold": 5,
        "aggregate_ratio_threshold": 0.6,
        "comparison": "strictly_greater"
      },
      "by_poem": [],
      "by_target": [],
      "overall": {}
    },
    "reference_verdict": "尚未读取作品详情，参考量未评估。"
  },
  "analysis_support": {
    "level": "not_applicable",
    "target_ids": [],
    "verdict": "本次请求不涉及内容分析。"
  },
  "degraded": false
}
```

不需要诗词检索而直接结束的兼容路径会返回 `candidate_pool: null`，同时稳定返回 `analysis_support.level: "not_applicable"`。`analysis_support` 始终只包含 `level`、`target_ids`、`verdict` 三个字段。`degraded` 是布尔值：正常结果为 `false`；答案完整性或引用问题在统一重生成后仍未解决时为 `true`。

## 当前边界

- 知识范围受 `data/poems.json` 限制，库外作品不会凭模型记忆补全。
- 语义检索依赖本地 Chroma 索引和 BGE 模型；缺少索引时无法完成普通语义检索。
- `theme_coverage` 仍固定为 `null`，不宣称系统已经计算主题覆盖；主题分析是否得到当前合法依据支撑由 `analysis_support` 单独表达。
- 引用检查可以验证引用是否真实存在并绑定原文，但不能替代人工判断某段证据是否足以支持全部表述。
- `analysis_support` 表示现有详情和合法依据对用户分析任务的覆盖范围，不是答案为真的概率、检索分数，也不表示文学解释唯一正确。
