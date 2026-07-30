# poem-agent

一个可检索、可追溯、敢于说“不知道”的古诗文智能助手。

它不是让大模型直接凭记忆赏析诗词，而是先把用户请求解析成保留对象关系的完整请求，在 Session 中初始化或增量更新 Candidate Pool，再读取正文、注释与赏析并基于真实证据回答。回答中的解读引用可以绑定回具体语料块；遇到库外作品、错误前提或无效引用时，系统会标明缺失、诊断冲突、重试或诚实拒答。

## 版本状态

当前版本为 **v0.7**。在既有检索、Candidate Pool、引用和分析支撑机制上，新增了 CLI 进程内多轮 Session、完整请求更新、active/frozen target 切换、稳定诗号、详情缓存显式激活和有限历史。自动化测试使用确定性 FakeLLM，不会调用真实模型；本版本另已使用真实 DeepSeek 完成 target 修改、缓存唤醒和追加对比的连续多轮验收。

v0.7 最终验收包括 167 项自动测试、Python 编译检查和真实 DeepSeek 四轮连续对话。真实链路覆盖“赏析李白《静夜思》的月亮意象 → 改成杜甫写月亮的诗 → 回到《静夜思》 → 新增杜甫作品并比较情感”，验证了任务继承、target 冻结/唤醒、缓存详情激活、稳定诗号、逐 target 证据覆盖和引用绑定。

## 已实现能力

- **统一检索**：显式作者/朝代/标题硬过滤、候选池内正文/赏析双路语义检索与标签软打分。
- **结构化完整请求**：`targets + tasks` 同时表达 1–6 个 active 对象和 `search`、`read`、`appreciate`、`compare`、`verify` 五类任务；更新时提交合并后的完整意图。
- **进程内多轮 Session**：交互 CLI 的连续问题复用当前请求、Candidate Pool、稳定“诗N”、详情缓存和最近历史；退出进程后自然清空。
- **增量 Candidate Pool**：active targets 最多 6 个，frozen targets 最多 4 个；修正对象时冻结旧 target，回到旧对象时复用原 target ID，只有新增对象触发查询。
- **筛选池与详情池**：完整轻量候选、排序和 target 关联永久保留；成功读取的唯一作品原子进入详情池，已读或隔离候选会推动默认 5 项窗口滚动。
- **客观搜索状态**：逐 target 计算 `matched`、`partial_match`、`conflict`、`missing` 或 `not_applicable`，由系统生成固定 verdict。
- **客观参考量画像**：按全语料赏析/注释数量的下四分位数（当前分别为 4/5），提供逐诗、逐 target、全池统计和独立 `reference_verdict`。
- **任务相关分析支撑**：模型只申报分析等级和 target 范围；系统从最终合法 evidence 反查实际作品，结合 Candidate Pool 计算不可突破的客观上限，稳定返回 `analysis_support`。
- **动态决策预算**：手写 `thought → action → observation` 循环；每个 active target 提供 2 次详情额度，另有 2 次框架动作和 2 次恢复额度，随当前 active target 数动态计算。
- **多诗任务**：作者、朝代、标题和主题的对象关系保存在同一 target，可在一次初始化中处理多首作品，再分别读取详情。
- **稳定诗号与可核对引用**：同一 Session 中同一作品始终使用同一个“诗N”；解读使用 `[诗N-appr-x]` 或 `[诗N-note-x]`，最终映射到完整 `evidence_id` 和原始证据文本。
- **错误前提纠正**：用户给出的标题、作者或朝代与检索结果冲突时，基于已取回的详情纠正。
- **防空转与兜底**：拦截完全重复的工具调用；连续无有效结果或达到步数上限时停止继续检索。
- **统一最终检查**：一次收集空白/截断、悬空引用和分析支撑过度申报，最多合并重生成一次；仍有完整性或引用问题才设置 `degraded`。
- **有限历史与安全缓存**：Prompt 保留最近 8 个完整问答轮次且总计最多 16,000 字符；缓存摘要可定位旧作品，但本轮引用前必须显式调用详情动作激活。
- **CLI 与 JSON 输出**：支持临时单次问答、进程内多轮交互、完整 JSON 结果以及逐轮 Agent 轨迹。

## 工作流程

```text
用户问题 ──► 进程内 AgentSession
   │
   ▼
手写 Agent 循环
   ├── initialize_candidate_pool / update_candidate_pool
   │      └── 完整 targets + tasks / 增量查询 / 冻结或唤醒
   │          └── 批量主查询 / 条件诊断 ──► Chroma + poems.json
   ├── get_poem_detail ► 可见 ID 校验 / 详情池 / 窗口滚动 / 参考画像
   │      └── 本轮缓存激活 / not_found 重试 / 失败隔离 / 受控重筛
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

1. Agent 把当前完整意图解析为 1–6 个 active targets 和对应 tasks。属于同一对象的 `author`、`dynasty`、`title` 和 `themes` 保持在同一 target；系统不会对作者列表和标题列表做笛卡尔积或按位置猜配对。
2. 首轮通过 `initialize_candidate_pool` 提交完整请求；后续意图发生变化时通过 `update_candidate_pool` 提交合并后的完整版本。身份相同的 target 保留 ID，旧对象进入 frozen，新对象才执行查询，重新选择 frozen 对象时直接唤醒。
3. 公开 Python 接口 `search_poems` 不从 `query` 猜取作者、朝代或标题。显式 `author`、`dynasty`、`title` 是精确硬条件，多个维度取交集，单次调用不会静默放宽。
4. 提供 `query` 时，`BAAI/bge-large-zh-v1.5` 在硬过滤候选内分别检索正文与逐段赏析，同一作品取两路最高余弦相似度；标签参与软打分，候选按 `0.6 × semantic_score + 0.4 × tag_score` 排序。
5. 标题复用 Unicode、空白与书名号归一化：存在精确标题时排除部分命中；完全没有精确命中时才使用部分标题匹配。
6. 不提供 `query` 时，不加载 embedding 或查询 Chroma；候选按 `poems.json` 原始顺序返回，且 `score` 为 `null`。提供 `query` 时，`score` 只表示同一 query 下的排序信号。
7. 组合硬条件的主查询为空时，系统最多执行一个预编译诊断：有标题就只保留标题；无标题但作者与朝代并存时只保留作者。诊断结果只用于形成 `conflict` 依据。
8. 池内部保存每个查询的完整候选并按真实 `poem_id` 全局去重，同时保留每个 target 的候选关联。Prompt 只展示 active targets 及每个 target 排名最靠前的 5 个未读、未隔离轻量候选，不泄露 frozen targets；详情成功或失败隔离后自动滚动补位。`size`、`author_dist` 和 `candidate_count` 始终按完整原始候选计算。

向量存储使用本地持久化的 Chroma，正文和赏析分别写入 `poem_content`、`poem_appreciation` 两个 collection。

### v0.7 多轮控制框架

每个用户轮次先维护一份系统可验证的完整请求，而不是让模型用自由文本延续上下文。完整请求由 `targets + tasks` 组成：target 保存作者、朝代、标题和主题的对象关系，task 固定为 `search`、`read`、`appreciate`、`compare`、`verify` 五类，并携带受控 fields 或 aspects。Session 保存解析后的唯一总请求，后续修改必须提交合并后的完整版本。

请求进入 Candidate Pool 前会经过两层确定性守卫：

1. **任务语义守卫**：识别“赏析、对比、原文、核对、查找”等高置信度表达，防止模型把最终赏析任务误写成内部 `search`；未明确切换任务的 follow-up 会继承已有任务类型和分析角度。
2. **target 约束来源守卫**：用户未指定诗题或朝代时，模型不能根据常识自行增加硬条件；旧 target 未修改的字段可以原样继承。

一次 `initialize_candidate_pool` 或 `update_candidate_pool` 成功后，本轮请求解析阶段立即关闭，后续只允许读取详情或 finish，避免把同一句“再加一首”重复应用。`get_poem_detail` 的授权始终只依据真实 `poem_id` 与系统内部 active/cache 状态；模型偶发附加的 `target_id`、`target_ids`、`fields` 会被忽略，未知字段仍被拒绝。

对于 `appreciate/compare`，系统逐轮生成按 active target 关联的详情覆盖清单，分别列出：

- 本轮是否已有可产生 evidence 的详情；
- 可以显式激活的 Session 缓存 poem IDs；
- 当前可以读取的 visible candidate poem IDs。

存在未覆盖 target 时，Agent 优先补齐该对象，不能连续为已覆盖对象读取额外作品。`sufficient/partial` finish 也必须先覆盖申报范围；失败提示会按 target 返回可操作的缓存和候选 ID。所有分析 targets 覆盖后，仍可在既定详情预算内补充候选。

Session 本身只保存在当前 CLI 进程内，包括：

- 当前 resolved request 和 active/frozen Candidate Pool；
- 最多 6 个 active、4 个 frozen targets；
- 真实 poem ID 到稳定“诗N”的单调编号；
- 已读详情缓存；
- 最近 8 个完整轮次、总计最多 16,000 字符的历史。

历史裁剪只删除最旧的完整问答轮次，不回滚请求、Pool、编号或缓存；异常轮次不写入历史，但异常前已成功原子提交的 Session 状态继续保留。

### Candidate Pool 画像与引用

筛选池的 profile 包含全池 `size`、`author_dist`、逐项可追溯的 `target_results` 和固定为 `null` 的 `theme_coverage`。系统根据 target 状态生成固定搜索 verdict，类型包括全部命中、部分满足、标题部分匹配、请求不符、未命中，以及“已取得主题排序候选，但主题覆盖待评估”。

Agent 只能对当前 `visible_candidate_ids` 调用 `get_poem_detail`，或显式调用 Session 缓存摘要中的真实 `poem_id` 来激活旧详情。成功详情以真实 `poem_id` 唯一保存，并按当前 active targets 动态关联主查询/诊断来源；作品第一次读取时获得稳定的会话编号，例如：

```text
[诗1-appr-0]  第一首诗的第 0 个赏析证据块
[诗2-note-3]  第二首诗的第 3 条注释
```

回答完成后，系统会把这些短引用绑定到 `poems.json` 中的完整证据 ID，并返回对应原文。非法格式会被清理；引用了未取详情的作品或不存在的段号时，系统会要求模型修正一次，仍无法修正则保留风险提示并标记 `degraded: true`。

旧轮读取过的完整详情不会直接泄露到新一轮 Prompt。Prompt 只提供作品编号、真实 ID、标题、作者和朝代摘要；模型需要分析或引用旧作品时必须显式调用 `get_poem_detail`，缓存命中不会再次执行真实详情工具。公开 JSON 可以包含 `frozen_targets` 作为调试信息，但模型只看到 active Pool/current request。

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
│   ├── request.py                    # 完整 targets + tasks 请求协议与固定句式
│   ├── session.py                    # 进程内请求、Pool、诗号、缓存与有限历史
│   ├── candidate_pool.py             # active/frozen 筛选池、详情池与客观画像
│   ├── analysis_support.py           # finish 协议、逐 target 支撑与客观上限
│   ├── agent/
│   │   ├── __init__.py          # Agent 主循环、请求阶段、详情覆盖与统一收尾
│   │   ├── prompts.py           # 多轮状态、覆盖清单和系统指令构造
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
| `initialize_candidate_pool(targets, tasks)` | 首轮一次提交完整请求；Agent v0.7 支持 1–6 个 active targets |
| `update_candidate_pool(targets, tasks)` | 后续轮次提交合并后的完整请求；保留、冻结、唤醒或新增 targets |
| `get_poem_detail(poem_id)` | 读取指定作品的正文、注释、赏析和来源；模型偶发附带的 `target_id`、`target_ids`、`fields` 会被安全忽略，授权始终只看真实 poem ID 与系统内部状态 |

`search_poems(query=None, author=None, dynasty=None, title=None, top_k=5)` 仍是稳定的公开 Python 接口，可供 Candidate Pool 内部逻辑和直接调用者复用；它不作为主 Agent 手动逐轮构建候选的动作。轻量候选字段仍固定为 `poem_id`、`title`、`author`、`dynasty`、`score`，`top_k` 合法范围仍为 1–20。

兼容接口 `CandidatePool.initialize(raw_targets)` 仍保留 1–4 个 targets 上限；Agent v0.7 使用的是完整请求入口，不受该 legacy 上限约束。

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
# 交互模式；同一进程内连续问题复用 Session
# 输入 exit、quit 或 退出 结束，退出后状态清空
python main.py

# 查看 Agent 每一步的决策、池初始化、工具调用和观察摘要
python main.py -v "比较《静夜思》和《春望》的情感"

# verbose 也可以通过环境变量开启
POEM_AGENT_VERBOSE=1 python main.py "赏析《蜀道难》"

# 输出 answer、evidence、candidate_pool、analysis_support、degraded 等完整 JSON
python main.py --json "赏析《蜀道难》"

# 交互模式也可逐轮输出公开 JSON；不会额外暴露 Session 或 history
python main.py --json

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
    "frozen_targets": [],
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
- Session 只存在于当前 CLI 进程内；退出后不会持久化或恢复，也没有账户、外部 session ID、reset 命令、Web/API 多会话或并发隔离。
- 历史只做最近 8 轮和 16,000 字符的整轮裁剪，没有复杂摘要；旧轮完整详情必须在新一轮显式激活后才能用于引用。
- `theme_coverage` 仍固定为 `null`，不宣称系统已经计算主题覆盖；主题分析是否得到当前合法依据支撑由 `analysis_support` 单独表达。
- 引用检查可以验证引用是否真实存在并绑定原文，但不能替代人工判断某段证据是否足以支持全部表述。
- `analysis_support` 表示现有详情和合法依据对用户分析任务的覆盖范围，不是答案为真的概率、检索分数，也不表示文学解释唯一正确。
- 自动测试已覆盖 target 修正、frozen 唤醒、追加对比、跨轮“诗N”和冲突澄清接续，真实 DeepSeek 也已通过对应的连续多轮验收；模型决策仍具有非确定性，个别运行可能先触发可恢复守卫，或在预算内读取未用于最终回答的额外候选。
