# 计划二/计划三(全局共现路径索引 + 问题分解策略路由)实现与评测报告

日期：2026-06-15
对应设计：`reports/multihop_reasoning_implementation_plan.md` 的「实现计划二 / Milestone 2」
与「实现计划三 / Milestone 3」
前序：`reports/multihop_plan1_implementation_report.md`(两阶段桥接检索 MVP)

## 1. 摘要

在计划一(`multihop_bridge_query()` 按需桥接)之上，本次落地两件事：

- **Milestone 2**：把「从第一跳 chunk 局部抽 bridge」固化为**全局语义共现索引**
  `sem_cooccur_index` + 基于该图的 **`multihop_path_query()`** beam 检索。bridge 不再
  局限于第一跳召回 chunk 的 `sem_node_list`，而是沿全局共现邻接从 query 语义节点向外扩展，
  并对「被多个 query 约束共同支持」的 bridge 抬升 prior(计划 §5.5 的 University of Kansas 型)。
- **Milestone 3**：`plan_query()` 规则化问题分解 + **策略路由** `multihop_query()`，按问题类型
  分流到 comparison-focused / bridge-path / 单跳三条策略，**专门修复计划一遗留的 comparison
  MRR 回退**。

**不改索引结构、不改 finalize / semantic node 构建、不改 pickle schema**：
`sem_cooccur_index` 为运行时派生结构，按需构建、`__getstate__` 落盘时剥离、加载后重建
(计划 §5.7)。直接加载既有 fastidx 缓存评测，无需重建索引。

HotpotQA distractor 500 样本(同计划一口径，`chunk_cooccur_query()` 为单跳基线)：

| 分层 | 指标 | baseline | bridge (M1) | **path (M2)** | **routed (M3)** |
| --- | --- | --- | --- | --- | --- |
| **all** (n=500) | full_hit | 266 (0.532) | 319 (0.638) | 334 (0.668) | **336 (0.672)** |
| | recall@10 | 0.734 | 0.784 | 0.808 | **0.808** |
| | mrr@10 | 0.711 | 0.704 | 0.692 | **0.728** |
| **bridge** (n=404) | full_hit | 206 (0.510) | 259 (0.641) | **272 (0.673)** | 269 (0.666) |
| | recall@10 | 0.719 | 0.785 | **0.808** | 0.802 |
| | mrr@10 | 0.718 | 0.728 | 0.705 | 0.700 |
| **comparison** (n=96) | full_hit | 60 (0.625) | 60 (0.625) | 62 (0.646) | **67 (0.698)** |
| | recall@10 | 0.797 | 0.781 | 0.807 | **0.833** |
| | mrr@10 | 0.684 | 0.604 | 0.639 | **0.845** |

要点：

- **M2(path)** 在 bridge 分层把 full_hit 从 259 → **272**、recall 0.785 → **0.808**，all full_hit
  319 → **334**；全局图扩展比 chunk 局部抽更能找到第二个 gold。
- **M3(routed)** 是 comparison 回退的解药：comparison mrr **0.604(M1) → 0.845**(比 baseline
  0.684 还高 **+0.161**)、full_hit 60 → **67**、recall 0.781 → **0.833**；同时 all mrr 回到
  **0.728**(≥ baseline 0.711，彻底消除计划一 all mrr 的 −0.007)。
- routed 的 bridge 分层走 path 策略，full_hit/recall 显著高于 baseline；bridge mrr 0.700 略低于
  baseline 0.718(path 聚合的固有取舍)，但 all 维度 mrr 已净增。

**验收**(计划 §8 Milestone 2/3)：bridge/comparison 两分层 full_hit、recall 全部不低于 baseline；
comparison 不再回退且大幅改善；debug 输出能说明所采用的 strategy 与 plan。达成。

### 1.1 平均查询耗时（500 样本，单进程，GPU 常驻，含 query 编码）

| 方法 | 平均每条 query | 相对基线 | 备注 |
| --- | --- | --- | --- |
| baseline `chunk_cooccur_query()` | 87.6 ms | — | |
| bridge `multihop_bridge_query()` (M1) | 130.8 ms | +43 ms | chunk 局部抽 bridge |
| **path `multihop_path_query()`** (M2) | **83.6 ms** | −4 ms | 图邻接查表比遍历首跳 sem 列表更省 |
| **routed `multihop_query()`** (M3) | 146.6 ms | +59 ms | 含 `plan_query()` 一次额外 query 解析(**M4 已优化至 ~84 ms,见 M4 报告 §3.4**) |

`sem_cooccur_index` 一次性构建耗时 **15.7 s**(59,211 节点 / 794,189 边)，构建后所有
path/routed 查询共用，**不计入** per-query 时延。routed 的额外开销主要来自 `plan_query()` 复用了一次
`_resolve_query_matches()`(GPU 编码)与策略函数内部的 `chunk_cooccur_query()` 各编码一次；
后续可让路由器仅做一次首跳召回再分流以摊薄(见 §6)。原始数值见
`reports/multihop_m2m3_eval_results.json`。

## 2. 代码落点

延续计划一风格：所有逻辑放 `multihop_reasoning.py`，`RAG_graph.py` 仅加薄 wrapper。

- **`multihop_reasoning.py`**（在计划一之后追加）
  - *Milestone 2*
    - `build_sem_cooccur_index()` —— 全局共现邻接构建(一次性、可缓存、`force=` 重建)
    - `get_sem_neighbors()` —— 邻接访问器(首次调用惰性构建)
    - `extract_bridge_candidates_from_path_index()` —— 从图邻接发现 bridge(含 multi-constraint 支持)
    - `multihop_path_query()` —— 对外主入口(复用计划一的 `rank_evidence_chains` / `aggregate_chunk_ranking`)
    - 辅助：`_occurrence_sentence_map()`、`_best_shared_chunk()`、`_prepare_query_and_caches()`
  - *Milestone 3*
    - `QuestionPlan` dataclass、`plan_query()` —— 规则化 comparison/bridge/single 判定
    - `comparison_focused_query()` —— 两 side 并列检索 + 标题页优先 + 轮转交错
    - `multihop_query()` —— 统一入口，按 plan 路由
    - 辅助：`_anchor_sems_from_matches()`
- **`RAG_graph.py`**：`build_sem_cooccur_index` / `multihop_path_query` / `plan_query` /
  `comparison_focused_query` / `multihop_query` 五个薄 wrapper；`__init__` 新增运行时字段
  `self.sem_cooccur_index=None`；`_ensure_backward_compatible_attrs()` 对旧 pickle 默认补 `None`；
  `__getstate__()` 落盘时剥离 `sem_cooccur_index` 及 `_multihop_*` 惰性缓存(不持久化，加载后重建)。

评测脚本(`archive/eval_scripts/`，run-as-script)：
- `eval_hotpot_multihop_m2m3.py` —— 500 样本 baseline / bridge / path / routed 四方法分层评测
  (结果写 `reports/multihop_m2m3_eval_results.json`)

`chunk_cooccur_query()` 及计划一 `multihop_bridge_query()` 完全不受影响。

## 3. Milestone 2：全局 sem_cooccur_index + 路径检索

### 3.1 数据结构（计划 §5.2）

稀疏对称邻接，键为 `sem_node_id`(即 `db.sem_nodes` 下标)：

```python
sem_cooccur_index[a] = {
    b: {"weight": float,             # shared / sqrt(df_a*df_b) * same-sentence boost
        "shared_chunk_ids": [int],   # 上限 32 条样本，用于回选 hop1 代表 chunk
        "same_sentence_count": int,
        "chunk_count": int}          # 实际共享 chunk 总数
}
```

### 3.2 构建（计划 §5.3）

```
遍历 ChunkNode.sem_node_list:
  ├─ 过滤泛词/hub：df > 10%·N 丢弃；非短语/非实体且 idf<2.0 丢弃(短语/实体豁免 idf 门槛)
  ├─ 每 chunk 上限 max_nodes_per_chunk=80(按 实体/短语 + idf 取 top，防 O(n²) 爆炸)
  ├─ 两两累计 shared_count / same_sentence_count(用一次性 (chunk_id,sem_id)->句子集 表 O(1) 判同句)
  └─ weight = shared/sqrt(df_a·df_b) × (1 + 0.5·same_sent/shared)；每节点保留 top_neighbors=30
```

同句计数借助 `_occurrence_sentence_map()` 一次遍历所有 sem 的 `span_occurrences`，把
`(chunk_id, sem_id) → {sentence_id}` 预表化，避免逐对重扫 occurrence。

500 文档 fastidx：59,211 节点、794,189 条边、构建 15.7 s。运行时结构，**不进 pickle**。

### 3.3 路径桥接发现（计划 §5.4–5.5）

`multihop_path_query()` 复用首跳 `chunk_cooccur_query()` 拿 `first_hop_score_map` 和 query 语义节点；
bridge 候选改由 `extract_bridge_candidates_from_path_index()` 从**全局图**产生：

- 对每个 query 语义节点取其 `get_sem_neighbors()`；邻居跨多个 query 节点聚合。
- 沿用计划一的 gating(排除 query 命中、低 idf 单词、hub)与类型/标题系数。
- 新增两路 path 信号叠加到 `BridgePrior`：
  - `1 + 0.5·(归一化边权)`：图上越强共现的 bridge 越可信；
  - `1 + 0.30·(支持它的 query 约束数 − 1)`：**constraint_supported_bridge**，实现计划 §5.5
    「多个属性约束共同定位中间实体」(University of Kansas 由 Lawrence + Kansas City 共同支持)。
- `hop1` 代表 chunk 取该边 `shared_chunk_ids` 中 `first_hop_score` 最高者，确保 hop1 证据落在真正被
  首跳召回的 anchor chunk 上。

ChainScore、第二跳「同名页面注入」、聚合权重沿用计划一(§4 评分、§6 注入、§7 消融默认
`h1=0.5,h2=1.0,min=0.2`)，故 path 与 bridge 评分口径可比、可对照。

## 4. Milestone 3：问题分解 + 策略路由

### 4.1 `plan_query()`（计划 §6.2，规则、无 LLM）

输出 `QuestionPlan(mode, anchor_units, answer_focus, cues, reason)`：

- **comparison**：命中比较线索词(`same/both/older/.../either/neither/...` 或两 anchor 间 `A or B`)
  **且** ≥2 个不同 anchor 实体 **且** 句首是 `Are/Were/Is/Was/Do/Does/Did/Which/Who/...`。
- **bridge**：存在 anchor 实体但无比较信号(默认多跳路径)。
- **single**：无强 anchor 实体，回退单跳。

anchor 由 `_anchor_sems_from_matches()` 从 `resolved_matches` 的 exact 命中里取实体/专名/短语，按 idf 降序。

### 4.2 `comparison_focused_query()`（计划 §6.3，修复回退的核心）

comparison **不做桥接扩展**(桥接会把 director/producer 这类共现职业词当 bridge，反噬排序)。
改为两 side 并列检索 + 轮转交错：

1. 对每个 anchor side，取其自身 `SemNode.BM25` 排序的 chunk 列表；
2. **标题页优先**：若 anchor 规范名等于某文档标题，把该「实体自己的页面」强制提到该 side 列表首位
   —— 直接命中 comparison 属性证据，并压过同名干扰页(如 `Ed Wood` 人物页 vs `Ed Wood (film)`)；
3. 跨 side **轮转交错**(round-robin)，让每个 side 的最佳 chunk 都排进 top；
4. 末尾补 `chunk_cooccur_query()` 排序填满剩余槽位。

返回 `chains=None`(comparison 是并列证据，不构 A→bridge→B 链)。这套交错正是 comparison
full_hit/recall/mrr 三项同时跳升的来源(两个 side 的 gold 几乎稳进 top-2)。

### 4.3 `multihop_query()` 路由

`plan_query()` 定策略 → comparison 走 `comparison_focused_query()`、bridge 走
`multihop_path_query()`(`use_path_index=False` 可切回计划一)、single 走 `chunk_cooccur_query()`；
`debug_info["strategy"]` 与 `["plan"]` 记录决策。bridge 路径在无良好 bridge 时 chain 为空、聚合
退化为首跳排序 ≈ baseline，故非 comparison 误判到 bridge 也安全。

## 5. 复现方式

```bash
# M2/M3 四方法分层评测(自动加载 fastidx 缓存，首查构建 sem_cooccur_index)
PYTHONPATH=. /home/xiaoyue/anaconda3/envs/llm_graph/bin/python \
    archive/eval_scripts/eval_hotpot_multihop_m2m3.py
```

API 用法：

```python
db.build_sem_cooccur_index()                       # 可选：预构建(否则首查惰性构建)
plan = db.plan_query(question)                      # QuestionPlan(mode=...)
chains, chunk_ids, debug = db.multihop_query(       # 统一路由入口
    question, top_k_chain=10, first_hop_k=20, bridge_top_k=8, second_hop_k=20)
# 也可直接调用单条策略：
chains, ids, dbg = db.multihop_path_query(question)          # 仅 path 桥接
_,     ids, dbg = db.comparison_focused_query(question)      # 仅 comparison
```

## 6. 局限与后续

- ~~routed 比 path 多 ~63 ms/query，来自 `plan_query()` 与策略函数各自的一次 query 解析(GPU 编码)。~~
  **✅ 已修复(M4 同期)**：`multihop_query()` 现只做一次 `_resolve_query_matches()`，把 `resolved_matches`
  同时喂给 `plan_query()` 与所选策略(`chunk_cooccur_query()` 新增 `resolved_matches=` 入参跳过重复编码)。
  **routed 148.1 → 83.6 ms(−44%)、与 path 持平，指标逐位不变**。详见 `multihop_m4_implementation_report.md` §3.4。
- path 的 bridge mrr(0.700)略低于 baseline(0.718)：全局图邻接召回更广，第二跳新证据偶尔轻压首个
  gold 的名次。可在 ChainScore 上对「首跳已高排名 chunk」做名次保护，或引入计划三的 chain reranker
  (cross-encoder，仍不依赖在线生成式 LLM)。
- `comparison_focused_query()` 的标题页优先依赖 `doc_name==title`(HotpotQA 成立)，跨数据集需复核。
- `plan_query()` 为轻量规则，comparison 召回依赖句首比较词 + ≥2 anchor；复杂措辞可能漏判到 bridge
  (安全回退，但拿不到 comparison 交错收益)。后续可补 spaCy 依存模式细化。
- **Milestone 4 ✅(2026-06-15 完成)**：句子级证据图——把 `LocalBridgeEvidence` 重标到句子级
  (同句/邻句/同 chunk)压低伪 bridge,并让证据链携带可读的 hop1/hop2 句子。path/routed 处处非负、
  bridge mrr +0.0025、all mrr +0.002。→ 详见 `reports/multihop_m4_implementation_report.md`,
  综合报告见 `reports/multihop_implementation_summary.md`。
