# 计划四(句子级证据图)实现与评测报告

日期：2026-06-15
对应设计：`reports/multihop_reasoning_implementation_plan.md` 的「实现计划四 / Milestone 4」
前序：`reports/multihop_plan1_implementation_report.md`(M1 按需桥接)、
`reports/multihop_m2m3_implementation_report.md`(M2 全局共现路径 / M3 策略路由)

## 1. 摘要

把多跳推理的「局部证据」粒度从 **chunk 降到 sentence**，目标是计划 §7.1 的两件事：
**进一步压低伪 bridge**(同一长 chunk 内相隔数句、其实无关的两个概念，被 chunk 级共现误判成桥接)
与**提升链路可读性**(证据链从「chunk id → chunk id」变成可直接阅读的两句证据)。

复用 schema 7 已持久化的 `ChunkNode.sentence_boundaries` 与每次出现的 `SpanOccurrence.sentence_id`，
**无需重切文档、不改索引/finalize/pickle schema**。新增两个运行时派生结构，按需构建、`__getstate__`
落盘剥离、加载后重建(与 M2 的 `sem_cooccur_index` 同一套生命周期)：

```python
sentence_evidence_index[(chunk_id, sentence_id)] = {"sem_node_ids": set[int], "doc_id": int}
sem_to_sentences[sem_node_id] = set[(chunk_id, sentence_id)]
```

用法(计划 §7.3)：

1. **伪 bridge 压制**：把证据链的 `LocalBridgeEvidence` 从 M1/M2 的「chunk/边级粗估」改为
   bridge 与 query anchor 在 hop1 chunk 内的**实际句子距离**——同句 1.0 / 邻句 0.8 / 仅同 chunk 0.6——
   并按 `新 tier / 旧 tier` 重标 `ChainScore`。只同 chunk、从不与 anchor 同句或邻句的链停在 0.6 被压低。
2. **链路可读性**：每条链直接携带 `hop1_evidence_sentence` / `hop2_evidence_sentence`
   两句字面证据，以及 `sentence_evidence_tier`(`same-sentence` / `adjacent-sentence` / `same-chunk`)。

### 1.1 HotpotQA distractor 500 消融(句子级证据 OFF vs ON)

直接加载既有 fastidx 缓存(4937 文档/chunk、59,214 语义节点、`fast_index=True`)，构建
`sem_cooccur_index`(15.3 s)与 `sentence_evidence_index`(**0.4 s**，20,237 句、59,214 sem)后评测。
M4 通过 `use_sentence_evidence`(默认开)开关，作用于 path / routed 两条桥接路径：

| 路径 | 分层 | 指标 | OFF | **ON (M4)** | Δ |
| --- | --- | --- | --- | --- | --- |
| **path** | all (n=500) | full_hit | 334 | **335** | +1 |
| | | mrr@10 | 0.6923 | **0.6949** | +0.0026 |
| | bridge (n=404) | mrr@10 | 0.7050 | **0.7075** | +0.0025 |
| | comparison (n=96) | full_hit | 62 | **63** | +1 |
| | | recall@10 | 0.8073 | **0.8125** | +0.0052 |
| **routed** | all (n=500) | full_hit | 336 | 336 | — |
| | | mrr@10 | 0.7282 | **0.7302** | +0.0020 |
| | bridge (n=404) | mrr@10 | 0.7004 | **0.7029** | +0.0025 |
| | comparison (n=96) | 全部 | 不变 | 不变 | — |

要点(诚实口径)：

- **M4 处处非负、小幅为正**：bridge mrr +0.0025、all mrr +0.002，无任何分层回退；path 还多拿 1 个
  all full_hit、1 个 comparison full_hit。routed 的 comparison 分层完全不变——comparison 路由不建链
  (`chains=None`)，句子级证据不参与,符合预期。
- **为什么增量小**：这套 fastidx 缓存里每文档≈1 chunk、平均≈4 句,200 样本 bridge 子集上 735 条链的
  句子级 tier 分布为 **same-sentence 698 / adjacent-sentence 36 / same-chunk 1**——即 ~95% 的 bridge
  本就同句锚定,真正的「跨句伪共现」仅 ~5%。所以压制伪 bridge 的**头部空间天然有限**,M4 精确地命中
  并轻压了这 37 条非同句链,把同句强证据链相对抬升,换来稳定不亏的小幅 mrr 增益。
- **可读性是主要质化收益**(见 §4 示例)。M4 的伪 bridge 压制机制对**长 chunk 语料**(一个 chunk 含
  大段多句正文)收益会显著放大,本数据集只是其下界。

### 1.2 平均查询耗时(500 样本,单进程,GPU 常驻)

| 方法 | OFF | ON (M4) | Δ(M4 开销) |
| --- | --- | --- | --- |
| path `multihop_path_query()` | 84.6 ms | **83.9 ms** | ≈0(噪声内) |
| routed `multihop_query()` | 83.6 ms | **93.0 ms** | +9.4 ms(句子级 refine) |

`sentence_evidence_index` 一次性构建仅 **0.4 s**(远低于 `sem_cooccur_index` 的 15.3 s),构建后所有
查询共用,不计入 per-query。M4 的 per-query 额外开销是「按链做 O(链数) 次集合查 + 句子切片」,
path 路径在测量噪声内;routed 的 +9.4 ms 即句子级 refine 本身。原始数值见
`reports/multihop_m4_eval_results.json`。

> **路由器单次编码优化(本里程碑同期落地)**：M2/M3 报告曾记 routed 比 path 多 ~60–70 ms,源于
> `plan_query()` 与策略函数各做一次 query 编码。已按方案修复——`multihop_query()` 现只调用一次
> `_resolve_query_matches()`,把 `resolved_matches`(含 query 编码)同时喂给 `plan_query()` 与所选
> 策略;`chunk_cooccur_query()` 新增 `resolved_matches=` 入参,传入即跳过重复 GPU 编码、仅跑 CPU 评分。
> **routed 由 148.1 ms 降到 83.6 ms(−44%),与 path 持平**,且 500 样本全方法/全分层指标**逐位不变**
> (纯性能重构)。详见 §3.4。

## 2. 代码落点

延续既有风格:逻辑全部在 `multihop_reasoning.py`,`RAG_graph.py` 仅加薄 wrapper。

- **`multihop_reasoning.py`**(在 M2/M3 之后追加 Milestone 4 段)
  - `build_sentence_evidence_index()` —— 构建/缓存 `sentence_evidence_index` + `sem_to_sentences`
    (一次性、可缓存、`force=` 重建;pre-7 pickle 无句子 id 时产出空结构且
    `_sentence_evidence_available=False`,所有句子级特性自动降级为无操作)
  - `refine_chains_with_sentence_evidence()` —— 按句子级 tier 重标 `LocalBridgeEvidence` 与
    `ChainScore`,并挂载可读 hop1/hop2 句子,重排序链(原地修改)
  - `_bridge_sentence_evidence()` —— bridge 在 hop1 chunk 内对 query anchor 的句子距离 tier
    (同句 1.0 / 邻句 0.8 / 同 chunk 0.6)
  - 辅助:`_sentence_text()`(按 `sentence_boundaries` 切句)、`_resolve_sentence_id()`
    (优先持久化 `sentence_id`,否则按 boundaries 重算,均无则 None)
  - `EvidenceChain` 新增字段 `sentence_evidence_tier` / `hop1_evidence_sentence` /
    `hop2_evidence_sentence`(并进 `as_debug()`)
  - `multihop_path_query()` / `multihop_bridge_query()` 新增 `use_sentence_evidence=True`,在链排序后、
    聚合前调用 refine;`multihop_query()` 经 `**kwargs` 透传该开关
- **`RAG_graph.py`**:`build_sentence_evidence_index` 薄 wrapper;`__init__` 新增运行时字段
  `self.sentence_evidence_index=None` / `self.sem_to_sentences=None`;`_ensure_backward_compatible_attrs()`
  对旧 pickle 默认补 `None`;`__getstate__()` 落盘时剥离这两个结构及 `_sentence_evidence_*` 缓存
  (不持久化,加载后重建)。
- **路由器单次编码优化**(同期落地,见 §3.4):`chunk_cooccur_query()` 新增 `resolved_matches=None`
  入参;`plan_query()` / `multihop_path_query()` / `multihop_bridge_query()` / `comparison_focused_query()`
  各新增 `resolved_matches=None`;`multihop_query()` 改为先 `_resolve_query_matches()` 一次,再把
  `resolved_matches` 同时喂给 `plan_query()` 与所选策略。

评测脚本(`archive/eval_scripts/`,run-as-script):
- `eval_hotpot_multihop_m4.py` —— 500 样本 baseline / path(OFF·ON)/ routed(OFF·ON)消融,
  结果写 `reports/multihop_m4_eval_results.json`

`chunk_cooccur_query()`、M1/M2/M3 既有路径在 `use_sentence_evidence=False` 时行为完全不变。

## 3. 实现要点

### 3.1 句子级结构构建(计划 §7.2)

`build_sentence_evidence_index()` 一次遍历所有 `SemNode.span_occurrences`:对每次出现解析其
`sentence_id`(schema 7 持久化值,缺失则按 `bisect_right(sentence_boundaries, span_start)` 重算),
把 `(chunk_id, sentence_id)` 累进 `sentence_evidence_index` 的 `sem_node_ids` 集合,并反向登记到
`sem_to_sentences`。**句子文本不入库**,而是按需用 `_sentence_text()` 从 `chunk_text + sentence_boundaries`
切片(避免把整个语料正文复制进索引,有意偏离计划 §7.2 字面的 `"text": str`)。500 文档:20,237 句、
0.4 s。运行时结构,**不进 pickle**。

### 3.2 伪 bridge 压制(计划 §7.1 / §7.3)

`refine_chains_with_sentence_evidence(db, chains, query_sem_ids)` 对每条已生成的链:

1. `_bridge_sentence_evidence()` 在 **hop1 chunk** 内求 bridge 句集与 anchor 句集:有交集→同句 1.0;
   否则存在 `|Δsentence|==1`→邻句 0.8;都不满足→仅同 chunk 0.6。
2. 用 `新 tier / 旧 tier` 重标 `ChainScore`(把 M1/M2 的粗 `LocalBridgeEvidence` 换成句子级实测):
   真正同句的链被抬升、跨句伪共现链停在 0.6 被相对压低,链表重排序。
3. 挂载可读证据:hop1 取「与 anchor 同/邻句」那一句(否则 bridge 自身句);hop2 取 bridge 在 hop2 chunk
   的所在句,**无记录时回退该 chunk 首句**(第二跳常是注入的同名标题页,bridge 是页面标题而非索引 span)。

无句子 id 的旧 pickle 上,`_sentence_evidence_available=False`,refine 直接返回原链——安全降级到 chunk 级。

### 3.3 与既有评分/路由的关系

M4 只**重标已有链的一个因子**并重排序,不新增检索召回、不改 bridge 抽取/第二跳注入/聚合权重,
因此 path 在 `use_sentence_evidence=False/True` 间的差异完全归因于句子级证据。routed 经
`multihop_query(**kwargs)` 把开关透传给 bridge/path 路径;comparison 路由不建链,天然不受影响。

### 3.4 路由器单次编码优化(修复 M2/M3 报告 §6 的重复编码)

**问题**:`multihop_query()` 原本先调 `plan_query()`(内部一次 `_resolve_query_matches()` → GPU 编码),
再调所选策略,策略内部又各自调 `chunk_cooccur_query()` → 再编码一次。同一条 query 被编码两遍,routed
因此比 path 多 ~60–70 ms。

**修复**(按报告方案:路由器只做一次 query 解析,复用给 plan 与策略):

- `chunk_cooccur_query()` 新增 `resolved_matches=None`:传入即跳过 `_resolve_query_matches()`(编码),
  直接进入 `_collect_query_cooccurrence_sems()` 及后续 **CPU 评分**;编码只发生在 `_resolve_query_matches`
  里,与 `top_k_chunk`/`idf_prune`/`candidate_pool` 等评分参数无关,故复用安全、各策略自有参数不受影响。
- `plan_query()` / `multihop_path_query()` / `multihop_bridge_query()` / `comparison_focused_query()`
  各透传 `resolved_matches`。
- `multihop_query()` 改为:先 `_resolve_query_matches(query_text, search_mode=..., expand_compositional=True)`
  **一次**,把结果同时给 `plan_query()`、所选策略、以及 single 路径的 `chunk_cooccur_query()`。

**效果**:routed `148.1 → 83.6 ms`(−44%,与 path 持平);500 样本全方法/全分层 full_hit/recall/mrr
**逐位一致**(`eval_hotpot_multihop_m4.py` 重跑对比),确认是纯性能重构。`plan_query()` 剩余开销仅为
规则判定(cue/lead/wh 正则 + anchor 取用),无额外编码。

## 4. 链路可读性示例(真实输出)

`multihop_path_query("What 1945 film starred the actress who later played Corliss Archer?")` top1 链:

```
[same-sentence] Kiss and Tell (1945 film) --shirley temple--> Shirley Temple  (score=1.86)
  hop1: "Kiss and Tell is a 1945 American comedy film starring then 17-year-old
         Shirley Temple as Corliss Archer."
  hop2: "Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American
         actress, singer, dancer, businesswoman, and diplomat ..."   # bridge 同名传记页(首句回退)
```

`sentence_evidence_tier=same-sentence` 直接说明「桥接实体 Shirley Temple 与 query 锚点
(Corliss Archer / 1945 film)在 hop1 同一句出现」,而非仅靠 chunk 级共现——这正是链可信度的核心证据,
也让结果可被人直接核验。200 样本 bridge 子集的 tier 分布(§1.1)进一步量化了这种「同句锚定占比」。

## 5. 复现方式

```bash
# M4 消融:baseline / path(OFF·ON)/ routed(OFF·ON)
PYTHONPATH=. /home/xiaoyue/anaconda3/envs/llm_graph/bin/python \
    archive/eval_scripts/eval_hotpot_multihop_m4.py
```

API 用法:

```python
db.build_sem_cooccur_index()            # M2 全局共现图(path 桥接所需)
db.build_sentence_evidence_index()      # M4 句子级证据(可选预构建,否则首查惰性构建)
chains, ids, dbg = db.multihop_path_query(question)          # 默认 use_sentence_evidence=True
for c in chains:
    print(c.sentence_evidence_tier, c.hop1_evidence_sentence, "->", c.hop2_evidence_sentence)
# 关闭句子级证据(回到 M2 纯 chunk/边级 LocalBridgeEvidence):
chains, ids, dbg = db.multihop_path_query(question, use_sentence_evidence=False)
```

## 6. 局限与后续

- **本数据集头部空间小**:fastidx 每文档≈1 chunk、平均≈4 句,~95% bridge 已同句锚定,M4 的伪 bridge
  压制只命中 ~5% 跨句链,故指标增量小。**长 chunk / 整段正文语料**上收益会明显放大,值得在此类数据上复测。
- 句子级 tier 只进入 `LocalBridgeEvidence` 一个乘子。可进一步:对「仅同 chunk」链做更强惩罚或硬过滤
  (需评估 recall 取舍)、把邻句窗口放宽到 ±k、或在 hop2 也加 anchor↔bridge 同句校验。
- hop2 句子在 bridge 为标题页时回退首句;若标题页正文多句,首句通常即定义句(可读性足够),但并非永远
  是承载答案属性的那一句。后续可结合 `answer_focus` 选最相关句。
- 与 M3 的 chain reranker(cross-encoder,仍不依赖在线生成式 LLM)正交,可叠加:句子级证据天然为
  cross-encoder 提供更短、更聚焦的 rerank 输入。

## 7. 里程碑收尾

至此多跳推理路线图 M1–M4 全部落地(详见综合报告 `reports/multihop_implementation_summary.md`):
M1 按需桥接 → M2 全局共现路径 → M3 问题分解/策略路由(修复 comparison 回退)→ M4 句子级证据图
(压伪 bridge + 可读链)。全程**查询端实现、不改索引/finalize/pickle schema、不依赖在线生成式 LLM**。
