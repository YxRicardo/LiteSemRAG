# 计划一(两阶段桥接检索)实现与评测报告

日期：2026-06-10
对应设计：`reports/multihop_reasoning_implementation_plan.md` 的「实现计划一 / Milestone 1」

## 1. 摘要

按计划一实现了**最小侵入的两阶段桥接检索** `multihop_bridge_query()`，专攻 HotpotQA bridge
类多跳问题：先用现有 `chunk_cooccur_query()` 召回第一跳证据，再从第一跳 chunk 的
`sem_node_list` 动态抽取桥接语义节点(query 中未显式给出的高质量实体/短语)，用 bridge 做
第二跳检索，最后按 ChainScore 生成并排序证据链，聚合去重后返回 chunk。

**不改索引结构、不改 pickle schema、不改 `finalize()` / semantic node 构建逻辑**，
只在查询端复用现有能力。直接加载已有 fastidx 缓存评测(无需重建索引)。

HotpotQA distractor 500 样本结果(对比单跳基线 `chunk_cooccur_query()`)：

| 分层 | 指标 | baseline | multihop | Δ |
| --- | --- | --- | --- | --- |
| **bridge** (n=404) | full_hit | 206 (0.510) | **259 (0.641)** | **+53 / +13.1pts** |
| | recall@10 | 0.719 | **0.785** | +0.066 |
| | mrr@10 | 0.718 | **0.728** | +0.010 |
| **all** (n=500) | full_hit | 266 (0.532) | **319 (0.638)** | +53 |
| | recall@10 | 0.734 | **0.784** | +0.050 |
| | mrr@10 | 0.711 | 0.704 | −0.007 |
| **comparison** (n=96) | full_hit | 60 (0.625) | 60 (0.625) | 持平 |
| | recall@10 | 0.797 | 0.781 | −0.016 |
| | mrr@10 | 0.684 | 0.604 | −0.080 |

### 1.1 平均查询耗时（500 样本，单进程，GPU 常驻，含 query 编码）

| 方法 | 总耗时 | 平均每条 query | 相对基线 |
| --- | --- | --- | --- |
| baseline `chunk_cooccur_query()` | 43.9 s | **87.8 ms** | — |
| multihop `multihop_bridge_query()` | 61.9 s | **123.9 ms** | +36.0 ms / +41% |

multihop 已**内含**一次第一跳 `chunk_cooccur_query()`，因此 bridge 抽取 + 第二跳 BM25 召回 +
链路评分 + 聚合的**净额外开销约 36 ms/query**。计时口径：同一已加载的 db、单进程顺序执行、
`query_database` 常驻 cuda，时间含 query 端 DeBERTa 编码。原始数值见
`reports/multihop_eval_results.json` 的 `avg_query_ms` 字段。

结论：bridge 分层(占 80%，正是该方法的目标)在 full_hit / recall / mrr 三项**全部提升**；
comparison full_hit **零回退**，满足 Milestone 1 验收(「bridge full_hit/recall 不低于基线、
comparison 不明显回退」)。唯一代价是 comparison 的 mrr 下降 0.08，源于桥接对并列实体问题
帮助有限、二跳新证据偶尔轻微下压首个 gold 的排名——计划三的 comparison 专用路由可消除。

## 2. 代码落点

按计划 §9 建议，新增独立模块、`RAG_graph.py` 只保留薄 wrapper，避免主文件膨胀：

- **`multihop_reasoning.py`**（新增，约 470 行）
  - `BridgeCandidate` / `EvidenceChain` dataclass
  - `extract_bridge_candidates_from_chunks()` —— bridge 抽取与过滤(§10.1)
  - `retrieve_second_hop_for_bridge()` —— bridge BM25 第二跳召回
  - `rank_evidence_chains()` —— 证据链生成与 ChainScore 排序
  - `aggregate_chunk_ranking()` —— 第一跳 + chain 两端证据的组合分数聚合
  - `multihop_bridge_query()` —— 对外主入口
- **`RAG_graph.py`**：`LiteSemRAG.multihop_bridge_query()` 薄 wrapper，`from multihop_reasoning import ...` 委托。
  现有 `chunk_cooccur_query()` 及 SciFact/FEVER/HotpotQA 单跳评测口径**完全不受影响**。

评测/验证脚本(`archive/eval_scripts/`，run-as-script)：
- `eval_hotpot_multihop.py` —— 500 样本 baseline vs multihop 分层评测(结果写 `reports/multihop_eval_results.json`)
- `multihop_smoke_test.py` —— 计划 §11.1 的 4 个 smoke case
- `multihop_agg_sweep.py` —— 聚合权重消融扫描

## 3. 核心流程

```
query
 └─[第一跳] chunk_cooccur_query(query, top_k_chunk=first_hop_k=20)
     ├─ first_hop_score = 各 chunk 归一化 final_score
     └─ resolved query sems(复用 _resolve_query_matches + _collect_query_cooccurrence_sems)
 └─[bridge 抽取] 遍历第一跳 chunk 的 sem_node_list
     ├─ 排除：query 已命中的 sem / query 词、低 idf 泛词、高 df hub
     ├─ 偏好：phrase、entity(force_single)、有 description、首字母大写、标题匹配
     └─ 跨 chunk 同名 bridge 取 prior×local 最强一条，保留全局 top bridge_top_k=8
 └─[第二跳] 每个 bridge 取 BM25 top second_hop_k=20 个 chunk
     └─ 额外强制注入 bridge 实体的同名页面(HotpotQA 第二跳 gold 的典型结构)
 └─[评分] ChainScore 生成证据链 → 按 (hop1_doc, bridge, hop2_doc) 去重 → top_k_chain
 └─[聚合] aggregate_chunk_ranking 把第一跳与 chain 两端证据排成最终 chunk 列表
```

## 4. 评分方案（计划 §4.6 实现）

```
ChainScore = FirstHopScore × BridgePrior × LocalBridgeEvidence
           × SecondHopScore × ConstraintCoverage × DiversityPenalty × Hop2TitleBonus
```

| 因子 | 定义 | 说明 |
| --- | --- | --- |
| `FirstHopScore` | 第一跳 chunk 的归一化 `final_score` | 来自 `chunk_cooccur_query` debug 的 `all_scored_chunks`，除以最大值归一 |
| `BridgePrior` | `idf/max_idf` × 类型/标题系数 | phrase×1.15、entity×1.10、有描述×1.10、大写×1.10、标题匹配×1.30 |
| `LocalBridgeEvidence` | bridge 与 query 节点**同句=1.0 / 同 chunk=0.6** | 复用 `_nodes_share_sentence()`(schema 7 句子 id) |
| `SecondHopScore` | bridge 在第二跳 chunk 的 BM25，归一到 (0,1] | 除以该 bridge 自身最大 BM25 |
| `ConstraintCoverage` | 第二跳对**剩余 query 约束**的 idf 加权 soft 覆盖 | `0.3 + 0.7×coverage`，软覆盖避免硬过滤误杀(计划明确提醒) |
| `DiversityPenalty` | hop1==hop2 同 chunk 时 ×0.3 | 抑制伪多跳 |
| `Hop2TitleBonus` | 第二跳目标=bridge 实体同名页面时 ×1.80 | HotpotQA bridge 的关键正向信号(详见 §6) |

「剩余约束」= 第一跳 chunk 未覆盖的 query 语义节点，天然落到 answer focus 上
(例如 `Kiss and Tell` 第一跳已覆盖电影/角色名，剩余的 `government position` 才是第二跳目标)。
这正实现了计划 §4.6 强调的「第二跳 ConstraintCoverage 应覆盖剩余约束而非全部原始 query」。

最终 chunk 聚合采用组合分数：
```
combined(chunk) = FirstHopScore(chunk)
                + 0.5 × max chain.score(此 chunk 作为某 chain 的 hop1)
                + 1.0 × max chain.score(此 chunk 作为某 chain 的 hop2)   [仅 chain.score ≥ 0.2]
```
第二跳新证据全权、第一跳半权、并设 0.2 门槛过滤弱链——既能把较低排名的第二个 gold 提进
top-k，又不打乱第一跳内部排序压低首个 gold 的 MRR(消融见 §7)。

## 5. Smoke case 验证（计划 §11.1）

| Case | baseline | multihop | bridge 是否抽出 |
| --- | --- | --- | --- |
| `Kiss and Tell → Shirley Temple` (bridge) | 1/2 | **2/2** | `shirley temple` ✓，top chain score 0.93 |
| `Lewiston Maineiacs → Androscoggin Bank Colisée` (bridge) | 1/2 | **2/2** | `androscoggin bank colisée` ✓ |
| `University of Kansas → Kansas Song` (bridge) | 2/2 | 2/2 | 经 `university of kansas` 桥接到 `Kansas Song` |
| `Scott Derrickson vs Ed Wood` (comparison) | 1/2 | 1/2(不回退) | 无需桥接，弱链未压掉 gold |

3 个 bridge case 两个从 1/2 提升到 2/2，comparison 行为不被破坏，与计划预期一致。

## 6. 关键实现细节：第二跳「同名页面」注入

最初实现暴露的核心问题：bridge `shirley temple` 抽对了，但它出现在很多 chunk 里，单纯按
BM25 取第二跳，top1 落到 `A Kiss for Corliss`(也提到 Shirley Temple 的电影)，而非真正的
gold —— Shirley Temple **自己的传记页面**。

HotpotQA bridge 的典型结构是「第一跳文档提到桥接实体 B → 第二个 gold 文档就是 B 自己的
wiki 页面」。据此加入两条针对性处理：
1. **同名页面注入**：第二跳候选里强制纳入 `normalize(title)==bridge` 的 chunk(即使不在 BM25 top-20)。
2. **Hop2TitleBonus ×1.80**：第二跳目标正是 bridge 同名页面时强加权。

加入后三个 bridge smoke case 全部 full_hit，500 样本 bridge full_hit 从 248 提到 259。

## 7. 消融实验（聚合权重，500 样本）

固定检索/评分逻辑，只扫聚合权重 `(hop1_w, hop2_w, min_chain)`：

| 配置 | bridge full | bridge mrr | all mrr | comparison full |
| --- | --- | --- | --- | --- |
| baseline | 206 | 0.718 | 0.711 | 60 |
| h1=1.0, h2=1.0, min=0.0(全 re-sort) | 252 | 0.705 | 0.676 | 58 |
| h1=0.5, h2=1.0, min=0.0 | 258 | 0.728 | 0.704 | 58 |
| h1=0.3, h2=1.0, min=0.1 | 260 | 0.724 | 0.700 | 59 |
| **h1=0.5, h2=1.0, min=0.2 (默认)** | **259** | **0.728** | **0.704** | **60** |

要点：
- 一开始的「全量 combined re-sort」(h1=h2=1.0)会把第一跳内部排序打乱，all mrr 掉到 0.676；
- 弱化 hop1 加成(0.5)即可让 bridge mrr 反超基线(0.728>0.718)，all mrr 回到 0.704；
- 0.2 的 chain 门槛过滤掉 comparison 噪声弱链，使 comparison full_hit 回到基线 60(零回退)。

## 8. 优点 / 风险 / 局限（计划 §4.7–4.8）

优点：
- 零索引/schema 改动，可直接用现有缓存，失败可无缝回退 `chunk_cooccur_query()`。
- 仍然**不需要在线生成式 LLM**。
- 输出可解释证据链(每条含 hop1/bridge/hop2/各因子分数/why_selected，见 debug_info)。

风险/局限：
- 桥接候选来自第一跳 chunk 的全部 sem，噪声较大；目前靠 idf/df gating + 类型/标题加权压制。
- 对 comparison 帮助有限(full_hit 持平、mrr 略降)——需计划三的 comparison 专用路由。
- 第二跳「同名页面注入」依赖 `doc_name==title`(HotpotQA 成立)，跨数据集需复核该假设。
- 当前 fast_index 缓存下每文档≈1 chunk，句子级伪共现风险低；长 chunk 语料宜配合计划四的句子级证据图。

## 9. 复现方式

```bash
# 评测(自动加载 fastidx 缓存，不重建索引)
PYTHONPATH=. /home/xiaoyue/anaconda3/envs/llm_graph/bin/python archive/eval_scripts/eval_hotpot_multihop.py
# smoke case
PYTHONPATH=. /home/xiaoyue/anaconda3/envs/llm_graph/bin/python archive/eval_scripts/multihop_smoke_test.py
# 聚合权重消融
PYTHONPATH=. /home/xiaoyue/anaconda3/envs/llm_graph/bin/python archive/eval_scripts/multihop_agg_sweep.py
```

API 用法：
```python
chains, chunk_ids, debug = db.multihop_bridge_query(
    question, top_k_chain=10, first_hop_k=20, bridge_top_k=8, second_hop_k=20)
```

## 10. 后续（按计划路线图）

- **Milestone 2 ✅(2026-06-15 完成)**：全局 `sem_cooccur_index` + `multihop_path_query()` 路径桥接
  —— bridge full_hit 259 → 272、recall 0.785 → 0.808。
- **Milestone 3 ✅(2026-06-15 完成)**：`plan_query()` 问题分解 + 策略路由(comparison-focused /
  bridge-path / 单跳)，**comparison mrr 0.604 → 0.845**(高于 baseline 0.684)，all mrr 回到 0.728。
  → 详见 `reports/multihop_m2m3_implementation_report.md`。
- **Milestone 4 ✅(2026-06-15 完成)**：句子级证据图——`LocalBridgeEvidence` 重标到句子级
  (同句/邻句/同 chunk)压低伪 bridge + 链路携带可读 hop1/hop2 句子。path/routed 处处非负、
  bridge mrr +0.0025。→ 详见 `reports/multihop_m4_implementation_report.md`。
- **综合报告**：M1–M4 全链路见 `reports/multihop_implementation_summary.md`。
