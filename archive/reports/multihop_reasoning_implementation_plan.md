# LiteSemRAG 多跳推理实现分析报告

日期：2026-06-10

## 1. 结论摘要

当前 LiteSemRAG 的数据结构已经具备实现多跳推理的基础：文档、chunk、token/phrase、semantic node 之间有清晰的双向连接；query 端也已经能把问题解析成多个语义节点，并用 BM25、短语匹配、modifier boost 和 chunk co-occurrence 对 chunk 排序。

但现有实现还不是多跳推理。当前 `multi_level_query()` 和 `chunk_cooccur_query()` 本质是“一次性多证据检索”：把 query 中已经出现的概念映射到 semantic nodes，再找同时命中这些节点的 chunk。它没有维护 hop 状态、没有桥接实体发现、没有跨 chunk 路径搜索、没有把第一跳结果生成的新约束用于第二跳检索，也没有返回可解释的证据链。

建议采用分阶段实现：

1. 先做最小侵入的“两阶段桥接检索”，复用现有 `chunk_cooccur_query()`、`ChunkNode.sem_node_list` 和 query 解析能力，快速验证 HotpotQA bridge 类问题收益。
2. 再补一个轻量全局 semantic co-occurrence/path index，把 chunk 内共现从 query-time 临时图升级为可遍历的全局图。
3. 最后做面向多跳问题的 query decomposition 与 evidence-chain reranking，让系统能够显式输出 `query concept -> bridge -> answer evidence` 的路径。

基于 HotpotQA dev distractor 的真实样本看，这条路线是可验证的。该文件 `jupyter_notebooks/hotpot_dev_distractor_v1.json` 共 7,405 个 hard 样本，其中 bridge 5,918 个、comparison 1,487 个；每个样本的 gold supporting titles 都是 2 个，平均 supporting facts 为 2.43 条。这说明评测目标天然适合“找出两个证据文档/证据 chunk，并解释它们如何连接”的 chain-level 检索，而不是只返回一个全局排序的 chunk 列表。

## 2. 当前结构中可复用的多跳基础

### 2.1 节点和边已经足够表达证据

`RAG_graph.py` 中的核心 dataclass 已经形成了图：

- `DocumentNode`：文档级节点，持有 `chunk_node_list`。
- `ChunkNode`：chunk 级节点，持有 `doc_node`、`sem_node_list`、`sentence_boundaries`。
- `TokenNode`：规范化 token/phrase 节点，持有 `sem_node_list`、出现记录、组合短语 head/modifier 信息。
- `SemNode`：语义节点，持有 `chunk_node_list`、`span_occurrences`、`BM25`、`idf`、`embed` 和可选 `description`。
- `SpanOccurrence`：保留 span 的 chunk、字符范围、surface、phrase type、modifier、sentence id。

这意味着系统已经能回答两个多跳实现需要的底层问题：

- 一个语义节点出现在哪些 chunk 中。
- 一个 chunk 中有哪些语义节点，以及它们是否同句/同 chunk 出现。

`finalize()` 末尾调用 `build_chunk2sem_edge()`，把 `SemNode -> ChunkNode` 的边反向补到 `ChunkNode.sem_node_list`。这是实现“从第一跳命中的 chunk 抽取桥接候选”的关键入口。

### 2.2 Query 端已经有成熟的语义解析

`_resolve_query_matches()` 已经完成：

- query 清洗和编码；
- entity / phrase / token 抽取；
- compositional phrase 展开为 full phrase、head、modifier；
- exact token/phrase match；
- phrase word intersection fuzzy match；
- embedding fallback；
- 多义 semantic node 的 query-time sense disambiguation。

这部分可以直接复用为多跳算法的“起始概念解析器”。新增多跳逻辑不应重写 query tokenization，而应在 `_resolve_query_matches()` 和 `_collect_query_cooccurrence_sems()` 输出上继续构建。

### 2.3 当前 co-occurrence 能做局部证据合流

`CoOccurrenceGraph` 当前有两套逻辑：

- `build_edges()`：把共享 chunk 的 query semantic nodes 连边，并把边权加回 node weight，供 legacy `multi_level_query()` 使用。
- `build_pair_edges()` + `assign_base_evidence()` + `nodes_by_chunk()`：供 `chunk_cooccur_query()` 使用，做 chunk-level bounded pair boost。

`get_COG_edge_weight()` 的边权是两个 semantic nodes 的 chunk 集合 overlap：

```text
|chunks(a) ∩ chunks(b)| / sqrt(|chunks(a)| * |chunks(b)|)
```

这适合作为多跳路径中的全局相关性 prior，但当前只在 query-time 对“已经由 query 命中的节点集合”临时计算，没有作为全局图保存。

### 2.4 当前 HotpotQA 评测已按 bridge/comparison 分层

`eval_hotpot_litesem_inproc.py` 已经把 HotpotQA distractor 样本按 `comparison / bridge` 分层评估，指标包括 `Recall@10`、`MRR@10`、`full_hit`。这可以直接作为多跳改动的回归评测入口。

### 2.5 HotpotQA 样本对方法设计的直接约束

HotpotQA 的 bridge 样本通常不是要求系统在同一个 chunk 内凑齐所有 query concept，而是要求系统先定位一个中间实体，再用这个中间实体检索第二个证据文档。下面几个真实样本能直接说明为什么需要 bridge candidate extraction 和 evidence chain：

| 类型 | 问题 | Gold 支持句形成的链 | 对实现的启发 |
| --- | --- | --- | --- |
| bridge | `What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?` | `Kiss and Tell (1945 film)` 句 0：影片由 Shirley Temple 饰演 Corliss Archer；`Shirley Temple` 句 1：她曾任 Chief of Protocol of the United States。 | 第一跳 chunk 中的 `Shirley Temple` 不在 query 的显式目标位置，但它是必须抽出的 bridge。第二跳不能强制匹配 `Kiss and Tell`，应围绕 bridge 和 answer focus `government position` 检索。 |
| bridge | `The arena where the Lewiston Maineiacs played their home games can seat how many people?` | `Lewiston Maineiacs` 句 1：主场是 Androscoggin Bank Colisée；`Androscoggin Bank Colisée` 句 0：容量为 `4,000 capacity (3,677 seated)`。 | bridge candidate 应偏好同句实体/专名短语；`home games` 与 `arena` 是第一跳关系，`seat/how many people` 是第二跳 answer focus。 |
| bridge | `What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?` | `University of Kansas` 句 1/2：main campus 和 branch campuses 约束定位到 University of Kansas；`Kansas Song` 句 0：它是 University of Kansas 的 fight song。 | 并非所有 bridge 都从一个已命名 anchor 开始；有些问题先用属性约束识别 bridge entity，再跳到 answer 文档。first-hop retrieval 需要允许多个约束共同召回同一实体文档。 |
| comparison | `Were Scott Derrickson and Ed Wood of the same nationality?` | `Scott Derrickson` 句 0：American；`Ed Wood` 句 0：American。 | comparison 不应强行构造 `A -> bridge -> B`，而应分别检索两个 side 的证据，再比较同一个属性。 |

这些样本也暴露了一个设计细节：第二跳的 `ConstraintCoverage` 应当主要覆盖“剩余约束/answer focus”，而不是要求 second-hop chunk 再次覆盖全部原始 query。以 `Kiss and Tell -> Shirley Temple` 为例，正确 second-hop 句子通常不会再出现 `Kiss and Tell` 或 `Corliss Archer`，但会出现 bridge `Shirley Temple` 和目标属性 `Chief of Protocol`。

## 3. 当前缺口

### 3.1 没有显式 hop 状态

现有查询是单轮函数调用：

```text
query -> resolved semantic nodes -> candidate chunks -> sorted chunks
```

多跳推理至少需要：

```text
query -> first-hop concepts -> first-hop evidence -> bridge candidates
      -> second-hop query/constraints -> second-hop evidence -> chain score
```

当前没有 `HopState`、`EvidenceChain`、`BridgeCandidate` 之类的状态对象，也没有记录“某个 chunk 是第几跳、由哪个前驱触发、连接关系是什么”。

### 3.2 没有桥接实体/桥接短语发现

HotpotQA bridge 问题通常需要先找到中间实体，例如：

```text
Question mentions A and asks property of entity related to A.
Hop 1: find B from document about A.
Hop 2: retrieve document/chunk about B and answer property.
```

当前系统只检索 query 中已经出现的实体/短语。它不会从第一跳 chunk 的 `sem_node_list` 或 `span_occurrences` 中挑出可能的 B，再把 B 作为第二跳检索条件。

真实样本中这个缺口很明显。`Kiss and Tell` 问题的 query 显式给出的是电影名、角色名和目标属性，答案所在的 `Shirley Temple` 文档需要通过第一跳句子中的新实体触发；`Lewiston Maineiacs` 问题需要从第一跳句子抽出 `Androscoggin Bank Colisée`；`University of Kansas` 问题则需要从多个地理约束共同定位出 `University of Kansas`，再跳到 `Kansas Song`。如果只检索 query 中已有的 phrase，系统很容易返回第一跳文档，却漏掉第二个 gold title。

### 3.3 Co-occurrence graph 不是全局可遍历图

当前 `CoOccurrenceGraph` 是 query-specific wrapper。它只包含当前 query 匹配出来的 semantic nodes。多跳需要从一个 node 扩展到相邻 node，例如：

```text
sem(A) -> chunk containing A and B -> sem(B) -> chunk containing B and C
```

这要求有全局 adjacency 或至少按需构建的 chunk-to-sem 扩展函数。现有 `ChunkNode.sem_node_list` 可以支持按需扩展，但还没有封装成可控的 path search。

### 3.4 排序目标还是 chunk，不是 evidence chain

`chunk_cooccur_query()` 返回的是 chunk 列表和 debug 信息，debug 解释的是 chunk 的 base evidence / pair boost。多跳推理需要排序证据链，例如：

```text
chain = [
  hop1 chunk: contains query entity A and bridge B,
  hop2 chunk: contains bridge B and answer constraint C
]
```

链路评分应该同时考虑：

- query 起点匹配质量；
- bridge 节点可靠性；
- hop1 局部证据强度；
- hop2 对剩余 query 约束的覆盖；
- 两个 hop 是否跨文档、同文档、同标题或通过实体表面一致连接；
- 是否重复返回同一 chunk 导致伪多跳。

当前没有这层模型。

### 3.5 缺少问题分解/剩余约束建模

对 comparison 问题，多跳不一定是桥接实体扩展，而可能是两个并列实体分别检索后比较。对 bridge 问题，需要区分起点实体、关系短语、目标属性。当前 `_prepare_query_tokens()` 能抽词，但没有把 query units 分成：

- anchor entity；
- bridge relation；
- answer constraint；
- comparison sides；
- generic predicate。

没有这层，第二跳容易把第一跳 chunk 中的高频词当桥接点，带来噪声。

HotpotQA 样本建议至少先做轻量规则分解，而不是完全依赖 raw query units：

- `Kiss and Tell`：anchor 是 `Kiss and Tell` / `Corliss Archer`，bridge relation 是 `portrayed`，answer focus 是 `government position`。
- `Lewiston Maineiacs`：anchor 是 `Lewiston Maineiacs`，bridge relation 是 `played their home games`，answer focus 是 `can seat how many people`。
- `University of Kansas`：anchor 不是单一命名实体，而是 `main campus in Lawrence` + `branch campuses in Kansas City metropolitan area` 这组定位约束，answer focus 是 `fight song`。
- `Scott Derrickson / Ed Wood`：两个 side 是并列实体，目标属性是 `nationality`，应该走 comparison focused retrieval。

## 4. 实现计划一：最小侵入的两阶段桥接检索

### 4.1 目标

在不改索引结构和 pickle schema 的前提下，实现一个新的查询入口：

```python
multihop_bridge_query(
    query_text,
    top_k_chain=10,
    first_hop_k=20,
    bridge_top_k=8,
    second_hop_k=20,
)
```

它专门解决 bridge 类多跳：先找 query anchor 相关 chunk，再从这些 chunk 中抽 bridge candidates，再用 bridge + 原 query 剩余概念做第二跳检索，最后返回证据链和去重后的 chunk。

### 4.2 复用现有能力

- 用 `_resolve_query_matches(..., expand_compositional=True)` 获取 query semantic nodes。
- 用 `chunk_cooccur_query()` 做 first-hop candidate generation。
- 用 `ChunkNode.sem_node_list` 从 first-hop chunks 中抽 bridge semantic nodes。
- 用 `get_top_k_chunks_for_sem_node()` 或 `SemNode.BM25` 找 bridge node 的 second-hop chunks。
- 用 `_nodes_share_sentence()` 判断 first-hop chunk 内 query node 和 bridge node 是否同句。
- 用 `chunk_cooccur_query()` 的 debug 中 `resolved_matches` 和 `token_weight_map` 做可解释输出。

### 4.3 核心流程

1. 解析 query，得到 query sem nodes 和 query token 权重。
2. 执行 `chunk_cooccur_query(query_text, top_k_chunk=first_hop_k)`，拿到 first-hop chunks。
3. 对每个 first-hop chunk，从 `chunk.sem_node_list` 中抽取 bridge candidates。
4. 过滤 bridge candidates：
   - 排除已经由 query 直接命中的 sem nodes。
   - 排除低 IDF / 高 df 的泛词。
   - 优先保留 entity、atomic phrase、capitalized phrase、带 `description` 的 semantic node。
   - 若 bridge 与 query anchor 同句，给局部证据加权。
5. 对每个 bridge candidate 做 second-hop retrieval：
   - 取该 bridge sem node 的 top chunks。
   - 可选：把 query 中未被 first-hop 覆盖的高 IDF query sem nodes 加入约束，要求 second-hop chunk 至少命中一个剩余约束。
6. 生成 evidence chain：
   - `hop1_chunk_id`
   - `bridge_sem_node_id`
   - `hop2_chunk_id`
   - `score`
   - `explanation`
7. 按 chain score 排序，返回 top chains 和聚合 chunk ids。

### 4.4 HotpotQA 例子：`Kiss and Tell -> Shirley Temple`

以问题 `What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?` 为例，MVP 的理想执行轨迹如下：

1. Query 解析得到高权重概念：`Kiss and Tell`、`Corliss Archer`、`portrayed`、`government position`。
2. `chunk_cooccur_query()` 第一跳召回 `Kiss and Tell (1945 film)` 的句 0，因为该句同时覆盖电影名、角色名和 `starring/portrayed` 关系。
3. 从该 chunk 的 `sem_node_list` 抽取 bridge candidates，`Shirley Temple` 因为是同句专名实体、IDF 较高、并且不是 query 已显式命中的节点而排在前面。
4. 第二跳用 `Shirley Temple` 加 soft answer focus `government position` 检索，召回 `Shirley Temple` 句 1，其中包含 `Chief of Protocol of the United States`。
5. 生成 chain：

```text
query concept: Kiss and Tell / Corliss Archer
hop1 evidence: Kiss and Tell (1945 film), sentence 0
bridge: Shirley Temple
hop2 evidence: Shirley Temple, sentence 1
answer-bearing span: Chief of Protocol of the United States
```

这个例子支持两个实现选择：第一，bridge 必须从 first-hop evidence 中动态发现；第二，second-hop rerank 应该允许原 query anchor 缺席，而用 bridge 命中和 answer focus 命中补足链路可信度。

### 4.5 HotpotQA 例子：`Lewiston Maineiacs -> Androscoggin Bank Colisée`

问题 `The arena where the Lewiston Maineiacs played their home games can seat how many people?` 的 gold chain 是：

```text
Lewiston Maineiacs sentence 1:
  The team played its home games at the Androscoggin Bank Colisée.

Androscoggin Bank Colisée sentence 0:
  The Androscoggin Bank Colisée ... is a 4,000 capacity (3,677 seated) multi-purpose arena ...
```

该样本说明 `LocalBridgeEvidence` 的 same-sentence boost 很有价值：`Lewiston Maineiacs`、`home games`、`Androscoggin Bank Colisée` 在同一句内形成强局部关系。第二跳的目标则是容量属性，因此 `can seat/how many people/arena` 应作为 soft constraint 参与 rerank，而不是作为硬过滤条件；否则如果句子只写 `3,677 seated` 而不重复 `home games`，正确 evidence 会被误杀。

### 4.6 初版评分建议

```text
ChainScore =
  FirstHopScore
  * BridgePrior
  * LocalBridgeEvidence
  * SecondHopScore
  * ConstraintCoverage
  * DiversityPenalty
```

建议初始定义：

- `FirstHopScore`：来自 `chunk_cooccur_query().debug_info["all_scored_chunks"]` 的 normalized final score。
- `BridgePrior`：`idf(bridge_token)`，并对 entity/phrase 加系数。
- `LocalBridgeEvidence`：bridge 与任一 query sem 同句为 `1.0`，同 chunk 为 `0.6`。
- `SecondHopScore`：bridge sem node 在 second-hop chunk 的 BM25，归一化到 `[0, 1]`。
- `ConstraintCoverage`：second-hop chunk 命中剩余 query sem 的比例或加权 BM25。
- `DiversityPenalty`：hop1 与 hop2 是同一 chunk 时降权，除非用户允许单 chunk answer。

基于上面的样本，`ConstraintCoverage` 建议拆成两项：

- `ResidualQueryCoverage`：second-hop chunk 对剩余高 IDF query units 的覆盖，作为 soft score。
- `AnswerFocusCoverage`：对 `government position`、`can seat how many people`、`fight song`、`founded in what year` 这类目标属性词的覆盖，权重高于普通 query token。

这样可以避免把 `Kiss and Tell` 这样的第一跳 anchor 错误地要求出现在 `Shirley Temple` second-hop evidence 中。

### 4.7 优点

- 实现成本低，不需要改 `finalize()`、持久化或 schema。
- 与当前系统风格一致，仍然不需要在线生成式 LLM。
- 可以快速在 HotpotQA bridge 分层上验证收益。
- 失败时容易回退到 `chunk_cooccur_query()`。

### 4.8 风险

- 桥接候选来自 first-hop chunk 的所有 sem nodes，噪声较大。
- 没有全局 path index，second-hop 扩展可能慢，需要限制 `bridge_top_k` 和每个 bridge 的 chunk fanout。
- 对 comparison 问题帮助有限。

### 4.9 建议优先级

最高。建议作为第一个 PR/commit 实现。

## 5. 实现计划二：全局 Semantic Co-occurrence Path Index

### 5.1 目标

把当前 query-time 临时 co-occurrence 能力升级为可遍历的全局轻量图索引，支持：

```text
sem_node -> neighbor sem_nodes -> shared chunks -> path search
```

新增结构可以命名为：

```python
self.sem_cooccur_index
```

建议默认按需构建，也可以在 `finalize()` 后显式调用：

```python
build_sem_cooccur_index(max_nodes_per_chunk=80, min_edge_weight=..., top_neighbors=...)
```

### 5.2 数据结构

建议使用稀疏 adjacency：

```python
sem_cooccur_index = {
    sem_node_id: {
        neighbor_sem_node_id: {
            "weight": float,
            "shared_chunk_ids": list[int],
            "same_sentence_count": int,
            "chunk_count": int,
        }
    }
}
```

如果内存压力大，可以拆成：

- `neighbor_weights: dict[int, list[tuple[int, float]]]`
- `edge_shared_chunks: dict[tuple[int, int], list[int]]`
- `edge_stats: dict[tuple[int, int], EdgeStats]`

### 5.3 构建方法

遍历 `ChunkNode.sem_node_list`，对 chunk 内 sem nodes 两两计数：

1. 过滤太泛的 sem nodes：低 idf、高 df、stop-like token。
2. 对过大的 chunk 做 cap：只保留 top IDF / entity / phrase sem nodes，避免 O(n²) 爆炸。
3. 对 pair 累计：
   - shared chunk count；
   - same sentence count；
   - optional role stats：entity-entity、entity-phrase、phrase-token。
4. 构建完成后计算 edge weight：

```text
weight = shared_count / sqrt(df_a * df_b)
```

可加入 same-sentence boost：

```text
weight *= (1 + alpha * same_sentence_count / shared_count)
```

5. 每个 node 只保留 top N neighbors。

### 5.4 新查询入口

新增：

```python
multihop_path_query(query_text, max_hops=2, beam_size=20, top_k_chain=10)
```

流程：

1. query sem nodes 作为起点。
2. 从起点在 `sem_cooccur_index` 中做 beam search。
3. 每条 path 维护：
   - visited sem nodes；
   - supporting chunk ids；
   - path edge weights；
   - query coverage；
   - bridge node properties。
4. 对 path 的末端节点或 shared chunks 做 chunk retrieval。
5. 输出 evidence chains。

### 5.5 HotpotQA 例子：属性约束先定位 bridge entity

`What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?` 不是简单的“已知 A，找到 B，再问 B 的属性”。它的第一跳更像是用多个属性约束共同定位 `University of Kansas`：

```text
University of Kansas sentence 1:
  The main campus in Lawrence ...

University of Kansas sentence 2:
  Two branch campuses are in the Kansas City metropolitan area ...

Kansas Song sentence 0:
  Kansas Song ... is a fight song of the University of Kansas.
```

这个样本说明全局 path index 不能只从 query 中的显式实体出发，也应支持从一组 query constraint 命中的 chunk 向外扩展：

```text
sem(main campus) + sem(Lawrence) + sem(branch campuses)
  -> chunk(University of Kansas)
  -> sem(University of Kansas)
  -> chunk(Kansas Song)
  -> sem(fight song)
```

因此 `multihop_path_query()` 的 beam state 不应只记录 path edge weights，还应记录“哪个中间节点由多个 query constraints 共同支持”。这类 bridge 的 `BridgePrior` 可以由局部 constraint coverage 提升，即使 bridge entity 本身没有在原 query 中以完整名称出现。

### 5.6 优点

- 真正支持“从已知概念走到未知桥接概念”的图遍历。
- 可解释性强：路径中的 sem nodes 和 shared chunks 都能展示。
- 可支持 2-hop，也能限制后扩展到 3-hop。

### 5.7 风险

- 全局 pair 数量可能很大，需要严格 fanout 控制。
- schema/pickle 兼容需要设计：如果把 index 持久化，必须更新 backward compatibility；如果运行时按需构建，则查询首次成本较高。
- 需要避免把 common entities 或泛词变成 hub，必须做 idf/df gating。

### 5.8 建议优先级

第二阶段。先用计划一验证桥接信号有效，再把高频操作固化为全局索引。

## 6. 实现计划三：Query Decomposition + Chain Reranking

### 6.1 目标

让系统根据问题类型选择检索策略，而不是所有问题都走同一个多跳流程。

新增模块建议：

```text
multihop_reasoning.py
```

核心对象：

```python
QuestionPlan
QueryUnit
HopConstraint
EvidenceChain
```

### 6.2 不依赖在线 LLM 的 decomposition

先实现规则/轻模型版本：

- 使用现有 spaCy entity / noun phrase 抽取。
- 用 wh-word 和 dependency pattern 判断目标属性。
- 用 HotpotQA `type` 仅作评测分层，不作为生产输入。
- 规则区分：
  - bridge：一个强 anchor entity + 一个未知目标属性。
  - comparison：两个并列 entity / noun phrase + 比较属性。
  - simple/mixed：回退到 `chunk_cooccur_query()`。

示例输出：

```python
QuestionPlan(
    mode="bridge",
    anchor_units=[...],
    constraint_units=[...],
    answer_focus="birthplace",
)
```

### 6.3 Comparison 问题策略

对 comparison 问题，不应强行桥接，而应：

1. 分别对两个 entity/side 做 focused retrieval。
2. 为每个 side 找 top chunks。
3. 用共享属性词或 question focus 做 rerank。
4. 返回两条并列 evidence chain。

这类问题的输出是：

```text
side A evidence + side B evidence
```

而不是：

```text
A -> bridge -> B
```

真实样本 `Were Scott Derrickson and Ed Wood of the same nationality?` 可以作为 comparison routing 的最小测试：

```text
side A: Scott Derrickson sentence 0 -> American director, screenwriter and producer
side B: Ed Wood sentence 0 -> American filmmaker, actor, writer, producer, and director
shared property: nationality
answer: yes
```

这个样本中 `Scott Derrickson` 和 `Ed Wood` 之间没有需要发现的 bridge entity。正确的 evidence chain 应该是两条并列证据：

```text
comparison_chain = {
  side_a: chunk/title for Scott Derrickson,
  side_b: chunk/title for Ed Wood,
  property: nationality,
  comparison_result_signal: both contain American
}
```

如果错误地走 bridge 扩展，系统可能把 `director`、`screenwriter`、`producer` 这类共现职业词当作桥接点，反而偏离问题。因此 `plan_query()` 只要检测到 `Are/Was/Were ... both/same`、`Who/Which is older/higher/more`、两个并列实体等模式，就应优先走 comparison focused retrieval，并把 bridge query 作为 fallback 而不是默认路径。

### 6.4 Chain reranking

计划一和计划二产生的候选 chain 都可以进入统一 reranker。

无 LLM reranker 版本：

- 结构特征：hop 数、是否同 chunk、是否跨文档、bridge idf、edge weight。
- 检索特征：first-hop score、second-hop score、BM25、query coverage。
- 语义特征：query embedding 与 concatenated evidence embedding 的相似度。
- 局部证据：same sentence、span 距离、modifier match。

可选 cross-encoder 版本：

- 使用已有 cross-encoder 依赖做 `query, evidence_chain_text` 相关性打分。
- 保持在线生成式 LLM 不参与。

### 6.5 优点

- 能分别优化 bridge 和 comparison。
- evidence chain 输出更稳定，适合论文/实验分析。
- 便于后续加入 LLM answer generation，但检索层仍可独立运行。

### 6.6 风险

- 规则 decomposition 容易覆盖不全。
- 如果没有人工标注 chain-level gold，调参可能只能看 doc recall/full hit，难定位错误。
- 需要较多 debug 工具支持。

### 6.7 建议优先级

第三阶段。它依赖计划一/二能产出足够好的候选 chain。

## 7. 实现计划四：句子级 Evidence Graph

### 7.1 目标

把推理粒度从 chunk 降到 sentence，减少 chunk 过长导致的伪共现。

当前 schema 7 已经保存：

- `ChunkNode.sentence_boundaries`
- `SpanOccurrence.sentence_id`

这些足以构建 sentence-level evidence graph，不需要重新切文档。

### 7.2 数据结构

新增运行时结构：

```python
sentence_evidence_index = {
    (chunk_id, sentence_id): {
        "sem_node_ids": set[int],
        "text": str,
        "doc_id": int,
    }
}
```

以及：

```python
sem_to_sentences = {
    sem_node_id: list[(chunk_id, sentence_id)]
}
```

### 7.3 用法

- first-hop evidence 不再只看 chunk，而看 sentence。
- bridge candidate 必须与 query anchor 同句或邻句，减少噪声。
- second-hop chain 可以返回两个 sentence，而不是两个完整 chunk。
- `chunk_cooccur_query()` 仍用于 candidate chunk recall，sentence graph 用于 rerank/explain。

### 7.4 优点

- 对多跳解释更友好。
- 能显著减少“同 chunk 但关系很弱”的伪 bridge。
- 复用现有 schema 7 字段。

### 7.5 风险

- 旧 pickle 可能没有 sentence id，需要 fallback。
- chunk text 中 sentence offset 到 sentence text 的切片要仔细处理。
- 如果原始 chunk 太短，句子级收益有限。

### 7.6 建议优先级

可作为计划一的增强，也可与计划二并行实现。

## 8. 推荐路线图

### Milestone 1：两阶段桥接检索 MVP

新增：

- `BridgeCandidate` / `EvidenceChain` dataclass。
- `LiteSemRAG.multihop_bridge_query()`。
- `_extract_bridge_candidates_from_chunks()`。
- `_score_bridge_candidate()`。
- `_retrieve_second_hop_for_bridge()`。
- `_rank_evidence_chains()`。

不改：

- `finalize()`。
- pickle schema。
- semantic node 构建逻辑。

验收：

- HotpotQA bridge 分层 `full_hit` 或 `recall@10_pq` 不低于 `chunk_cooccur_query()`。
- comparison 分层不明显回退；必要时默认只对疑似 bridge 问题启用。
- 返回 debug 信息能解释每条 chain。

### Milestone 2：全局 co-occurrence/path index

新增：

- `build_sem_cooccur_index()`。
- `get_sem_neighbors()`。
- `multihop_path_query()`。

关键约束：

- 每个 chunk 内最多参与 pair 构建的 sem nodes 数量可配置。
- 每个 sem node 只保留 top neighbors。
- 默认不持久化，先运行时构建；稳定后再考虑 schema 升级。

验收：

- 查询延迟可控。
- bridge 类问题相对 Milestone 1 有额外收益。
- hub node 不主导路径。

### Milestone 3：query decomposition 与策略路由

新增：

- `QuestionPlan`。
- `plan_query()`。
- `multihop_query()` 作为统一入口，内部路由到：
  - `chunk_cooccur_query()`；
  - `multihop_bridge_query()`；
  - comparison focused retrieval；
  - `multihop_path_query()`。

验收：

- HotpotQA bridge/comparison 分层都不低于 baseline。
- debug 输出能说明采用的 strategy。

### Milestone 4：sentence-level evidence graph

新增：

- `build_sentence_evidence_index()`。
- sentence-level local evidence / bridge filtering。
- chain evidence sentence rendering。

验收：

- evidence chain 可读性提升。
- 伪 bridge 数量下降。

## 9. 建议的代码落点

短期最小改动可以直接加在 `RAG_graph.py`：

- 新 dataclass 放在 `SpanOccurrence` / `SemNode` 附近。
- 新 query 方法放在 `chunk_cooccur_query()` 后面。
- 复用 `_nodes_share_sentence()`、`_sem_node_occurrences_in_chunk()` 等 helper。

中期建议拆模块：

```text
multihop_reasoning.py
```

放置：

- evidence chain dataclass；
- bridge candidate extraction；
- path search；
- question planning；
- chain reranking。

`LiteSemRAG` 只保留薄 wrapper：

```python
def multihop_query(...):
    from multihop_reasoning import multihop_query
    return multihop_query(self, ...)
```

这样可以避免 `RAG_graph.py` 继续膨胀。

## 10. 关键实现细节建议

### 10.1 Bridge candidate 过滤

优先选择：

- `token_node.node_type == "phrase"`；
- `phrase_type` 为 atomic/entity；
- `sem_node.description` 非空；
- IDF 高；
- df 不过大；
- surface text 首字母大写或包含专名形态；
- 与 query anchor 同句出现。

排除：

- 已经在 query 中出现的 token；
- 长度为 1 且低 IDF 的普通词；
- df 过高的 hub；
- purely modifier 且无独立语义的节点；
- BM25 贡献过低的节点。

结合 HotpotQA 样本，bridge candidate 还应加入两条正向特征：

- `bridge_title_match_like`：candidate surface 与某个文档标题高度一致时加权。例如 `Shirley Temple`、`Androscoggin Bank Colisée`、`University of Kansas` 都是第二跳 gold title。
- `constraint_supported_bridge`：candidate 所在 chunk 同时覆盖多个 query constraints 时加权。例如 `University of Kansas` 由 `Lawrence` 和 `Kansas City metropolitan area` 两个约束共同支持。

### 10.2 Chain 去重

同一 bridge 可能产生大量相似 chain。建议按以下 key 去重：

```text
(hop1_doc, bridge_token_norm, hop2_doc)
```

或者更细：

```text
(hop1_chunk_id, bridge_sem_node_id, hop2_chunk_id)
```

最终返回 chunk ids 时，应避免一个 query 返回 10 个来自同一 bridge 的近重复 chunk。

### 10.3 Debug 输出

每条 chain 至少包含：

- `strategy`
- `score`
- `hop1_chunk_id`
- `bridge_sem_node_id`
- `bridge_text`
- `hop2_chunk_id`
- `first_hop_score`
- `bridge_prior`
- `local_bridge_evidence`
- `second_hop_score`
- `constraint_coverage`
- `why_selected`

这比只返回 chunk ids 更重要，因为多跳效果失败时必须知道是 query 解析错、bridge 选错，还是 second-hop rerank 错。

### 10.4 与现有 API 的兼容

不要替换 `chunk_cooccur_query()`。新增入口应独立：

```python
multihop_bridge_query(...)
multihop_path_query(...)
multihop_query(...)
```

原因：

- 现有 SciFact/FEVER/HotpotQA 评测脚本依赖 `chunk_cooccur_query()`。
- 单跳检索和多跳检索目标不同，强行合并会增加回归风险。

## 11. 评测方案

### 11.1 HotpotQA

复用 `eval_hotpot_litesem_inproc.py`：

- 按 `all / bridge / comparison` 分层。
- 对比 baseline：
  - `chunk_cooccur_query()`
  - `multihop_bridge_query()`
  - `multihop_query()` strategy router

重点看：

- bridge `full_hit`；
- bridge `recall@10_pq`；
- all `MRR@10`；
- comparison 是否回退。

除 aggregate metrics 外，建议固定一组 smoke cases，用于每次改动后人工检查 debug chain：

| Case | 期望行为 |
| --- | --- |
| `Kiss and Tell -> Shirley Temple` | top chains 中出现 bridge `Shirley Temple`，两个 gold titles 都在 returned chunks/docs 内。 |
| `Lewiston Maineiacs -> Androscoggin Bank Colisée` | bridge candidate 排名前列包含 `Androscoggin Bank Colisée`，second-hop evidence 包含容量句。 |
| `University of Kansas -> Kansas Song` | first-hop 可由地理约束定位 `University of Kansas`，second-hop 能跳到 `Kansas Song`。 |
| `Scott Derrickson vs Ed Wood` | strategy router 选择 comparison，输出两条 side evidence，而不是单条 bridge chain。 |

这些 case 不替代完整评测，但能快速暴露三类常见错误：bridge 没抽出、second-hop 硬过滤过强、comparison 被误路由到 bridge。

### 11.2 消融实验

建议做：

- bridge candidate 是否要求同句；
- IDF gating 阈值；
- bridge_top_k；
- second-hop 是否加入剩余 query constraints；
- 是否允许 hop1 == hop2 chunk；
- 是否使用 sentence-level evidence rerank。

### 11.3 错误分析

每个失败样本输出：

- top chains；
- bridge candidates；
- first-hop chunks；
- second-hop chunks；
- gold supporting titles；
- miss reason。

这比只看 aggregate metrics 更有价值。

## 12. 总体建议

不要一开始就做复杂的通用 graph reasoning。当前系统最有价值的复用点是：query semantic matching 已经成熟，chunk/sem 反向边已经存在，HotpotQA bridge/comparison 评测也已经有脚本。因此最务实的路线是先实现两阶段桥接检索，把“第一跳 chunk 中出现但 query 未显式给出的高质量 semantic node”作为 bridge，再用它检索第二跳证据。

如果这个 MVP 在 bridge 类问题上有收益，再把按需桥接扩展固化为全局 co-occurrence/path index。最后再加入 question decomposition 和 sentence-level evidence graph，形成稳定的多跳推理入口。
