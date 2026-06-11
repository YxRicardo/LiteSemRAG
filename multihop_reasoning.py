"""多跳推理 —— 实现计划一:最小侵入的两阶段桥接检索。

对应 multihop_reasoning_implementation_plan.md 的 "实现计划一 / Milestone 1"。

不改索引结构、不改 pickle schema、不改 finalize / semantic node 构建逻辑,只在
查询端复用现有能力实现 bridge 类多跳:

    query
      -> 第一跳: chunk_cooccur_query() 召回 anchor chunk
      -> 从第一跳 chunk 的 sem_node_list 抽取 bridge candidates(query 中未显式给出的高质量语义节点)
      -> 第二跳: 用 bridge sem node 的 BM25 找证据 chunk,并用剩余 query 约束做 soft rerank
      -> 生成 evidence chain,按 ChainScore 排序,聚合去重后返回 chunk

ChainScore (计划 §4.6,全部因子归一到 (0,1],相乘):

    ChainScore =
        FirstHopScore        # 第一跳 chunk 的归一化 final_score
      * BridgePrior          # bridge idf 归一 * 类型/标题系数
      * LocalBridgeEvidence  # bridge 与 query 节点同句=1.0 / 同 chunk=0.6
      * SecondHopScore       # bridge 在第二跳 chunk 的归一化 BM25
      * ConstraintCoverage   # 第二跳对"剩余 query 约束/answer focus"的 soft 覆盖
      * DiversityPenalty     # hop1==hop2 同 chunk 时降权,避免伪多跳

入口在 LiteSemRAG.multihop_bridge_query() 里以薄 wrapper 委托到本模块,
现有 chunk_cooccur_query() / SciFact / FEVER / HotpotQA 评测口径完全不受影响。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from text_processing import normalize_text


# ----------------------------- 默认超参 -----------------------------
# 第一跳召回的 anchor chunk 数。
DEFAULT_FIRST_HOP_K = 20
# 总共保留的 bridge candidate 数(跨所有第一跳 chunk 的全局 fanout 上限)。
DEFAULT_BRIDGE_TOP_K = 8
# 每个 bridge 第二跳取的 chunk 数。
DEFAULT_SECOND_HOP_K = 20
# 返回的 evidence chain 数。
DEFAULT_TOP_K_CHAIN = 10

# bridge candidate 过滤阈值。
# idf 低于此值视为泛词,不做 bridge。
DEFAULT_MIN_BRIDGE_IDF = 3.0
# df 高于 (语料 chunk 数 * 此比例) 视为 hub,不做 bridge。
DEFAULT_DF_HUB_RATIO = 0.10
# 单 token bridge 的更严 idf 门槛(单字泛词噪声大)。
DEFAULT_MIN_SINGLE_TOKEN_IDF = 4.0

# 局部证据等级(计划 §4.6)。
LOCAL_EVIDENCE_SENTENCE = 1.0
LOCAL_EVIDENCE_CHUNK = 0.6
# hop1 == hop2 同 chunk 的伪多跳惩罚。
DIVERSITY_PENALTY_SAME_CHUNK = 0.3

# BridgePrior 类型系数。
BRIDGE_COEF_PHRASE = 1.15          # 多词短语
BRIDGE_COEF_FORCE_SINGLE = 1.10    # entity / 原子短语 (force_single_semantic)
BRIDGE_COEF_HAS_DESC = 1.10        # 带 description 的语义节点
BRIDGE_COEF_CAPITALIZED = 1.10     # 专名形态(首字母大写)
BRIDGE_COEF_TITLE_MATCH = 1.30     # surface 与某文档标题一致(HotpotQA 第二跳 gold title 强信号)

# 第二跳目标 = bridge 实体自己的同名页面时的强加权(HotpotQA bridge 的典型结构:
# 第一跳文档提到桥接实体 B,第二个 gold 文档就是 B 自己的 wiki 页面)。
HOP2_TITLE_MATCH_BONUS = 1.80

# ConstraintCoverage soft 下限与残余约束权重(避免硬过滤误杀正确二跳)。
CONSTRAINT_COVERAGE_FLOOR = 0.30
CONSTRAINT_COVERAGE_SPAN = 0.70
# 没有残余约束(问题在第一跳已被完整覆盖)时给的中性覆盖值。
CONSTRAINT_COVERAGE_NO_RESIDUAL = 0.50

# 最终 chunk 聚合(aggregate_chunk_ranking)默认权重:
# 第二跳是 bridge 带来的新证据 -> 全权;第一跳已在第一跳列表里 -> 半权,避免打乱内部排序压低 MRR。
AGG_HOP2_BONUS_WEIGHT = 1.0
AGG_HOP1_BONUS_WEIGHT = 0.5
# 只有 chain.score 达到此门槛才参与加成(过滤 comparison 等弱噪声 chain)。
# 默认 0.2:HotpotQA 500 上 bridge full_hit/recall/mrr 三项全涨,comparison full_hit 零回退。
AGG_MIN_CHAIN_BONUS_SCORE = 0.2


@dataclass
class BridgeCandidate:
    """从第一跳 chunk 抽出的桥接语义节点候选。"""

    sem_node: object
    hop1_chunk_id: int
    token_norm: str
    idf: float
    df: int
    bridge_prior: float
    local_evidence: float
    is_phrase: bool
    is_force_single: bool
    has_description: bool
    is_capitalized: bool
    title_match: bool


@dataclass
class EvidenceChain:
    """一条 query concept -> bridge -> answer evidence 的证据链。"""

    strategy: str
    score: float
    hop1_chunk_id: int
    hop1_doc: str
    bridge_sem_node_id: int
    bridge_text: str
    hop2_chunk_id: int
    hop2_doc: str
    first_hop_score: float
    bridge_prior: float
    local_bridge_evidence: float
    second_hop_score: float
    constraint_coverage: float
    diversity_penalty: float
    why_selected: str = ""

    def as_debug(self) -> dict:
        return {
            "strategy": self.strategy,
            "score": round(self.score, 6),
            "hop1_chunk_id": self.hop1_chunk_id,
            "hop1_doc": self.hop1_doc,
            "bridge_sem_node_id": self.bridge_sem_node_id,
            "bridge_text": self.bridge_text,
            "hop2_chunk_id": self.hop2_chunk_id,
            "hop2_doc": self.hop2_doc,
            "first_hop_score": round(self.first_hop_score, 6),
            "bridge_prior": round(self.bridge_prior, 6),
            "local_bridge_evidence": round(self.local_bridge_evidence, 6),
            "second_hop_score": round(self.second_hop_score, 6),
            "constraint_coverage": round(self.constraint_coverage, 6),
            "diversity_penalty": round(self.diversity_penalty, 6),
            "why_selected": self.why_selected,
        }


# ----------------------------- 内部 helper -----------------------------
def _doc_name(db, chunk_id):
    return db.chunk_nodes[chunk_id].doc_node.doc_name


def _collect_query_context(db, resolved_matches):
    """从 resolved query 里收集多跳要用的 query 上下文。

    返回:
      query_sems        : 去重后的 query sem node 列表
      query_sem_ids     : query sem node 的 sem_node_id 集合(bridge 排除用)
      query_token_norms : query token / query sem token 的归一化集合(bridge 排除用)
    """
    low_level_sems, high_level_sems = db._collect_query_cooccurrence_sems(
        resolved_matches, False
    )
    query_sems = []
    query_sem_ids = set()
    query_token_norms = set()

    for sem_info in low_level_sems + high_level_sems:
        sem_node = sem_info[0]
        if sem_node is None:
            continue
        sid = sem_node.sem_node_id
        if sid not in query_sem_ids:
            query_sem_ids.add(sid)
            query_sems.append(sem_node)
        norm = normalize_text(sem_node.token_node.token_text)
        if norm:
            query_token_norms.add(norm)
            query_token_norms.update(norm.split())

    # raw query token 也加入排除集(包含没解析成 sem 的词)。
    for match in resolved_matches:
        norm = normalize_text(match.get("token", ""))
        if norm:
            query_token_norms.add(norm)
            query_token_norms.update(norm.split())

    return query_sems, query_sem_ids, query_token_norms


def _chunk_sem_id_set(chunk_node):
    return {sem.sem_node_id for sem in getattr(chunk_node, "sem_node_list", []) or []}


def _surface_is_capitalized(sem_node, chunk_id):
    """该 sem node 在指定 chunk 内是否以专名形态(首字母大写)出现。"""
    for occ in getattr(sem_node, "span_occurrences", []) or []:
        cn = getattr(occ, "chunk_node", None)
        if cn is None or cn.chunk_node_id != chunk_id:
            continue
        surface = getattr(occ, "surface_text", None) or getattr(occ, "span_text", None)
        if surface and surface[:1].isupper():
            return True
    return False


# ----------------------------- bridge 抽取 -----------------------------
def extract_bridge_candidates_from_chunks(
    db,
    first_hop_chunk_ids,
    query_sems,
    query_sem_ids,
    query_token_norms,
    title_norm_set,
    max_corpus_idf,
    share_sentence_cache,
    *,
    min_bridge_idf=DEFAULT_MIN_BRIDGE_IDF,
    df_hub_ratio=DEFAULT_DF_HUB_RATIO,
    min_single_token_idf=DEFAULT_MIN_SINGLE_TOKEN_IDF,
    require_same_sentence=False,
    bridge_top_k=DEFAULT_BRIDGE_TOP_K,
):
    """从第一跳 chunk 的 sem_node_list 抽取并打分 bridge candidates(计划 §4.4 / §10.1)。"""
    n_chunks = max(len(db.chunk_nodes), 1)
    df_hub_threshold = df_hub_ratio * n_chunks

    # query sems 按 hop1 chunk 分组,用于 LocalBridgeEvidence / 同句判定。
    candidates = {}  # token_norm -> 最佳 BridgeCandidate(同一 bridge 词只保留最强一条)

    for hop1_chunk_id in first_hop_chunk_ids:
        chunk_node = db.chunk_nodes[hop1_chunk_id]
        chunk_sem_ids = _chunk_sem_id_set(chunk_node)
        # 该 chunk 内出现的 query sems(用于局部证据)。
        query_sems_in_chunk = [s for s in query_sems if s.sem_node_id in chunk_sem_ids]

        for sem_node in getattr(chunk_node, "sem_node_list", []) or []:
            token_node = sem_node.token_node
            token_norm = normalize_text(token_node.token_text)
            if not token_norm:
                continue
            # 排除 query 已显式命中的节点 / query 词。
            if sem_node.sem_node_id in query_sem_ids or token_norm in query_token_norms:
                continue

            idf = float(getattr(token_node, "idf", 0.0) or 0.0)
            df = int(getattr(token_node, "df", 0) or 0)
            is_phrase = token_node.node_type == "phrase"
            is_force_single = bool(getattr(token_node, "force_single_semantic", False))
            has_desc = bool(getattr(sem_node, "description", None))
            is_capitalized = _surface_is_capitalized(sem_node, hop1_chunk_id)
            title_match = token_norm in title_norm_set

            # ---- 泛词 / hub 过滤 ----
            if idf < min_bridge_idf and not (is_phrase or title_match):
                continue
            if not is_phrase and idf < min_single_token_idf and not title_match:
                continue
            if df > df_hub_threshold and not title_match:
                continue

            # ---- LocalBridgeEvidence ----
            local_evidence = LOCAL_EVIDENCE_CHUNK
            for q_sem in query_sems_in_chunk:
                if db._nodes_share_sentence(
                    q_sem, sem_node, hop1_chunk_id, share_sentence_cache
                ):
                    local_evidence = LOCAL_EVIDENCE_SENTENCE
                    break
            if require_same_sentence and local_evidence < LOCAL_EVIDENCE_SENTENCE:
                continue

            # ---- BridgePrior ----
            prior = min(idf / max_corpus_idf, 1.0) if max_corpus_idf > 0 else 0.0
            coef = 1.0
            if is_phrase:
                coef *= BRIDGE_COEF_PHRASE
            if is_force_single:
                coef *= BRIDGE_COEF_FORCE_SINGLE
            if has_desc:
                coef *= BRIDGE_COEF_HAS_DESC
            if is_capitalized:
                coef *= BRIDGE_COEF_CAPITALIZED
            if title_match:
                coef *= BRIDGE_COEF_TITLE_MATCH
            bridge_prior = prior * coef

            cand = BridgeCandidate(
                sem_node=sem_node,
                hop1_chunk_id=hop1_chunk_id,
                token_norm=token_norm,
                idf=idf,
                df=df,
                bridge_prior=bridge_prior,
                local_evidence=local_evidence,
                is_phrase=is_phrase,
                is_force_single=is_force_single,
                has_description=has_desc,
                is_capitalized=is_capitalized,
                title_match=title_match,
            )
            # 同一 bridge 词跨多个 hop1 chunk 时,保留 prior*local 最强的一条。
            prev = candidates.get(token_norm)
            if prev is None or (cand.bridge_prior * cand.local_evidence) > (
                prev.bridge_prior * prev.local_evidence
            ):
                candidates[token_norm] = cand

    ranked = sorted(
        candidates.values(),
        key=lambda c: c.bridge_prior * c.local_evidence,
        reverse=True,
    )
    return ranked[:bridge_top_k]


# ----------------------------- 第二跳检索 -----------------------------
def retrieve_second_hop_for_bridge(bridge_sem_node, second_hop_k):
    """取 bridge sem node BM25 最高的 second_hop_k 个 chunk。

    返回 [(chunk_id, normalized_bm25), ...],bm25 归一到 (0,1]。
    """
    bm25 = getattr(bridge_sem_node, "BM25", None) or {}
    if not bm25:
        return []
    max_bm25 = max(bm25.values())
    if max_bm25 <= 0:
        return []
    ranked = sorted(bm25.items(), key=lambda kv: kv[1], reverse=True)[:second_hop_k]
    return [(cid, score / max_bm25) for cid, score in ranked]


def _constraint_coverage(db, residual_query_sems, hop2_chunk_id):
    """第二跳 chunk 对剩余 query 约束的 idf 加权 soft 覆盖(计划 §4.6 ConstraintCoverage)。"""
    if not residual_query_sems:
        return CONSTRAINT_COVERAGE_NO_RESIDUAL
    hop2_sem_ids = _chunk_sem_id_set(db.chunk_nodes[hop2_chunk_id])
    total_w = 0.0
    covered_w = 0.0
    for sem in residual_query_sems:
        w = float(getattr(sem.token_node, "idf", 0.0) or 0.0) + 1e-6
        total_w += w
        if sem.sem_node_id in hop2_sem_ids:
            covered_w += w
    coverage = covered_w / total_w if total_w > 0 else 0.0
    return CONSTRAINT_COVERAGE_FLOOR + CONSTRAINT_COVERAGE_SPAN * coverage


# ----------------------------- 链路评分主流程 -----------------------------
def rank_evidence_chains(
    db,
    first_hop_score_map,
    bridge_candidates,
    query_sems,
    title_to_chunk,
    *,
    second_hop_k=DEFAULT_SECOND_HOP_K,
    top_k_chain=DEFAULT_TOP_K_CHAIN,
    allow_same_chunk=False,
):
    """对 (hop1, bridge, hop2) 三元组生成并排序 evidence chain。"""
    chains = []
    seen_keys = set()  # (hop1_doc, bridge_token_norm, hop2_doc) 去重

    for cand in bridge_candidates:
        hop1_chunk_id = cand.hop1_chunk_id
        hop1_doc = _doc_name(db, hop1_chunk_id)
        first_hop_score = first_hop_score_map.get(hop1_chunk_id, 0.0)
        # 残余约束 = hop1 chunk 未覆盖的 query sems(自然落到 answer focus 上)。
        hop1_sem_ids = _chunk_sem_id_set(db.chunk_nodes[hop1_chunk_id])
        residual_query_sems = [s for s in query_sems if s.sem_node_id not in hop1_sem_ids]

        second_hop = retrieve_second_hop_for_bridge(cand.sem_node, second_hop_k)
        # 注入 bridge 实体自己的同名页面作为第二跳候选(HotpotQA 第二跳 gold 的典型),
        # 即使它不在 bridge BM25 的 top-second_hop_k 内也强制纳入。
        bridge_title_chunk = title_to_chunk.get(cand.token_norm)
        if bridge_title_chunk is not None and bridge_title_chunk != hop1_chunk_id:
            if all(cid != bridge_title_chunk for cid, _ in second_hop):
                second_hop.append((bridge_title_chunk, 1.0))

        for hop2_chunk_id, second_hop_score in second_hop:
            same_chunk = hop2_chunk_id == hop1_chunk_id
            if same_chunk and not allow_same_chunk:
                continue
            hop2_doc = _doc_name(db, hop2_chunk_id)
            key = (hop1_doc, cand.token_norm, hop2_doc)
            if key in seen_keys:
                continue

            diversity = DIVERSITY_PENALTY_SAME_CHUNK if same_chunk else 1.0
            coverage = _constraint_coverage(db, residual_query_sems, hop2_chunk_id)
            # 第二跳目标正是 bridge 实体的同名页面 -> 强加权。
            hop2_title_match = normalize_text(hop2_doc) == cand.token_norm
            title_bonus = HOP2_TITLE_MATCH_BONUS if hop2_title_match else 1.0
            score = (
                first_hop_score
                * cand.bridge_prior
                * cand.local_evidence
                * second_hop_score
                * coverage
                * diversity
                * title_bonus
            )
            if score <= 0:
                continue
            seen_keys.add(key)

            tags = []
            if cand.title_match:
                tags.append("title-match")
            if hop2_title_match:
                tags.append("hop2-own-page")
            if cand.is_phrase:
                tags.append("phrase")
            if cand.is_force_single:
                tags.append("entity")
            if cand.local_evidence >= LOCAL_EVIDENCE_SENTENCE:
                tags.append("same-sentence")
            why = (
                f"bridge '{cand.sem_node.token_node.token_text}' "
                f"[{','.join(tags) if tags else 'generic'}] "
                f"links '{hop1_doc}' -> '{hop2_doc}'"
            )

            chains.append(
                EvidenceChain(
                    strategy="bridge",
                    score=score,
                    hop1_chunk_id=hop1_chunk_id,
                    hop1_doc=hop1_doc,
                    bridge_sem_node_id=cand.sem_node.sem_node_id,
                    bridge_text=cand.sem_node.token_node.token_text,
                    hop2_chunk_id=hop2_chunk_id,
                    hop2_doc=hop2_doc,
                    first_hop_score=first_hop_score,
                    bridge_prior=cand.bridge_prior,
                    local_bridge_evidence=cand.local_evidence,
                    second_hop_score=second_hop_score,
                    constraint_coverage=coverage,
                    diversity_penalty=diversity,
                    why_selected=why,
                )
            )

    chains.sort(key=lambda c: c.score, reverse=True)
    return chains[:top_k_chain]


def aggregate_chunk_ranking(
    first_hop_chunk_ids,
    first_hop_score_map,
    chains,
    *,
    hop1_bonus_weight=AGG_HOP1_BONUS_WEIGHT,
    hop2_bonus_weight=AGG_HOP2_BONUS_WEIGHT,
    min_chain_bonus_score=AGG_MIN_CHAIN_BONUS_SCORE,
):
    """把第一跳证据与 chain 两端证据按组合分数排成最终 chunk 列表。

    combined(chunk) = first_hop_score(chunk)               # 单跳 anchor / comparison 两侧
                    + hop2_bonus_weight * chain.score       # 该 chunk 是某 chain 的第二跳证据
                    + hop1_bonus_weight * chain.score       # 该 chunk 是某 chain 的第一跳证据

    第二跳加成(新证据,bridge 目标)权重高,第一跳加成(已在第一跳列表里)权重低,
    避免 comparison 等弱 chain 把第一跳内部排序打乱、压低首个 gold 的 MRR;同时保留
    "把较低排名的第二个 gold 提进 top-k" 的能力。只有 chain.score >= 门槛才计加成。
    """
    combined = {}
    # 第一跳基础分。
    for cid in first_hop_chunk_ids:
        combined[cid] = first_hop_score_map.get(cid, 0.0)
    # chain 加成:hop2 权重高、hop1 权重低,取各 chunk 的最高加成。
    bonus = {}
    for chain in chains:
        if chain.score < min_chain_bonus_score:
            continue
        b2 = hop2_bonus_weight * chain.score
        b1 = hop1_bonus_weight * chain.score
        if b2 > bonus.get(chain.hop2_chunk_id, 0.0):
            bonus[chain.hop2_chunk_id] = b2
        if b1 > bonus.get(chain.hop1_chunk_id, 0.0):
            bonus[chain.hop1_chunk_id] = b1
    for cid, b in bonus.items():
        combined[cid] = combined.get(cid, first_hop_score_map.get(cid, 0.0)) + b
    # 稳定排序:组合分降序,平分时第一跳分降序、chunk_id 升序。
    ranked = sorted(
        combined.items(),
        key=lambda kv: (-kv[1], -first_hop_score_map.get(kv[0], 0.0), kv[0]),
    )
    return [cid for cid, _ in ranked]


# ----------------------------- 对外入口 -----------------------------
def multihop_bridge_query(
    db,
    query_text,
    top_k_chain=DEFAULT_TOP_K_CHAIN,
    first_hop_k=DEFAULT_FIRST_HOP_K,
    bridge_top_k=DEFAULT_BRIDGE_TOP_K,
    second_hop_k=DEFAULT_SECOND_HOP_K,
    *,
    search_mode="broad",
    require_same_sentence=False,
    allow_same_chunk=False,
    min_bridge_idf=DEFAULT_MIN_BRIDGE_IDF,
    df_hub_ratio=DEFAULT_DF_HUB_RATIO,
    min_single_token_idf=DEFAULT_MIN_SINGLE_TOKEN_IDF,
    hop1_bonus_weight=AGG_HOP1_BONUS_WEIGHT,
    hop2_bonus_weight=AGG_HOP2_BONUS_WEIGHT,
    min_chain_bonus_score=AGG_MIN_CHAIN_BONUS_SCORE,
    idf_prune_tau=None,
    idf_prune_min_max_idf=None,
    idf_prune_keep_top=2,
    print_important_tokens=False,
):
    """两阶段桥接检索(计划一)。

    返回 (chains, retrieved_chunk_ids, debug_info):
      chains             : top EvidenceChain 列表
      retrieved_chunk_ids: 第一跳与 bridge 第二跳证据交错合并、去重后的 chunk id 列表
                           (供 HotpotQA 等下游做 chunk->title 评测)
      debug_info         : 解释每条 chain 以及第一跳 / bridge 的结构化调试信息
    """
    # ---- 第一跳: 复用 chunk_cooccur_query ----
    _, first_hop_chunk_ids, cooccur_debug = db.chunk_cooccur_query(
        query_text,
        top_k_chunk=first_hop_k,
        print_important_tokens=False,
        search_mode=search_mode,
        idf_prune_tau=idf_prune_tau,
        idf_prune_min_max_idf=idf_prune_min_max_idf,
        idf_prune_keep_top=idf_prune_keep_top,
    )
    resolved_matches = cooccur_debug["resolved_matches"]

    # 第一跳 chunk 归一化 final_score。
    all_scored = cooccur_debug.get("all_scored_chunks", []) or []
    max_final = max((r["final_score"] for r in all_scored), default=0.0)
    if max_final > 0:
        first_hop_score_map = {
            r["chunk_id"]: r["final_score"] / max_final for r in all_scored
        }
    else:
        first_hop_score_map = {r["chunk_id"]: 0.0 for r in all_scored}

    # ---- query 上下文 ----
    query_sems, query_sem_ids, query_token_norms = _collect_query_context(
        db, resolved_matches
    )

    # 语料级最大 idf + 标题集合(只算一次,可缓存在 db 上)。
    max_corpus_idf = getattr(db, "_multihop_max_idf", None)
    if max_corpus_idf is None:
        max_corpus_idf = max(
            (float(getattr(t, "idf", 0.0) or 0.0) for t in db.token_nodes),
            default=1.0,
        )
        db._multihop_max_idf = max_corpus_idf
    title_norm_set = getattr(db, "_multihop_title_norms", None)
    title_to_chunk = getattr(db, "_multihop_title_to_chunk", None)
    if title_norm_set is None or title_to_chunk is None:
        title_norm_set = set()
        title_to_chunk = {}
        for d in db.doc_nodes:
            norm = normalize_text(d.doc_name) if d.doc_name else ""
            if not norm:
                continue
            title_norm_set.add(norm)
            chunk_list = getattr(d, "chunk_node_list", []) or []
            if chunk_list and norm not in title_to_chunk:
                # 取该文档第一个 chunk 作为标题页代表(本语料多为 1 doc=1 chunk)。
                title_to_chunk[norm] = chunk_list[0].chunk_node_id
        db._multihop_title_norms = title_norm_set
        db._multihop_title_to_chunk = title_to_chunk

    share_sentence_cache = {}

    # ---- bridge 抽取 ----
    bridge_candidates = extract_bridge_candidates_from_chunks(
        db,
        first_hop_chunk_ids,
        query_sems,
        query_sem_ids,
        query_token_norms,
        title_norm_set,
        max_corpus_idf,
        share_sentence_cache,
        min_bridge_idf=min_bridge_idf,
        df_hub_ratio=df_hub_ratio,
        min_single_token_idf=min_single_token_idf,
        require_same_sentence=require_same_sentence,
        bridge_top_k=bridge_top_k,
    )

    # ---- 链路生成与排序 ----
    chains = rank_evidence_chains(
        db,
        first_hop_score_map,
        bridge_candidates,
        query_sems,
        title_to_chunk,
        second_hop_k=second_hop_k,
        top_k_chain=top_k_chain,
        allow_same_chunk=allow_same_chunk,
    )

    # ---- 聚合 chunk: 组合分数排序 ----
    # base = 第一跳归一化分数(保留单跳 anchor / comparison 两侧证据);
    # bonus = 该 chunk 作为某条 chain 的 hop1 或 hop2 时的最高 chain 分数(把 bridge
    # 两端证据顶到前面)。两者相加排序,bridge 两个 gold 与 comparison 两侧都能尽量进 top-k。
    retrieved_chunk_ids = aggregate_chunk_ranking(
        first_hop_chunk_ids,
        first_hop_score_map,
        chains,
        hop1_bonus_weight=hop1_bonus_weight,
        hop2_bonus_weight=hop2_bonus_weight,
        min_chain_bonus_score=min_chain_bonus_score,
    )

    if print_important_tokens:
        print(f"[multihop] query sems: {[s.token_node.token_text for s in query_sems]}")
        print(f"[multihop] bridges: {[c.token_norm for c in bridge_candidates]}")
        for c in chains:
            print(f"  chain score={c.score:.5f} :: {c.why_selected}")

    debug_info = {
        "params": {
            "top_k_chain": top_k_chain,
            "first_hop_k": first_hop_k,
            "bridge_top_k": bridge_top_k,
            "second_hop_k": second_hop_k,
            "require_same_sentence": require_same_sentence,
            "allow_same_chunk": allow_same_chunk,
        },
        "query_sems": [s.token_node.token_text for s in query_sems],
        "first_hop_chunk_ids": list(first_hop_chunk_ids),
        "first_hop_docs": [_doc_name(db, c) for c in first_hop_chunk_ids],
        "bridge_candidates": [
            {
                "bridge": c.token_norm,
                "text": c.sem_node.token_node.token_text,
                "idf": round(c.idf, 3),
                "df": c.df,
                "bridge_prior": round(c.bridge_prior, 4),
                "local_evidence": c.local_evidence,
                "hop1_doc": _doc_name(db, c.hop1_chunk_id),
                "title_match": c.title_match,
                "is_phrase": c.is_phrase,
                "is_entity": c.is_force_single,
            }
            for c in bridge_candidates
        ],
        "chains": [c.as_debug() for c in chains],
        "cooccur_debug": cooccur_debug,
    }

    return chains, retrieved_chunk_ids, debug_info
