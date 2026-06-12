"""Multi-hop reasoning: Implementation Plan 1, a minimally invasive two-stage
bridge retrieval pipeline.

This corresponds to "Implementation Plan 1 / Milestone 1" in
multihop_reasoning_implementation_plan.md.

It does not modify the index structure, pickle schema, or finalize / semantic
node construction logic. It only reuses existing query-time capabilities to
implement bridge-style multi-hop retrieval:

    query
      -> first hop: chunk_cooccur_query() recalls anchor chunks
      -> extract bridge candidates from each first-hop chunk's sem_node_list
         (high-quality semantic nodes not explicitly mentioned in the query)
      -> second hop: use bridge sem-node BM25 to find evidence chunks, then do
         a soft rerank with the remaining query constraints
      -> generate evidence chains, rank by ChainScore, aggregate, deduplicate,
         and return chunks

ChainScore (plan §4.6, all factors normalized to (0,1] and multiplied):

    ChainScore =
        FirstHopScore        # normalized final_score of the first-hop chunk
      * BridgePrior          # normalized bridge IDF * type/title coefficient
      * LocalBridgeEvidence  # bridge appears in same sentence as query node=1.0 / same chunk=0.6
      * SecondHopScore       # normalized BM25 of the bridge in the second-hop chunk
      * ConstraintCoverage   # soft coverage of remaining query constraints / answer focus by the second hop
      * DiversityPenalty     # downweight hop1==hop2 within the same chunk to avoid fake multi-hop

LiteSemRAG.multihop_bridge_query() delegates here through a thin wrapper, so
existing chunk_cooccur_query() / SciFact / FEVER / HotpotQA evaluation behavior
remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from text_processing import normalize_text


# ----------------------------- Default hyperparameters -----------------------------
# Number of anchor chunks recalled in the first hop.
DEFAULT_FIRST_HOP_K = 20
# Total number of bridge candidates retained (global fanout cap across all
# first-hop chunks).
DEFAULT_BRIDGE_TOP_K = 8
# Number of second-hop chunks taken per bridge.
DEFAULT_SECOND_HOP_K = 20
# Number of evidence chains returned.
DEFAULT_TOP_K_CHAIN = 10

# Bridge-candidate filtering thresholds.
# DF above (corpus chunk count * this ratio) is treated as a hub and is not
# used as a bridge.
DEFAULT_DF_HUB_RATIO = 0.10
# Stricter IDF threshold for single-token bridges, which are noisier.
DEFAULT_MIN_SINGLE_TOKEN_IDF = 4.0

# Local evidence levels (plan §4.6).
LOCAL_EVIDENCE_SENTENCE = 1.0
LOCAL_EVIDENCE_CHUNK = 0.6
# Penalty for fake multi-hop when hop1 == hop2 in the same chunk.
DIVERSITY_PENALTY_SAME_CHUNK = 0.3

# BridgePrior type coefficients.
BRIDGE_COEF_PHRASE = 1.15          # multi-word phrase
BRIDGE_COEF_FORCE_SINGLE = 1.10    # entity / atomic phrase (force_single_semantic)
BRIDGE_COEF_HAS_DESC = 1.10        # semantic node with a description
BRIDGE_COEF_CAPITALIZED = 1.10     # proper-name surface form (initial capital)
BRIDGE_COEF_TITLE_MATCH = 1.30     # surface matches a document title (strong HotpotQA second-hop gold-title signal)

# Strong upweighting when the second-hop target is the bridge entity's own
# title-matching page, a common HotpotQA bridge pattern:
# the first-hop document mentions bridge entity B, and the second gold document
# is B's own wiki page.
HOP2_TITLE_MATCH_BONUS = 1.80

# ConstraintCoverage soft lower bound and residual-constraint span, to avoid
# hard filtering out correct second hops.
CONSTRAINT_COVERAGE_FLOOR = 0.30
CONSTRAINT_COVERAGE_SPAN = 0.70
# Neutral coverage value when no residual constraints remain.
CONSTRAINT_COVERAGE_NO_RESIDUAL = 0.50

# Default weights for final chunk aggregation (aggregate_chunk_ranking):
# second-hop evidence is new bridge evidence and gets full weight; first-hop
# evidence is already present in the first-hop list and only gets half weight,
# to avoid hurting MRR by disturbing the internal ranking too much.
AGG_HOP2_BONUS_WEIGHT = 1.0
AGG_HOP1_BONUS_WEIGHT = 0.5
# Only chains with score above this threshold contribute bonus weight, which
# filters weak noisy chains such as comparison cases.
# The default 0.2 improved bridge full_hit/recall/MRR on HotpotQA 500 with no
# regression on comparison full_hit.
AGG_MIN_CHAIN_BONUS_SCORE = 0.2


@dataclass
class BridgeCandidate:
    """Bridge semantic-node candidate extracted from a first-hop chunk."""

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
    """An evidence chain from query concept -> bridge -> answer evidence."""

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


# ----------------------------- Internal helpers -----------------------------
def _doc_name(db, chunk_id):
    return db.chunk_nodes[chunk_id].doc_node.doc_name


def _collect_query_context(db, resolved_matches):
    """Collect the query context needed for multi-hop from resolved matches.

    Returns:
      query_sems        : deduplicated list of query sem nodes
      query_sem_ids     : set of sem_node_id values for query sem nodes
                          (used to exclude bridge candidates)
      query_token_norms : normalized set of query tokens / query-sem tokens
                          (used to exclude bridge candidates)
    """
    low_level_sems, high_level_sems = db._collect_query_cooccurrence_sems(
        resolved_matches, False
    )
    query_sems = []
    query_sem_ids = set()
    query_token_norms = set()

    for sem_info in low_level_sems + high_level_sems:
        sem_node = sem_info.sem_node
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

    # Also add raw query tokens to the exclusion set, including words that were
    # not resolved into sem nodes.
    for match in resolved_matches:
        norm = normalize_text(match.get("token", ""))
        if norm:
            query_token_norms.add(norm)
            query_token_norms.update(norm.split())

    return query_sems, query_sem_ids, query_token_norms


def _chunk_sem_id_set(chunk_node):
    return {sem.sem_node_id for sem in getattr(chunk_node, "sem_node_list", []) or []}


def _surface_is_capitalized(sem_node, chunk_id):
    """Whether this sem node appears in the given chunk as a proper-name form
    with an initial capital."""
    for occ in getattr(sem_node, "span_occurrences", []) or []:
        cn = getattr(occ, "chunk_node", None)
        if cn is None or cn.chunk_node_id != chunk_id:
            continue
        surface = getattr(occ, "surface_text", None) or getattr(occ, "span_text", None)
        if surface and surface[:1].isupper():
            return True
    return False


# ----------------------------- Bridge extraction -----------------------------
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
    df_hub_ratio=DEFAULT_DF_HUB_RATIO,
    min_single_token_idf=DEFAULT_MIN_SINGLE_TOKEN_IDF,
    require_same_sentence=False,
    bridge_top_k=DEFAULT_BRIDGE_TOP_K,
):
    """Extract and score bridge candidates from first-hop chunk sem_node_list
    entries (plan §4.4 / §10.1)."""
    n_chunks = max(len(db.chunk_nodes), 1)
    df_hub_threshold = df_hub_ratio * n_chunks

    # Group query sems by hop1 chunk for LocalBridgeEvidence / same-sentence checks.
    candidates = {}  # token_norm -> best BridgeCandidate (keep only the strongest candidate per bridge term)

    for hop1_chunk_id in first_hop_chunk_ids:
        chunk_node = db.chunk_nodes[hop1_chunk_id]
        chunk_sem_ids = _chunk_sem_id_set(chunk_node)
        # Query sems present in this chunk, used for local evidence.
        query_sems_in_chunk = [s for s in query_sems if s.sem_node_id in chunk_sem_ids]

        for sem_node in getattr(chunk_node, "sem_node_list", []) or []:
            token_node = sem_node.token_node
            token_norm = normalize_text(token_node.token_text)
            if not token_norm:
                continue
            # Exclude nodes / tokens explicitly matched by the query.
            if sem_node.sem_node_id in query_sem_ids or token_norm in query_token_norms:
                continue

            idf = float(getattr(token_node, "idf", 0.0) or 0.0)
            df = int(getattr(token_node, "df", 0) or 0)
            is_phrase = token_node.node_type == "phrase"
            is_force_single = bool(getattr(token_node, "force_single_semantic", False))
            has_desc = bool(getattr(sem_node, "description", None))
            is_capitalized = _surface_is_capitalized(sem_node, hop1_chunk_id)
            title_match = token_norm in title_norm_set

            # ---- Generic-term / hub filtering ----
            # Single tokens are noisier, so they are gated by an IDF floor
            # (min_single_token_idf). Phrases are intentionally exempt from any
            # IDF floor: a compositional/atomic phrase is inherently distinctive
            # enough to act as a bridge regardless of its head token's IDF.
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
            # If the same bridge term appears across multiple hop1 chunks, keep
            # the strongest candidate by prior*local.
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


# ----------------------------- Second-hop retrieval -----------------------------
def retrieve_second_hop_for_bridge(bridge_sem_node, second_hop_k):
    """Take the top second_hop_k chunks by bridge sem-node BM25.

    Returns [(chunk_id, normalized_bm25), ...], with BM25 normalized to (0,1].
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
    """IDF-weighted soft coverage of residual query constraints by the
    second-hop chunk (plan §4.6 ConstraintCoverage)."""
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


# ----------------------------- Main chain-scoring flow -----------------------------
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
    """Generate and rank evidence chains for (hop1, bridge, hop2) triples."""
    chains = []
    seen_keys = set()  # deduplicate by (hop1_doc, bridge_token_norm, hop2_doc)

    for cand in bridge_candidates:
        hop1_chunk_id = cand.hop1_chunk_id
        hop1_doc = _doc_name(db, hop1_chunk_id)
        first_hop_score = first_hop_score_map.get(hop1_chunk_id, 0.0)
        # Residual constraints are the query sems not covered by the hop1
        # chunk, which naturally become the answer focus.
        hop1_sem_ids = _chunk_sem_id_set(db.chunk_nodes[hop1_chunk_id])
        residual_query_sems = [s for s in query_sems if s.sem_node_id not in hop1_sem_ids]

        second_hop = retrieve_second_hop_for_bridge(cand.sem_node, second_hop_k)
        # Inject the bridge entity's own title-matching page as a second-hop
        # candidate, a common HotpotQA second-hop gold pattern, even if it is
        # not in the bridge BM25 top-second_hop_k.
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
            # The second-hop target is exactly the bridge entity's own page ->
            # strong upweighting.
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
    """Rank the final chunk list by combining first-hop evidence with both ends
    of each chain.

    combined(chunk) = first_hop_score(chunk)               # single-hop anchor / both comparison sides
                    + hop2_bonus_weight * chain.score      # this chunk is second-hop evidence for some chain
                    + hop1_bonus_weight * chain.score      # this chunk is first-hop evidence for some chain

    The second-hop bonus, which is new bridge evidence, gets higher weight.
    The first-hop bonus, which is already present in the first-hop list, gets
    lower weight. This avoids weak chains such as comparison noise from
    disrupting the internal first-hop ranking and hurting the MRR of the first
    gold document, while still preserving the ability to move a lower-ranked
    second gold document into the top-k. Only chains with score >= the
    threshold contribute bonus weight.
    """
    combined = {}
    # Base first-hop score.
    for cid in first_hop_chunk_ids:
        combined[cid] = first_hop_score_map.get(cid, 0.0)
    # Chain bonus: hop2 gets higher weight and hop1 gets lower weight. Keep the
    # highest bonus per chunk.
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
    # Stable sort: combined score descending, then first-hop score descending,
    # then chunk_id ascending.
    ranked = sorted(
        combined.items(),
        key=lambda kv: (-kv[1], -first_hop_score_map.get(kv[0], 0.0), kv[0]),
    )
    return [cid for cid, _ in ranked]


# ----------------------------- Public entry point -----------------------------
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
    """Two-stage bridge retrieval (Plan 1).

    Returns (chains, retrieved_chunk_ids, debug_info):
      chains             : top EvidenceChain list
      retrieved_chunk_ids: deduplicated chunk-id list built by interleaving
                           first-hop and bridge second-hop evidence
                           (used by downstream tasks such as HotpotQA for
                           chunk->title evaluation)
      debug_info         : structured debug information explaining each chain
                           and the first-hop / bridge decisions
    """
    # ---- First hop: reuse chunk_cooccur_query ----
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

    # Normalized final_score for first-hop chunks.
    all_scored = cooccur_debug.get("all_scored_chunks", []) or []
    max_final = max((r["final_score"] for r in all_scored), default=0.0)
    if max_final > 0:
        first_hop_score_map = {
            r["chunk_id"]: r["final_score"] / max_final for r in all_scored
        }
    else:
        first_hop_score_map = {r["chunk_id"]: 0.0 for r in all_scored}

    # ---- Query context ----
    query_sems, query_sem_ids, query_token_norms = _collect_query_context(
        db, resolved_matches
    )

    # Corpus-level max IDF and title set. Compute once and cache on the DB if needed.
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
                # Use the document's first chunk as the representative title
                # page. In this corpus, most cases are 1 doc = 1 chunk.
                title_to_chunk[norm] = chunk_list[0].chunk_node_id
        db._multihop_title_norms = title_norm_set
        db._multihop_title_to_chunk = title_to_chunk

    share_sentence_cache = {}

    # ---- Bridge extraction ----
    bridge_candidates = extract_bridge_candidates_from_chunks(
        db,
        first_hop_chunk_ids,
        query_sems,
        query_sem_ids,
        query_token_norms,
        title_norm_set,
        max_corpus_idf,
        share_sentence_cache,
        df_hub_ratio=df_hub_ratio,
        min_single_token_idf=min_single_token_idf,
        require_same_sentence=require_same_sentence,
        bridge_top_k=bridge_top_k,
    )

    # ---- Chain generation and ranking ----
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

    # ---- Aggregate chunks: sort by combined score ----
    # base = normalized first-hop score, preserving single-hop anchor evidence
    #        or both sides of comparison evidence;
    # bonus = the highest chain score where this chunk serves as hop1 or hop2,
    #         which lifts both sides of bridge evidence upward.
    # Ranking by base + bonus helps both bridge gold documents and both sides
    # of comparison evidence enter the top-k as much as possible.
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
