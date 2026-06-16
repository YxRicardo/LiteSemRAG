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

import math
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

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
    # Milestone 4: sentence-grounded evidence (filled by
    # refine_chains_with_sentence_evidence). sentence_evidence_tier is one of
    # "same-sentence" / "adjacent-sentence" / "same-chunk"; the two sentence
    # strings make the chain human-readable instead of opaque chunk ids.
    sentence_evidence_tier: str = ""
    hop1_evidence_sentence: str = ""
    hop2_evidence_sentence: str = ""

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
            "sentence_evidence_tier": self.sentence_evidence_tier,
            "hop1_evidence_sentence": self.hop1_evidence_sentence,
            "hop2_evidence_sentence": self.hop2_evidence_sentence,
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
    use_sentence_evidence=True,
    df_hub_ratio=DEFAULT_DF_HUB_RATIO,
    min_single_token_idf=DEFAULT_MIN_SINGLE_TOKEN_IDF,
    hop1_bonus_weight=AGG_HOP1_BONUS_WEIGHT,
    hop2_bonus_weight=AGG_HOP2_BONUS_WEIGHT,
    min_chain_bonus_score=AGG_MIN_CHAIN_BONUS_SCORE,
    idf_prune_tau=None,
    idf_prune_min_max_idf=None,
    idf_prune_keep_top=2,
    resolved_matches=None,
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
        resolved_matches=resolved_matches,
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

    # Milestone 4: re-ground local evidence at the sentence level (suppress
    # pseudo bridges) and attach readable hop1/hop2 sentences.
    if use_sentence_evidence:
        refine_chains_with_sentence_evidence(db, chains, query_sem_ids)

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


# ============================================================================
# Milestone 2: global semantic co-occurrence index + path-based beam search
# ============================================================================
#
# Implementation Plan 2 (plan §5). Plan 1's bridge extraction only looks at the
# sem_node_list of first-hop chunks, so it can only bridge through entities that
# literally co-occur with the query anchors in a recalled chunk. Milestone 2
# promotes that on-demand, chunk-local co-occurrence into a persistent global
# adjacency over semantic nodes:
#
#     sem_cooccur_index[sem_id] = {
#         neighbor_sem_id: {
#             "weight": float,             # shared / sqrt(df_a * df_b) * same-sentence boost
#             "shared_chunk_ids": [int],   # capped sample of co-occurring chunks
#             "same_sentence_count": int,
#             "chunk_count": int,          # total shared chunks
#         }
#     }
#
# The index is built once (on demand) and cached on the DB as a runtime-only
# attribute (stripped on pickling per plan §5.7 — not persisted, rebuilt on
# load). multihop_path_query() then runs a 2-hop beam search from the query
# semantic nodes over this graph, which lets a bridge be discovered globally
# (anywhere it co-occurs with a query concept) and lets a bridge supported by
# MULTIPLE query constraints get a higher prior (plan §5.5 — the
# "University of Kansas located by Lawrence + Kansas City" pattern).

# ----------------------------- Index defaults -----------------------------
# Per-chunk cap on sem nodes participating in pair construction (avoids O(n^2)
# blowup on long chunks); the top nodes by entity/phrase + IDF are kept.
DEFAULT_MAX_NODES_PER_CHUNK = 80
# Each node keeps only its top-weight neighbors.
DEFAULT_TOP_NEIGHBORS = 30
# Edges below this weight are dropped entirely.
DEFAULT_MIN_EDGE_WEIGHT = 0.0
# Generic single tokens below this IDF never enter the index (phrases/entities
# are exempt — they are inherently distinctive enough to bridge).
DEFAULT_INDEX_MIN_TOKEN_IDF = 2.0
# Same-sentence boost on edge weight: weight *= 1 + alpha * same_sent/shared.
DEFAULT_SAME_SENTENCE_ALPHA = 0.5
# Cap on stored shared_chunk_ids per edge (keep memory bounded; only used to
# pick a representative hop1 chunk at query time).
EDGE_SHARED_CHUNK_CAP = 32

# Per extra query constraint that supports the same bridge, multiply the bridge
# prior by (1 + this). Implements plan §5.5 constraint_supported_bridge.
CONSTRAINT_SUPPORT_BONUS = 0.30
# How strongly the normalized path edge weight folds into the bridge prior.
PATH_EDGE_WEIGHT_GAIN = 0.50


def _occurrence_sentence_map(db):
    """One pass over all sem-node occurrences -> {(chunk_id, sem_id): {sentence_id}}.

    Built once per index build so per-pair same-sentence checks during pair
    accumulation are O(1) set intersections instead of rescanning occurrences.
    """
    occ_sent = defaultdict(set)
    for sem in db.sem_nodes:
        sid = sem.sem_node_id
        for occ in getattr(sem, "span_occurrences", []) or []:
            cn = getattr(occ, "chunk_node", None)
            if cn is None:
                continue
            sentence_id = getattr(occ, "sentence_id", None)
            if sentence_id is not None:
                occ_sent[(cn.chunk_node_id, sid)].add(sentence_id)
    return occ_sent


def build_sem_cooccur_index(
    db,
    *,
    max_nodes_per_chunk=DEFAULT_MAX_NODES_PER_CHUNK,
    top_neighbors=DEFAULT_TOP_NEIGHBORS,
    min_edge_weight=DEFAULT_MIN_EDGE_WEIGHT,
    df_hub_ratio=DEFAULT_DF_HUB_RATIO,
    min_token_idf=DEFAULT_INDEX_MIN_TOKEN_IDF,
    same_sentence_alpha=DEFAULT_SAME_SENTENCE_ALPHA,
    force=False,
):
    """Build (and cache on the DB) the global semantic co-occurrence adjacency.

    Returns db.sem_cooccur_index. Idempotent: returns the cached index unless
    force=True. Runtime-only; stripped on pickling and rebuilt on demand.
    """
    cached = getattr(db, "sem_cooccur_index", None)
    if cached is not None and not force:
        return cached

    n_chunks = max(len(db.chunk_nodes), 1)
    df_hub_threshold = df_hub_ratio * n_chunks
    occ_sent = _occurrence_sentence_map(db)

    def node_keep(sem):
        tn = sem.token_node
        df = int(getattr(tn, "df", 0) or 0)
        if df > df_hub_threshold:
            return False
        is_phrase = tn.node_type == "phrase"
        is_entity = bool(getattr(tn, "force_single_semantic", False))
        if is_phrase or is_entity:
            return True
        idf = float(getattr(tn, "idf", 0.0) or 0.0)
        return idf >= min_token_idf

    def node_rank_key(sem):
        tn = sem.token_node
        distinctive = tn.node_type == "phrase" or bool(
            getattr(tn, "force_single_semantic", False)
        )
        return (distinctive, float(getattr(tn, "idf", 0.0) or 0.0))

    pair_shared = defaultdict(int)
    pair_same_sent = defaultdict(int)
    pair_chunks = defaultdict(list)

    for chunk in db.chunk_nodes:
        cid = chunk.chunk_node_id
        # Deduplicate by sem id within the chunk, then keep distinctive nodes.
        seen = set()
        sems = []
        for sem in getattr(chunk, "sem_node_list", []) or []:
            if sem.sem_node_id in seen or not node_keep(sem):
                continue
            seen.add(sem.sem_node_id)
            sems.append(sem)
        if len(sems) < 2:
            continue
        if len(sems) > max_nodes_per_chunk:
            sems.sort(key=node_rank_key, reverse=True)
            sems = sems[:max_nodes_per_chunk]

        for a, b in combinations(sems, 2):
            ia, ib = a.sem_node_id, b.sem_node_id
            key = (ia, ib) if ia < ib else (ib, ia)
            pair_shared[key] += 1
            sset_a = occ_sent.get((cid, ia))
            sset_b = occ_sent.get((cid, ib))
            if sset_a and sset_b and (sset_a & sset_b):
                pair_same_sent[key] += 1
            bucket = pair_chunks[key]
            if len(bucket) < EDGE_SHARED_CHUNK_CAP:
                bucket.append(cid)

    index = defaultdict(dict)
    for (a, b), shared in pair_shared.items():
        df_a = int(getattr(db.sem_nodes[a].token_node, "df", 0) or 0) or 1
        df_b = int(getattr(db.sem_nodes[b].token_node, "df", 0) or 0) or 1
        weight = shared / math.sqrt(df_a * df_b)
        ss = pair_same_sent.get((a, b), 0)
        if ss > 0:
            weight *= 1.0 + same_sentence_alpha * ss / shared
        if weight < min_edge_weight:
            continue
        edge = {
            "weight": weight,
            "shared_chunk_ids": pair_chunks[(a, b)],
            "same_sentence_count": ss,
            "chunk_count": shared,
        }
        index[a][b] = edge
        index[b][a] = edge  # symmetric; the dict is read-only at query time

    pruned = {}
    for node, neighbors in index.items():
        if len(neighbors) > top_neighbors:
            top = sorted(
                neighbors.items(), key=lambda kv: kv[1]["weight"], reverse=True
            )[:top_neighbors]
            pruned[node] = dict(top)
        else:
            pruned[node] = dict(neighbors)

    db.sem_cooccur_index = pruned
    db._sem_cooccur_index_params = {
        "max_nodes_per_chunk": max_nodes_per_chunk,
        "top_neighbors": top_neighbors,
        "min_edge_weight": min_edge_weight,
        "df_hub_ratio": df_hub_ratio,
        "min_token_idf": min_token_idf,
        "same_sentence_alpha": same_sentence_alpha,
        "num_nodes": len(pruned),
        "num_edges": sum(len(v) for v in pruned.values()) // 2,
    }
    return pruned


def get_sem_neighbors(db, sem_node_id):
    """Top neighbors of a semantic node in the global co-occurrence index.

    Lazily builds the index on first use. Returns {neighbor_id: edge_dict}.
    """
    index = build_sem_cooccur_index(db)
    return index.get(sem_node_id, {})


def _best_shared_chunk(edge, first_hop_score_map):
    """Pick the shared chunk of an edge with the highest first-hop score (so the
    hop1 evidence of a path bridge is an actually-recalled anchor chunk)."""
    shared = edge.get("shared_chunk_ids") or []
    if not shared:
        return None, 0.0
    best_cid = shared[0]
    best_score = first_hop_score_map.get(best_cid, 0.0)
    for cid in shared[1:]:
        score = first_hop_score_map.get(cid, 0.0)
        if score > best_score:
            best_cid, best_score = cid, score
    return best_cid, best_score


def extract_bridge_candidates_from_path_index(
    db,
    query_sems,
    query_sem_ids,
    query_token_norms,
    title_norm_set,
    max_corpus_idf,
    first_hop_score_map,
    *,
    df_hub_ratio=DEFAULT_DF_HUB_RATIO,
    min_single_token_idf=DEFAULT_MIN_SINGLE_TOKEN_IDF,
    bridge_top_k=DEFAULT_BRIDGE_TOP_K,
):
    """Discover bridge candidates by walking the global co-occurrence index out
    from the query semantic nodes (plan §5.4 first beam expansion).

    Unlike the Plan-1 chunk-local extractor, a bridge here can be reached from
    several query constraints; the count of distinct supporting query sems lifts
    its prior (plan §5.5 constraint_supported_bridge).
    """
    n_chunks = max(len(db.chunk_nodes), 1)
    df_hub_threshold = df_hub_ratio * n_chunks

    # neighbor_id -> aggregated path evidence across all query starting nodes.
    agg = {}
    max_edge_weight = 0.0
    for q_sem in query_sems:
        neighbors = get_sem_neighbors(db, q_sem.sem_node_id)
        q_idf = float(getattr(q_sem.token_node, "idf", 0.0) or 0.0)
        for nb_id, edge in neighbors.items():
            if nb_id in query_sem_ids:
                continue
            w = edge["weight"]
            max_edge_weight = max(max_edge_weight, w)
            entry = agg.get(nb_id)
            if entry is None:
                entry = {
                    "best_edge": edge,
                    "best_weight": w,
                    "support_idf": 0.0,
                    "support_count": 0,
                    "same_sentence": edge.get("same_sentence_count", 0) > 0,
                }
                agg[nb_id] = entry
            if w > entry["best_weight"]:
                entry["best_weight"] = w
                entry["best_edge"] = edge
            entry["support_idf"] += q_idf + 1e-6
            entry["support_count"] += 1
            if edge.get("same_sentence_count", 0) > 0:
                entry["same_sentence"] = True

    candidates = []
    for nb_id, entry in agg.items():
        sem_node = db.sem_nodes[nb_id]
        token_node = sem_node.token_node
        token_norm = normalize_text(token_node.token_text)
        if not token_norm or token_norm in query_token_norms:
            continue

        idf = float(getattr(token_node, "idf", 0.0) or 0.0)
        df = int(getattr(token_node, "df", 0) or 0)
        is_phrase = token_node.node_type == "phrase"
        is_force_single = bool(getattr(token_node, "force_single_semantic", False))
        has_desc = bool(getattr(sem_node, "description", None))
        title_match = token_norm in title_norm_set

        if not is_phrase and idf < min_single_token_idf and not title_match:
            continue
        if df > df_hub_threshold and not title_match:
            continue

        edge = entry["best_edge"]
        hop1_chunk_id, _ = _best_shared_chunk(edge, first_hop_score_map)
        if hop1_chunk_id is None:
            continue
        is_capitalized = _surface_is_capitalized(sem_node, hop1_chunk_id)
        local_evidence = (
            LOCAL_EVIDENCE_SENTENCE if entry["same_sentence"] else LOCAL_EVIDENCE_CHUNK
        )

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
        # Path-specific signals: edge strength + multi-constraint support.
        norm_edge = entry["best_weight"] / max_edge_weight if max_edge_weight > 0 else 0.0
        coef *= 1.0 + PATH_EDGE_WEIGHT_GAIN * norm_edge
        coef *= 1.0 + CONSTRAINT_SUPPORT_BONUS * (entry["support_count"] - 1)
        bridge_prior = prior * coef

        candidates.append(
            BridgeCandidate(
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
        )

    candidates.sort(
        key=lambda c: c.bridge_prior * c.local_evidence, reverse=True
    )
    return candidates[:bridge_top_k]


def _prepare_query_and_caches(db, cooccur_debug, first_hop_chunk_ids):
    """Shared post-first-hop setup: query context, corpus IDF / title caches,
    and the normalized first-hop score map. Reused by path / comparison paths."""
    resolved_matches = cooccur_debug["resolved_matches"]
    all_scored = cooccur_debug.get("all_scored_chunks", []) or []
    max_final = max((r["final_score"] for r in all_scored), default=0.0)
    if max_final > 0:
        first_hop_score_map = {
            r["chunk_id"]: r["final_score"] / max_final for r in all_scored
        }
    else:
        first_hop_score_map = {r["chunk_id"]: 0.0 for r in all_scored}

    query_sems, query_sem_ids, query_token_norms = _collect_query_context(
        db, resolved_matches
    )

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
                title_to_chunk[norm] = chunk_list[0].chunk_node_id
        db._multihop_title_norms = title_norm_set
        db._multihop_title_to_chunk = title_to_chunk

    return (
        resolved_matches,
        first_hop_score_map,
        query_sems,
        query_sem_ids,
        query_token_norms,
        max_corpus_idf,
        title_norm_set,
        title_to_chunk,
    )


def multihop_path_query(
    db,
    query_text,
    top_k_chain=DEFAULT_TOP_K_CHAIN,
    first_hop_k=DEFAULT_FIRST_HOP_K,
    bridge_top_k=DEFAULT_BRIDGE_TOP_K,
    second_hop_k=DEFAULT_SECOND_HOP_K,
    *,
    search_mode="broad",
    allow_same_chunk=False,
    use_sentence_evidence=True,
    df_hub_ratio=DEFAULT_DF_HUB_RATIO,
    min_single_token_idf=DEFAULT_MIN_SINGLE_TOKEN_IDF,
    hop1_bonus_weight=AGG_HOP1_BONUS_WEIGHT,
    hop2_bonus_weight=AGG_HOP2_BONUS_WEIGHT,
    min_chain_bonus_score=AGG_MIN_CHAIN_BONUS_SCORE,
    idf_prune_tau=None,
    idf_prune_min_max_idf=None,
    idf_prune_keep_top=2,
    resolved_matches=None,
    print_important_tokens=False,
):
    """Path-based bridge retrieval over the global co-occurrence index (Plan 2).

    Same interface and ChainScore as multihop_bridge_query(), but bridge
    candidates come from global graph adjacency (get_sem_neighbors) instead of
    the first-hop chunk sem_node_list, so bridges can be discovered anywhere
    they co-occur with a query concept and a multi-constraint-supported bridge
    is preferred. Returns (chains, retrieved_chunk_ids, debug_info).
    """
    build_sem_cooccur_index(db)

    _, first_hop_chunk_ids, cooccur_debug = db.chunk_cooccur_query(
        query_text,
        top_k_chunk=first_hop_k,
        print_important_tokens=False,
        search_mode=search_mode,
        idf_prune_tau=idf_prune_tau,
        idf_prune_min_max_idf=idf_prune_min_max_idf,
        idf_prune_keep_top=idf_prune_keep_top,
        resolved_matches=resolved_matches,
    )
    (
        _resolved,
        first_hop_score_map,
        query_sems,
        query_sem_ids,
        query_token_norms,
        max_corpus_idf,
        title_norm_set,
        title_to_chunk,
    ) = _prepare_query_and_caches(db, cooccur_debug, first_hop_chunk_ids)

    bridge_candidates = extract_bridge_candidates_from_path_index(
        db,
        query_sems,
        query_sem_ids,
        query_token_norms,
        title_norm_set,
        max_corpus_idf,
        first_hop_score_map,
        df_hub_ratio=df_hub_ratio,
        min_single_token_idf=min_single_token_idf,
        bridge_top_k=bridge_top_k,
    )

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
    for chain in chains:
        chain.strategy = "path"

    # Milestone 4: re-ground local evidence at the sentence level (suppress
    # pseudo bridges) and attach readable hop1/hop2 sentences before aggregating.
    if use_sentence_evidence:
        refine_chains_with_sentence_evidence(db, chains, query_sem_ids)

    retrieved_chunk_ids = aggregate_chunk_ranking(
        first_hop_chunk_ids,
        first_hop_score_map,
        chains,
        hop1_bonus_weight=hop1_bonus_weight,
        hop2_bonus_weight=hop2_bonus_weight,
        min_chain_bonus_score=min_chain_bonus_score,
    )

    if print_important_tokens:
        print(f"[path] query sems: {[s.token_node.token_text for s in query_sems]}")
        print(f"[path] bridges: {[c.token_norm for c in bridge_candidates]}")
        for c in chains:
            print(f"  chain score={c.score:.5f} :: {c.why_selected}")

    debug_info = {
        "strategy": "path",
        "params": {
            "top_k_chain": top_k_chain,
            "first_hop_k": first_hop_k,
            "bridge_top_k": bridge_top_k,
            "second_hop_k": second_hop_k,
            "allow_same_chunk": allow_same_chunk,
        },
        "index_params": getattr(db, "_sem_cooccur_index_params", None),
        "sentence_evidence_params": (
            getattr(db, "_sentence_evidence_params", None)
            if use_sentence_evidence else None
        ),
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


# ============================================================================
# Milestone 3: question decomposition + strategy routing
# ============================================================================
#
# Implementation Plan 3 (plan §6). A lightweight, LLM-free planner classifies
# the question and routes it to the right retrieval strategy instead of forcing
# every question through bridge expansion:
#
#   - comparison  -> comparison_focused_query()  (two parallel sides, NO bridge)
#   - bridge      -> multihop_path_query()        (global path bridge retrieval)
#   - single      -> chunk_cooccur_query()        (plain single-hop)
#
# The comparison route is the key fix for the Milestone-1 MRR regression: bridge
# expansion injected weak chains on comparison questions and slightly pushed the
# first gold down. Comparison questions instead get side-aware interleaving so
# BOTH side gold documents land near the top (plan §6.3).

# Cue words that signal a comparison question.
_COMPARISON_CUES = (
    "same",
    "both",
    "older",
    "younger",
    "newer",
    "earlier",
    "later",
    "first",
    "more",
    "less",
    "larger",
    "smaller",
    "bigger",
    "longer",
    "shorter",
    "higher",
    "lower",
    "greater",
    "taller",
    "closer",
    "faster",
    "slower",
    "either",
    "neither",
    "difference",
    "different",
)
_COMPARISON_LEAD = re.compile(
    r"^\s*(are|were|is|was|do|does|did|which|who|whose|whom)\b", re.IGNORECASE
)


@dataclass
class QuestionPlan:
    """Lightweight rule-based decomposition / routing decision (plan §6.2)."""

    mode: str  # "comparison" | "bridge" | "single"
    anchor_units: list = field(default_factory=list)   # side / anchor surfaces
    answer_focus: str = ""                              # wh / property focus
    cues: list = field(default_factory=list)            # matched comparison cues
    reason: str = ""

    def as_debug(self) -> dict:
        return {
            "mode": self.mode,
            "anchor_units": list(self.anchor_units),
            "answer_focus": self.answer_focus,
            "cues": list(self.cues),
            "reason": self.reason,
        }


def _anchor_sems_from_matches(db, resolved_matches):
    """Entity / proper-name / high-IDF phrase sem nodes from the query, used as
    comparison sides or bridge anchors. Deduplicated, ranked by IDF desc."""
    anchors = []
    seen = set()
    for match in resolved_matches:
        sem_node = match.get("exact_sem_node")
        if sem_node is None:
            continue
        token_node = sem_node.token_node
        token_norm = normalize_text(token_node.token_text)
        if not token_norm or token_norm in seen:
            continue
        token_type = match.get("token_type")
        is_phrase = token_node.node_type == "phrase"
        is_entity = bool(getattr(token_node, "force_single_semantic", False))
        surface = match.get("query_surface") or ""
        capitalized = bool(surface[:1].isupper())
        if not (token_type == "ent" or is_entity or is_phrase or capitalized):
            continue
        seen.add(token_norm)
        anchors.append(sem_node)
    anchors.sort(
        key=lambda s: float(getattr(s.token_node, "idf", 0.0) or 0.0), reverse=True
    )
    return anchors


def plan_query(db, query_text, resolved_matches=None):
    """Classify a question into comparison / bridge / single (plan §6.2).

    Rule-based and LLM-free: a comparison needs a comparison cue (same/both/
    older/... or an `A or B` choice) together with >= 2 distinct anchor entities;
    a question with a single strong anchor and a wh answer focus is bridge;
    everything else falls back to single-hop.

    resolved_matches may be supplied by the router (multihop_query) so the query
    is encoded once and shared with the chosen strategy; if None it is resolved
    here (standalone use re-encodes, as before).
    """
    if resolved_matches is None:
        _qt, _tokens, resolved_matches = db._resolve_query_matches(
            query_text, expand_compositional=True
        )
    anchors = _anchor_sems_from_matches(db, resolved_matches)
    anchor_norms = [normalize_text(a.token_node.token_text) for a in anchors]

    lowered = f" {query_text.lower()} "
    cues = [c for c in _COMPARISON_CUES if f" {c} " in lowered]
    # "A or B" choice pattern between two anchors also counts as a comparison cue.
    if " or " in lowered and len(anchors) >= 2:
        cues = cues or ["or"]

    wh = ""
    m = re.search(r"\b(what|which|who|whose|where|when|how|why)\b", query_text.lower())
    if m:
        wh = m.group(1)

    if cues and len(anchors) >= 2 and _COMPARISON_LEAD.search(query_text):
        return QuestionPlan(
            mode="comparison",
            anchor_units=anchor_norms[:4],
            answer_focus=wh,
            cues=cues,
            reason="comparison cue + >=2 anchors + comparative lead",
        )
    if anchors:
        return QuestionPlan(
            mode="bridge",
            anchor_units=anchor_norms[:4],
            answer_focus=wh,
            cues=cues,
            reason="anchor entity present, no comparison signal",
        )
    return QuestionPlan(
        mode="single",
        anchor_units=anchor_norms[:4],
        answer_focus=wh,
        cues=cues,
        reason="no strong anchor entity",
    )


def comparison_focused_query(
    db,
    query_text,
    top_k_chain=DEFAULT_TOP_K_CHAIN,
    first_hop_k=DEFAULT_FIRST_HOP_K,
    *,
    search_mode="broad",
    plan=None,
    resolved_matches=None,
):
    """Comparison strategy (plan §6.3): retrieve each side independently and
    interleave so BOTH side documents reach the top, instead of bridging through
    a co-occurring predicate (which would drift to e.g. director/producer).

    Returns (None, retrieved_chunk_ids, debug_info) — no evidence chain is built
    for comparison; the two sides are parallel evidence, not a bridge path.
    """
    _, first_hop_chunk_ids, cooccur_debug = db.chunk_cooccur_query(
        query_text,
        top_k_chunk=max(first_hop_k, top_k_chain),
        print_important_tokens=False,
        search_mode=search_mode,
        resolved_matches=resolved_matches,
    )
    (
        resolved_matches,
        _first_hop_score_map,
        _query_sems,
        _query_sem_ids,
        _query_token_norms,
        _max_corpus_idf,
        _title_norm_set,
        title_to_chunk,
    ) = _prepare_query_and_caches(db, cooccur_debug, first_hop_chunk_ids)
    anchors = _anchor_sems_from_matches(db, resolved_matches)

    # Per-anchor top chunk lists. The side's own wiki page (title == anchor) is
    # the canonical evidence for a comparison attribute, so it is forced to the
    # front; the rest follow the anchor's own BM25 (its name is most frequent on
    # its own page, but a same-named distractor — e.g. "Ed Wood (film)" vs the
    # person "Ed Wood" — can outscore it, so the title page must win explicitly).
    per_anchor_ranked = []
    for sem_node in anchors:
        token_norm = normalize_text(sem_node.token_node.token_text)
        bm25 = getattr(sem_node, "BM25", None) or {}
        ranked = [cid for cid, _ in sorted(bm25.items(), key=lambda kv: kv[1], reverse=True)]
        title_chunk = title_to_chunk.get(token_norm)
        if title_chunk is not None:
            ranked = [title_chunk] + [c for c in ranked if c != title_chunk]
        if ranked:
            per_anchor_ranked.append((sem_node, ranked))

    # Round-robin interleave so each side contributes its best chunk first.
    interleaved = []
    seen = set()
    if per_anchor_ranked:
        depth = max(len(r) for _, r in per_anchor_ranked)
        for d in range(depth):
            for _sem, ranked in per_anchor_ranked:
                if d < len(ranked):
                    cid = ranked[d]
                    if cid not in seen:
                        seen.add(cid)
                        interleaved.append(cid)

    # Append the plain co-occurrence ranking to fill any remaining slots.
    for cid in first_hop_chunk_ids:
        if cid not in seen:
            seen.add(cid)
            interleaved.append(cid)

    debug_info = {
        "strategy": "comparison",
        "params": {"top_k_chain": top_k_chain, "first_hop_k": first_hop_k},
        "plan": plan.as_debug() if plan is not None else None,
        "sides": [normalize_text(s.token_node.token_text) for s in anchors],
        "side_count": len(per_anchor_ranked),
        "first_hop_chunk_ids": list(first_hop_chunk_ids),
        "cooccur_debug": cooccur_debug,
    }
    return None, interleaved, debug_info


def multihop_query(
    db,
    query_text,
    top_k_chain=DEFAULT_TOP_K_CHAIN,
    first_hop_k=DEFAULT_FIRST_HOP_K,
    bridge_top_k=DEFAULT_BRIDGE_TOP_K,
    second_hop_k=DEFAULT_SECOND_HOP_K,
    *,
    use_path_index=True,
    search_mode="broad",
    print_important_tokens=False,
    **kwargs,
):
    """Unified multi-hop entry with strategy routing (plan §6 / Milestone 3).

    plan_query() picks the strategy; the chosen retriever runs; debug carries the
    plan and the strategy name. Returns (chains, retrieved_chunk_ids, debug_info)
    where chains is None for the comparison route (parallel side evidence).
    """
    # Resolve (encode) the query exactly once and share it across plan_query and
    # the chosen strategy, instead of plan_query and the strategy each calling
    # _resolve_query_matches (two GPU encodes). resolved_matches carries the query
    # embeddings/sem nodes, so every downstream chunk_cooccur_query reuses them and
    # only its CPU scoring runs (M2/M3 report §6 optimisation).
    _qt, _tokens, resolved_matches = db._resolve_query_matches(
        query_text, search_mode=search_mode, expand_compositional=True
    )
    plan = plan_query(db, query_text, resolved_matches=resolved_matches)

    if plan.mode == "comparison":
        chains, chunk_ids, debug_info = comparison_focused_query(
            db,
            query_text,
            top_k_chain=top_k_chain,
            first_hop_k=first_hop_k,
            search_mode=search_mode,
            plan=plan,
            resolved_matches=resolved_matches,
        )
    elif plan.mode == "bridge":
        bridge_fn = multihop_path_query if use_path_index else multihop_bridge_query
        chains, chunk_ids, debug_info = bridge_fn(
            db,
            query_text,
            top_k_chain=top_k_chain,
            first_hop_k=first_hop_k,
            bridge_top_k=bridge_top_k,
            second_hop_k=second_hop_k,
            search_mode=search_mode,
            resolved_matches=resolved_matches,
            print_important_tokens=print_important_tokens,
            **kwargs,
        )
    else:  # single
        _, chunk_ids, cooccur_debug = db.chunk_cooccur_query(
            query_text,
            top_k_chunk=max(first_hop_k, top_k_chain),
            print_important_tokens=False,
            search_mode=search_mode,
            resolved_matches=resolved_matches,
        )
        chains = []
        debug_info = {"strategy": "single", "cooccur_debug": cooccur_debug}

    debug_info["plan"] = plan.as_debug()
    debug_info["strategy"] = plan.mode
    if print_important_tokens:
        print(f"[multihop_query] plan={plan.mode} ({plan.reason})")
    return chains, chunk_ids, debug_info


# ============================================================================
# Milestone 4: sentence-level evidence graph
# ============================================================================
#
# Implementation Plan 4 (plan §7). Bridge / path retrieval reasons at chunk
# granularity, so two concepts that merely sit in the same (possibly long) chunk
# look co-occurrent even when they are sentences apart and unrelated — a "pseudo
# bridge". Schema 7 already persists ChunkNode.sentence_boundaries and per-
# occurrence sentence_id, which is enough to drop the reasoning granularity to
# the sentence without re-chunking.
#
# Two runtime-only structures (built on demand, stripped on pickle, rebuilt on
# load — exactly like the Milestone-2 co-occurrence index):
#
#     sentence_evidence_index[(chunk_id, sentence_id)] = {
#         "sem_node_ids": set[int],   # sem nodes occurring in this sentence
#         "doc_id": int,
#     }
#     sem_to_sentences[sem_node_id] = set[(chunk_id, sentence_id)]
#
# Sentence text is rendered on demand from chunk_text + sentence_boundaries
# (_sentence_text) rather than duplicated into the index, to keep memory bounded.
#
# Used for two things (plan §7.3):
#   1. Pseudo-bridge suppression — a chain's LocalBridgeEvidence is recomputed
#      from the actual sentence distance between the bridge and a query anchor in
#      the hop1 chunk (same sentence 1.0 / adjacent 0.8 / same chunk only 0.6),
#      replacing the coarse chunk/edge-level estimate. Chains whose bridge never
#      shares (or neighbours) an anchor sentence stay at 0.6 and are demoted.
#   2. Chain readability — each chain carries the literal hop1 / hop2 evidence
#      sentences instead of only chunk ids.

# Local-evidence tier when the bridge sits in a sentence directly adjacent to a
# query-anchor sentence in the hop1 chunk (between same-sentence and same-chunk).
LOCAL_EVIDENCE_ADJACENT = 0.8


def _resolve_sentence_id(occ):
    """Sentence index of a span occurrence: prefer the persisted sentence_id
    (schema 7), else recompute from the chunk's sentence_boundaries, else None
    (pre-7 pickle without boundaries)."""
    sid = getattr(occ, "sentence_id", None)
    if sid is not None:
        return sid
    cn = getattr(occ, "chunk_node", None)
    span_start = getattr(occ, "span_start", None)
    if cn is None or span_start is None:
        return None
    boundaries = getattr(cn, "sentence_boundaries", None)
    if not boundaries:
        return None
    return bisect_right(boundaries, span_start)


def _sentence_text(db, chunk_id, sentence_id):
    """Slice the literal sentence text out of a chunk using sentence_boundaries.

    Sentence i spans [boundaries[i-1], boundaries[i]); sentence 0 starts at 0 and
    the trailing sentence runs to end-of-chunk. Falls back to the whole chunk
    when boundaries are absent."""
    chunk = db.chunk_nodes[chunk_id]
    text = chunk.chunk_text or ""
    boundaries = getattr(chunk, "sentence_boundaries", None)
    if not boundaries or sentence_id is None:
        return text.strip()
    start = 0 if sentence_id <= 0 else boundaries[min(sentence_id - 1, len(boundaries) - 1)]
    end = boundaries[sentence_id] if sentence_id < len(boundaries) else len(text)
    return text[start:end].strip()


def build_sentence_evidence_index(db, *, force=False):
    """Build (and cache on the DB) the sentence-level evidence structures
    (plan §7.2). Returns (sentence_evidence_index, sem_to_sentences).

    Idempotent: returns the cached structures unless force=True. Runtime-only;
    stripped on pickling and rebuilt on demand. On pre-7 pickles with no
    sentence ids the structures come out empty and db._sentence_evidence_available
    is False, which makes every sentence-level feature a silent no-op (callers
    degrade to chunk-level evidence).
    """
    cached = getattr(db, "sentence_evidence_index", None)
    if cached is not None and not force:
        return cached, getattr(db, "sem_to_sentences", {}) or {}

    sentence_index = {}
    sem_to_sentences = defaultdict(set)
    any_sentence = False

    for sem in db.sem_nodes:
        sid = sem.sem_node_id
        for occ in getattr(sem, "span_occurrences", []) or []:
            cn = getattr(occ, "chunk_node", None)
            if cn is None:
                continue
            sentence_id = _resolve_sentence_id(occ)
            if sentence_id is None:
                continue
            any_sentence = True
            key = (cn.chunk_node_id, sentence_id)
            bucket = sentence_index.get(key)
            if bucket is None:
                bucket = {"sem_node_ids": set(), "doc_id": cn.doc_node.doc_node_id}
                sentence_index[key] = bucket
            bucket["sem_node_ids"].add(sid)
            sem_to_sentences[sid].add(key)

    db.sentence_evidence_index = sentence_index
    db.sem_to_sentences = dict(sem_to_sentences)
    db._sentence_evidence_available = any_sentence
    db._sentence_evidence_params = {
        "num_sentences": len(sentence_index),
        "num_sem_with_sentence": len(sem_to_sentences),
        "available": any_sentence,
    }
    return sentence_index, db.sem_to_sentences


def _bridge_sentence_evidence(db, anchor_sem_ids, bridge_sem_id, hop1_chunk_id):
    """Sentence-grounded local evidence tier for a bridge in its hop1 chunk.

    Returns (tier, bridge_sentence_id, anchor_sentence_id):
      tier 1.0 — bridge shares a sentence with a query anchor (strong);
      tier 0.8 — bridge sits in a sentence directly adjacent to an anchor;
      tier 0.6 — bridge only co-occurs at chunk level (possible pseudo bridge).
    Falls back to the chunk tier when sentence ids are unavailable.
    """
    s2s = getattr(db, "sem_to_sentences", None)
    if not s2s:
        return LOCAL_EVIDENCE_CHUNK, None, None
    bridge_sents = {s for (c, s) in s2s.get(bridge_sem_id, ()) if c == hop1_chunk_id}
    if not bridge_sents:
        return LOCAL_EVIDENCE_CHUNK, None, None
    anchor_sents = set()
    for aid in anchor_sem_ids:
        for (c, s) in s2s.get(aid, ()):
            if c == hop1_chunk_id:
                anchor_sents.add(s)
    if not anchor_sents:
        return LOCAL_EVIDENCE_CHUNK, min(bridge_sents), None
    common = bridge_sents & anchor_sents
    if common:
        sid = min(common)
        return LOCAL_EVIDENCE_SENTENCE, sid, sid
    best = None
    for bs in bridge_sents:
        for a_s in anchor_sents:
            if abs(bs - a_s) == 1 and (best is None or bs < best[0]):
                best = (bs, a_s)
    if best is not None:
        return LOCAL_EVIDENCE_ADJACENT, best[0], best[1]
    return LOCAL_EVIDENCE_CHUNK, min(bridge_sents), min(anchor_sents)


def refine_chains_with_sentence_evidence(db, chains, query_sem_ids, *, rescore=True):
    """Re-ground each chain's local evidence at the sentence level and attach
    readable hop1 / hop2 evidence sentences (plan §7.3).

    Replaces the coarse chunk/edge LocalBridgeEvidence with the precise tier from
    _bridge_sentence_evidence and rescales chain.score by (new_tier / old_tier),
    so pseudo bridges (chunk-only co-occurrence) lose ground to sentence-grounded
    ones. Re-sorts the chains. No-op (returns chains unchanged) when sentence ids
    are unavailable. Mutates chains in place.
    """
    build_sentence_evidence_index(db)
    if not getattr(db, "_sentence_evidence_available", False):
        return chains
    s2s = db.sem_to_sentences
    for ch in chains:
        bridge_id = ch.bridge_sem_node_id
        tier, _b_sid, a_sid = _bridge_sentence_evidence(
            db, query_sem_ids, bridge_id, ch.hop1_chunk_id
        )
        old_local = ch.local_bridge_evidence or LOCAL_EVIDENCE_CHUNK
        if rescore and old_local > 0:
            ch.score *= tier / old_local
        ch.local_bridge_evidence = tier
        # hop1 readable sentence: prefer the anchor-sharing sentence, else the
        # bridge's own sentence in the hop1 chunk.
        h1_sid = a_sid if a_sid is not None else _b_sid
        ch.hop1_evidence_sentence = (
            _sentence_text(db, ch.hop1_chunk_id, h1_sid) if h1_sid is not None else ""
        )
        # hop2 readable sentence: the bridge's sentence inside the hop2 chunk;
        # fall back to the hop2 chunk's lead sentence when the bridge has no
        # recorded in-sentence occurrence there (e.g. the injected title page,
        # where the bridge is the page title rather than an indexed span).
        hop2_sents = sorted(
            s for (c, s) in s2s.get(bridge_id, ()) if c == ch.hop2_chunk_id
        )
        ch.hop2_evidence_sentence = _sentence_text(
            db, ch.hop2_chunk_id, hop2_sents[0] if hop2_sents else 0
        )
        if tier >= LOCAL_EVIDENCE_SENTENCE:
            ch.sentence_evidence_tier = "same-sentence"
        elif tier >= LOCAL_EVIDENCE_ADJACENT:
            ch.sentence_evidence_tier = "adjacent-sentence"
        else:
            ch.sentence_evidence_tier = "same-chunk"
    chains.sort(key=lambda c: c.score, reverse=True)
    return chains
