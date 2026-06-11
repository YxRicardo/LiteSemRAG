import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback
from bisect import bisect_right
from html import escape
from collections import Counter, defaultdict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import queue
import threading

import spacy
import numpy as np
import torch
import torch.nn.functional as F

from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from transformers import AutoModel, AutoTokenizer

from text_processing import (
    _is_valid_token,
    _normalize_token_text,
    bm25_tf_saturation,
    clean_text,
    count_words,
    encode_chunk,
    encode_chunk_batch,
    encode_text,
    extract_important_phrases,
    extract_important_spans,
    extract_important_tokens,
    get_embed_by_offest,
    get_num_tokens,
    get_token_embeds,
    normalize_text,
    split_doc,
)
from phrase_analysis import (
    PHRASE_TYPE_ATOMIC,
    PHRASE_TYPE_COMPOSITIONAL,
    PHRASE_TYPE_SINGLE_TOKEN,
    PHRASE_TYPE_UNKNOWN,
    PhraseAnalyzer,
)
from utils import (
    average_embeds,
    build_wikidata_candidate_bank,
    extract_cross_encoder_scores,
    farthest_first_traversal,
    get_s_mean,
    load_wikidata_definition_candidates,
    plot_embeddings,
    print_size_mb,
    sem_embed_sim,
)

CURRENT_SCHEMA_VERSION = 8

# Anchor 传播分配法的默认数值超参(对齐
# jupyter_notebooks/hotpotqa_anchor_propagation_compare.ipynb)。只有少数核心项
# 通过 __init__ 暴露;其余写死成这里的 notebook 默认值,需要时可在实例上覆盖。
ANCHOR_PROP_DEFAULTS = {
    "anchor_fraction": 0.15,        # anchor 占全部 occurrence 的比例
    "anchor_fft_ratio": 0.70,       # anchor 中由 FFT 选出的比例(其余 random)
    "anchor_min_count": 2,          # anchor 最少个数
    "anchor_max_count": 20,         # anchor 个数上限(防止高频词把过多样本塞进单次 LLM 判定)
    "anchor_random_state": 42,      # anchor 随机采样种子
    "prop_knn_k": 8,                # 构图近邻数 k
    "prop_vote_ratio": 0.60,        # 普通类采纳所需近邻多数比例
    "prop_weak_vote_ratio": 0.50,   # 弱投票(需 anchor center)比例
    "prop_rare_vote_ratio": 0.80,   # 稀有类采纳所需更高比例
    "prop_high_margin": 1.5,        # CE margin 高于此视为"自信"(用于反对票)
    "prop_ce_oppose_gap": 0.5,      # CE 对多数类分数比 top1 低多少算强烈反对
    "prop_rare_anchor_threshold": 2,  # anchor 中样本数 <= 此值的类视为稀有类
    "prop_max_rounds": 50,          # 标签传播最大轮数
}

# 各版本消融开关。kNN 类型对 C/D 固定(C=plain、D=mutual),E/F 由方法名带出。
ANCHOR_PROP_VERSION_FLAGS = {
    "C": {"use_margin": False, "use_center": False, "rare_conservative": False},
    "D": {"use_margin": False, "use_center": False, "rare_conservative": False},
    "E": {"use_margin": True, "use_center": False, "rare_conservative": False},
    "F": {"use_margin": True, "use_center": True, "rare_conservative": True},
}
ANCHOR_PROP_DEFAULT_KNN_TYPE = {"C": "plain", "D": "mutual"}
ANCHOR_PROP_UNCERTAIN_MODES = ("ce_fallback", "re_llm")
DEFAULT_SEM_ASSIGNMENT_METHOD = "Anchor-F-mutual [ce_fallback]"


def _anchor_method_name(version, knn_type, uncertain_mode):
    """与 notebook 的 anchor_method_name() 一致的规范方法名。"""
    if version in {"E", "F"}:
        return f"Anchor-{version}-{knn_type} [{uncertain_mode}]"
    return f"Anchor-{version} [{uncertain_mode}]"


def _parse_anchor_method(method_name):
    """解析 Anchor 方法名 -> spec dict;非 Anchor 名(FFT-CE/FFT-LLM/None)返回 None。

    支持:
      Anchor-C [ce_fallback] / Anchor-D [re_llm]
      Anchor-E-plain [ce_fallback] / Anchor-F-mutual [re_llm] 等全部变体。
    """
    if not method_name or not isinstance(method_name, str):
        return None
    text = method_name.strip()
    if not text.startswith("Anchor-"):
        return None
    match = re.match(
        r"^Anchor-([CDEF])(?:-(plain|mutual))?\s*\[(ce_fallback|re_llm)\]$",
        text,
    )
    if match is None:
        raise ValueError(f"无法解析的 Anchor 分配方法名: {method_name!r}")
    version, knn_type, uncertain_mode = match.group(1), match.group(2), match.group(3)
    if version in {"E", "F"}:
        if knn_type is None:
            raise ValueError(f"方法 {version} 必须在名字里指定 plain/mutual: {method_name!r}")
    else:
        if knn_type is not None:
            raise ValueError(f"方法 {version} 不接受 plain/mutual 后缀: {method_name!r}")
        knn_type = ANCHOR_PROP_DEFAULT_KNN_TYPE[version]
    flags = dict(ANCHOR_PROP_VERSION_FLAGS[version])
    return {
        "method_name": _anchor_method_name(version, knn_type, uncertain_mode),
        "version": version,
        "knn_type": knn_type,
        "uncertain_mode": uncertain_mode,
        **flags,
    }


def _l2_normalize_rows(matrix):
    norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return matrix / norms


RUNTIME_FIELD_NAMES = (
    "text_encoder",
    "tokenizer",
    "executor",
    "nlp",
    "reranker",
    "sem_description_model",
    "llm_candidate_filter",
    "llm_semantic_label_client",
    "encoder_lock",
    "phrase_analyzer",
)
CO_OCCURRENCE_NODE_QUERY_WEIGHT = (1.2, 1, 0.6, 0.3)
COMPOSITIONAL_QUERY_ROLE_WEIGHT = {
    "phrase": 1.0,
    "head": 0.7,
    "modifier": 0.2,
}

# Chunk-level bounded co-occurrence boosting (chunk_cooccur_query) defaults.
# 每个 chunk 只取 top-k pair boost 求平均;ChunkScore = BaseEvidence * (1 + λ * PairBoost)。
COOCCUR_TOP_K_PAIR_BOOSTS = 3
COOCCUR_LAMBDA = 0.3
# Candidate-pool cutoff: only the top-N chunks by BaseEvidence get the (expensive) pair-boost
# pass; the rest keep pair_boost=0 and still rank by their base evidence. Because the boost is
# bounded (<= λ ≈ 30%), a chunk far down the base ranking cannot realistically overtake the pool,
# so this caps the pair-combination cost without changing the top results. Set to None to disable.
COOCCUR_CANDIDATE_POOL_SIZE = 200
# local_evidence_level:两档。same sentence=0.7(两端点 occurrence 落在同一句),否则 same chunk=0.4。
# 不再有 phrase 档:"真成短语"由 base_evidence 里满格命中的完整 phrase node 表达(且 base 层已对
# 同 group 做 phrase 优先去重),pair boost 无需也不应再用 query_group_id 顶格加成(会重复计分/虚高)。
COOCCUR_LOCAL_EVIDENCE = {
    "sentence": 0.7,
    "chunk": 0.4,
}

# Recursion limit used while pickling the deeply linked node graph.
DEEP_PICKLE_RECURSION_LIMIT = 1_000_000
# Worker-thread stack size (bytes) so a high recursion limit cannot crash the
# native C stack while pickle does its depth-first traversal.
DEEP_PICKLE_STACK_SIZE = 512 * 1024 * 1024

# Pickle a deeply linked object graph without hitting RecursionError.
#
# pickle traverses object references depth-first, so a long reference chain
# (DocumentNode -> ChunkNode -> SemNode -> TokenNode -> ...) can exceed the
# default recursion limit. We raise the limit and run the dump on a worker
# thread with a large stack, which keeps the native stack from overflowing.
#
# NOTE: sys.setrecursionlimit() and threading.stack_size() are process-global,
# not thread-local, so no other thread should rely on the default recursion
# depth while a dump is in flight. In practice save_data() / save_data_split()
# run when no other indexing/query threads are active; the worker restores the
# previous limit in a finally block.
def deep_pickle_dump(obj, fileobj, protocol=pickle.HIGHEST_PROTOCOL):
    error = {}

    def _dump():
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(DEEP_PICKLE_RECURSION_LIMIT)
        try:
            pickle.dump(obj, fileobj, protocol=protocol)
        except BaseException as exc:  # propagate to the calling thread
            error["exc"] = exc
        finally:
            sys.setrecursionlimit(old_limit)

    old_stack_size = None
    try:
        old_stack_size = threading.stack_size(DEEP_PICKLE_STACK_SIZE)
    except (ValueError, RuntimeError):
        # Platform refused the requested stack size; fall back to default.
        old_stack_size = None

    worker = threading.Thread(target=_dump)
    worker.start()
    worker.join()

    if old_stack_size is not None:
        try:
            threading.stack_size(old_stack_size)
        except (ValueError, RuntimeError):
            pass

    if "exc" in error:
        raise error["exc"]

# Compute the co-occurrence edge weight between two semantic nodes.
def get_COG_edge_weight(node_a, node_b):
    chunk_node_list_a = node_a.chunk_node_list
    chunk_node_list_b = node_b.chunk_node_list
    a_ids = {n.chunk_node_id for n in chunk_node_list_a}
    b_ids = {n.chunk_node_id for n in chunk_node_list_b}
    weight = len(a_ids & b_ids)/(math.sqrt(len(a_ids)*len(b_ids)))

    return weight

# Return unique chunk counts for two node lists and their overlap size.
def unique_counts_by_id(a_list, b_list):
    a_ids = {n.chunk_node_id for n in a_list}
    b_ids = {n.chunk_node_id for n in b_list}
    return len(a_ids), len(b_ids), len(a_ids & b_ids)

class CoOccurrenceGraph:
    # Initialize a co-occurrence graph wrapper around semantic node retrieval candidates.
    def __init__(self, sem_node_list):
        self.node_list = [CoOccurrenceNode(sem_node) for sem_node in sem_node_list]
        self.connected_node_list = []
        self.isolate_node_list = []
        self.weighted_chunk_node_list = []
        self.ranked_sem_node_list = []
        self.ranked_chunk_BM25 = []
        self.pair_edge_weight = {}

    # Build weighted co-occurrence links between semantic nodes that share chunks.
    def build_edges(self):
        for node_a, node_b in combinations(self.node_list, 2):
            weight = get_COG_edge_weight(node_a.sem_node , node_b.sem_node)
            if weight > 0:
                node_a.neighbor_node_list.append((node_b, weight))
                node_b.neighbor_node_list.append((node_a, weight))

        for node in self.node_list:
            unique_neighbor_weights = {}
            for neighbor_node, weight in node.neighbor_node_list:
                if neighbor_node.sem_node is node.sem_node:
                    continue
                neighbor_key = id(neighbor_node.sem_node)
                unique_neighbor_weights[neighbor_key] = max(
                    weight,
                    unique_neighbor_weights.get(neighbor_key, 0),
                )

            if len(unique_neighbor_weights) > 0:
                self.connected_node_list.append(node)
                for weight in unique_neighbor_weights.values():
                    node.node_weight += weight

                node.node_weight = node.node_weight * node.node_level_weight
            else:
                self.isolate_node_list.append(node)

    # Add one semantic-node contribution to a chunk score map.
    def _add_chunk_contribution(
        self,
        *,
        node,
        chunk_node_id,
        score,
        weight_map,
        token_weight_map,
        used_sem_node_ids,
    ):
        sem_node_id = id(node.sem_node)
        if sem_node_id in used_sem_node_ids:
            return False
        weight_map[chunk_node_id] = weight_map.get(chunk_node_id, 0) + score
        token_weight_map.setdefault(chunk_node_id, []).append(
            f"Token:{node.sem_node.token_node.token_text},Score:{score:.4f}"
        )
        used_sem_node_ids.add(sem_node_id)
        return True

    # Aggregate compositional phrase/head/modifier groups under their chunk constraints.
    def _assign_grouped_chunk_weight(self, grouped_nodes, weight_map, token_weight_map):
        chunk_group_candidates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for node in grouped_nodes:
            for chunk_node in node.sem_node.chunk_node_list:
                chunk_node_id = chunk_node.chunk_node_id
                bm25_score = node.node_weight * node.sem_node.BM25[chunk_node_id]
                chunk_group_candidates[chunk_node_id][node.query_group_id][node.query_group_role].append(
                    (node, bm25_score)
                )

        for chunk_node_id, group_candidates in chunk_group_candidates.items():
            consumed_member_keys = set()
            used_sem_node_ids = set()
            groups_with_phrase = set()

            for group_id in sorted(group_candidates):
                phrase_candidates = group_candidates[group_id].get("phrase", [])
                if not phrase_candidates:
                    continue
                groups_with_phrase.add(group_id)
                for node, bm25_score in phrase_candidates:
                    self._add_chunk_contribution(
                        node=node,
                        chunk_node_id=chunk_node_id,
                        score=bm25_score,
                        weight_map=weight_map,
                        token_weight_map=token_weight_map,
                        used_sem_node_ids=used_sem_node_ids,
                    )
                    consumed_member_keys.update(node.query_group_member_keys)

            for group_id in sorted(group_candidates):
                if group_id in groups_with_phrase:
                    continue
                for role in ("head", "modifier"):
                    for node, bm25_score in group_candidates[group_id].get(role, []):
                        member_key = node.query_member_key
                        if member_key and member_key in consumed_member_keys:
                            continue
                        added = self._add_chunk_contribution(
                            node=node,
                            chunk_node_id=chunk_node_id,
                            score=bm25_score,
                            weight_map=weight_map,
                            token_weight_map=token_weight_map,
                            used_sem_node_ids=used_sem_node_ids,
                        )
                        if added and member_key:
                            consumed_member_keys.add(member_key)

    # Aggregate semantic node weights into chunk-level retrieval scores.
    def assign_chunk_weight(self, avg_chunk_len, debug_mode=False):
        weight_map = {}
        token_record_map = defaultdict(set)
        token_weight_map = {}

        if len(self.connected_node_list) > 0 :
            grouped_nodes = []
            for node in self.connected_node_list:
                if node.query_group_id is not None:
                    grouped_nodes.append(node)
                    continue
                for chunk_node in node.sem_node.chunk_node_list:
                    chunk_node_id = chunk_node.chunk_node_id
                    if node.sem_node.token_node.token_text not in token_record_map[chunk_node_id]:

                        bm25_score = node.node_weight * node.sem_node.BM25[chunk_node_id]
                        weight_map[chunk_node_id] = weight_map.get(chunk_node_id, 0) + bm25_score
                        token_weight_map.setdefault(chunk_node_id, []).append(f"Token:{node.sem_node.token_node.token_text},Score:{(bm25_score ):.4f}")
                        token_record_map[chunk_node_id].add(node.sem_node.token_node.token_text)

            self._assign_grouped_chunk_weight(grouped_nodes, weight_map, token_weight_map)
            self.weighted_chunk_node_list = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)
        if debug_mode:
            print(token_weight_map)
            print(weight_map)
            print(self.weighted_chunk_node_list)
    # Chunk-level bounded boosting: compute global edge strengths WITHOUT mutating node_weight.
    #
    # Unlike build_edges(), this never folds edge weight into node_weight; it only returns a
    # lookup of normalized-overlap weights keyed by the unordered pair of semantic-node ids.
    def build_pair_edges(self):
        self.pair_edge_weight = {}
        for node_a, node_b in combinations(self.node_list, 2):
            if node_a.sem_node is node_b.sem_node:
                continue
            weight = get_COG_edge_weight(node_a.sem_node, node_b.sem_node)
            if weight <= 0:
                continue
            key = frozenset((id(node_a.sem_node), id(node_b.sem_node)))
            if weight > self.pair_edge_weight.get(key, 0):
                self.pair_edge_weight[key] = weight
        return self.pair_edge_weight

    # Chunk-level bounded boosting: BaseEvidence(c) for every candidate chunk.
    #
    # Each node contributes node_level_weight * BM25[c] (role/level weight, NOT edge weight),
    # reusing the existing phrase/head/modifier grouped aggregation. Computed over ALL nodes
    # (connected + isolated), so isolated chunks with strong base evidence still rank.
    def assign_base_evidence(self):
        weight_map = {}
        token_record_map = defaultdict(set)
        token_weight_map = {}
        grouped_nodes = []

        for node in self.node_list:
            node.node_weight = node.node_level_weight

        for node in self.node_list:
            if node.query_group_id is not None:
                grouped_nodes.append(node)
                continue
            for chunk_node in node.sem_node.chunk_node_list:
                chunk_node_id = chunk_node.chunk_node_id
                token_text = node.sem_node.token_node.token_text
                if token_text in token_record_map[chunk_node_id]:
                    continue
                bm25_score = node.node_weight * node.sem_node.BM25[chunk_node_id]
                weight_map[chunk_node_id] = weight_map.get(chunk_node_id, 0) + bm25_score
                token_weight_map.setdefault(chunk_node_id, []).append(
                    f"Token:{token_text},Score:{bm25_score:.4f}"
                )
                token_record_map[chunk_node_id].add(token_text)

        self._assign_grouped_chunk_weight(grouped_nodes, weight_map, token_weight_map)
        return weight_map, token_weight_map

    # Chunk-level bounded boosting: map each chunk id to the matched nodes appearing in it.
    #
    # Nodes are deduplicated per chunk by semantic-node identity, so head/modifier members of a
    # query phrase are kept as distinct nodes (BaseEvidence phrase dedup does NOT apply here).
    def nodes_by_chunk(self):
        chunk_to_nodes = defaultdict(list)
        seen_sem_ids = defaultdict(set)
        for node in self.node_list:
            sem_id = id(node.sem_node)
            for chunk_node in node.sem_node.chunk_node_list:
                chunk_node_id = chunk_node.chunk_node_id
                if sem_id in seen_sem_ids[chunk_node_id]:
                    continue
                seen_sem_ids[chunk_node_id].add(sem_id)
                chunk_to_nodes[chunk_node_id].append(node)
        return chunk_to_nodes

    # Rank chunks within each semantic node level using accumulated BM25 scores.
    def rank_chunk_by_BM25(self):
        self.rank_sem_node_by_level()
        results = []

        for group in self.ranked_sem_node_list:
            chunk_scores = defaultdict(float)

            for inst in group:
                for chunk_id, score in inst.BM25.items():
                    chunk_scores[chunk_id] += score

            ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

            results.append([chunk_id for chunk_id, _ in ranked])
        self.ranked_chunk_BM25 = results

    # Group semantic nodes by their query-match level.
    def rank_sem_node_by_level(self):
        self.ranked_sem_node_list = [[] for i in range(len(CO_OCCURRENCE_NODE_QUERY_WEIGHT))]
        for node in self.node_list:
            self.ranked_sem_node_list[node.node_level].append(node.sem_node)

    # Sort semantic nodes by match level and token IDF for downstream retrieval.
    def rank_sem_node(self):
        ranked_con_list = sorted(
            self.node_list,
            key=lambda node: (node.node_level, -node.sem_node.token_node.idf)
        )
        self.ranked_sem_node_list = [co_node.sem_node for co_node in ranked_con_list]

    # Print the current retrieval weight assigned to each semantic node.
    def print_node_weight(self):
        node_weights = []
        for node in self.node_list:
            node_weights.append(f"token:{node.sem_node.token_node.token_text}, weight: {node.node_weight:.4f}")
        print(node_weights)

CoOccurrenceNode_query_weight = CO_OCCURRENCE_NODE_QUERY_WEIGHT

@dataclass
class CoOccurrenceNode:
    sem_node: object
    node_level: int
    node_query_weight: float
    neighbor_node_list: list = field(default_factory=list)
    node_weight: float = 0
    node_level_weight: float = field(init=False)

    # Store a semantic node candidate with its match level and query weight.
    def __init__(self, sem_node_info):
        self.sem_node = sem_node_info[0]
        self.node_level = sem_node_info[1]
        self.node_query_weight = sem_node_info[2]
        if self.node_level < 0 or self.node_level >= len(CO_OCCURRENCE_NODE_QUERY_WEIGHT):
            raise ValueError(f"Unsupported co-occurrence node level: {self.node_level}")
        self.query_group_id = None
        self.query_group_role = None
        self.query_member_key = None
        self.query_group_member_keys = set()
        role_weight = None
        if len(sem_node_info) > 3:
            self.query_group_id = sem_node_info[3]
            self.query_group_role = sem_node_info[4]
            self.query_member_key = sem_node_info[5]
            self.query_group_member_keys = set(sem_node_info[6] or [])
            role_weight = sem_node_info[7]
        self.neighbor_node_list = []
        self.node_weight = 0
        if role_weight is None:
            self.node_level_weight = CO_OCCURRENCE_NODE_QUERY_WEIGHT[self.node_level]
        else:
            self.node_level_weight = role_weight
        # Fold the per-match query weight into the effective node weight. This value
        # is always in (0, 1]: exact matches pass 1.0 (no change); fuzzy matches pass
        # the phrase coverage ratio (query words / matched-phrase words, <=1 because a
        # fuzzy hit must contain every query word); semantic fallbacks pass the cosine
        # similarity. Previously it was stored on the node but never read.
        self.node_level_weight *= self.node_query_weight

@dataclass
class DocumentNode:
    doc_name: str
    doc_node_id: int
    chunk_node_list: list = field(default_factory=list)

@dataclass
class ChunkNode:
    chunk_text: str
    chunk_node_id: int
    doc_node: DocumentNode
    sem_node_list: list = field(default_factory=list)
    length_norm: float = 0
    num_tokens: int | None = None
    # schema 7: sorted sentence-end char offsets (spaCy sent.end_char), aligned with chunk_text.
    sentence_boundaries: list | None = None

@dataclass
class TokenNode:
    token_text: str
    token_node_id: int
    node_type: str
    is_multi_semantic: bool | None = field(init=False)
    descriptions: list | None = field(init=False)
    wikidata_info_loaded: bool | None = field(init=False)
    has_semantic: bool = False
    sem_node_list: list = field(default_factory=list)
    embeds_buffer: list = field(default_factory=list)
    span_occurrences: list = field(default_factory=list)
    idf: float = 0
    df: int = 0
    is_placeholder: bool = False
    force_single_semantic: bool = False
    compositional_head: object | None = None
    compositional_head_text: str | None = None
    compositional_head_text_norm: str | None = None
    compositional_modifiers: list = field(default_factory=list)
    compositional_modifier_texts: list = field(default_factory=list)
    compositional_modifier_texts_norm: list = field(default_factory=list)
    headed_phrase_records: dict = field(default_factory=dict)
    # finalize 时 semantic_type_cls 算出的 embedding 集中度;None 表示从未走过
    # s_mean 闸门(entity/原子短语单义,或出现次数未达阈值的 basic 节点)。
    s_mean: float | None = None
    # schema 8: Anchor 传播中由 LLM 直接判定的 anchor 样本记录。
    llm_anchor_sample_assignments: list = field(default_factory=list)

    # Initialize optional token metadata after dataclass construction.
    def __post_init__(self):
        self.is_multi_semantic = None
        self.descriptions = None
        self.wikidata_info_loaded = None

# Map a span's start offset to the index of its chunk sentence using precomputed
# sentence boundaries (schema 7). Returns None when boundaries or the offset are absent.
def sentence_id_for_span(chunk_node, span_start):
    if span_start is None or chunk_node is None:
        return None
    boundaries = getattr(chunk_node, "sentence_boundaries", None)
    if not boundaries:
        return None
    return bisect_right(boundaries, span_start)

@dataclass
class TextEmbedding:
    embed: object
    chunk_node: ChunkNode
    span_start: int | None = None
    span_end: int | None = None
    span_text: str | None = None
    surface_text: str | None = None
    surface_start: int | None = None
    surface_end: int | None = None
    phrase_type: str | None = None
    modifier_text: str | None = None
    modifier_texts: list = field(default_factory=list)
    modifier_texts_norm: list = field(default_factory=list)
    modifier_spans: list = field(default_factory=list)
    atomic_modifier_spans: list = field(default_factory=list)
    # schema 7: index-time sentence index of this span within its chunk (None if unknown).
    sentence_id: int | None = None

    # Convert a retained text embedding into a span occurrence record.
    def to_span_occurrence(self):
        span_text = self.span_text
        if (
            span_text is None
            and self.span_start is not None
            and self.span_end is not None
        ):
            span_text = self.chunk_node.chunk_text[self.span_start:self.span_end]
        sentence_id = self.sentence_id
        if sentence_id is None:
            sentence_id = sentence_id_for_span(self.chunk_node, self.span_start)
        return SpanOccurrence(
            chunk_node=self.chunk_node,
            span_start=self.span_start,
            span_end=self.span_end,
            span_text=span_text,
            surface_text=self.surface_text or span_text,
            surface_start=self.surface_start if self.surface_start is not None else self.span_start,
            surface_end=self.surface_end if self.surface_end is not None else self.span_end,
            phrase_type=self.phrase_type,
            modifier_text=self.modifier_text,
            modifier_texts=list(self.modifier_texts or []),
            modifier_texts_norm=list(self.modifier_texts_norm or []),
            modifier_spans=list(self.modifier_spans or []),
            atomic_modifier_spans=list(self.atomic_modifier_spans or []),
            sentence_id=sentence_id,
        )

@dataclass
class SpanOccurrence:
    chunk_node: ChunkNode
    span_start: int | None = None
    span_end: int | None = None
    span_text: str | None = None
    surface_text: str | None = None
    surface_start: int | None = None
    surface_end: int | None = None
    phrase_type: str | None = None
    modifier_text: str | None = None
    modifier_texts: list = field(default_factory=list)
    modifier_texts_norm: list = field(default_factory=list)
    modifier_spans: list = field(default_factory=list)
    atomic_modifier_spans: list = field(default_factory=list)
    # schema 7: index-time sentence index of this span within its chunk (None if unknown).
    sentence_id: int | None = None

    # Return the span boundaries when both endpoints are available.
    def get_span_tuple(self):
        if self.span_start is None or self.span_end is None:
            return None
        return self.span_start, self.span_end

@dataclass
class SemNode:
    sem_node_id: int
    token_node: TokenNode
    description: str | None = None
    chunk_node_list: list = field(default_factory=list)
    span_occurrences: list = field(default_factory=list)
    chunk_node_embed: list = field(default_factory=list)
    retained_text_embeddings: list = field(default_factory=list)
    retained_text_embedding_source_count: int = 0
    pending_embed_rebuild: bool = False
    chunk_edge_weight: list = field(default_factory=list)
    embed: object = None
    tf_dict_by_chunk_id: dict | None = None
    idf: float = 0
    df: int = 0
    chunk_len_dict_by_id: dict | None = None
    BM25: dict | None = None

    # Keep occurrence-related lists aligned when adding a text embedding occurrence.
    def append_text_embedding_occurrence(self, text_embedding, edge_weight=None):
        self.chunk_node_list.append(text_embedding.chunk_node)
        self.span_occurrences.append(text_embedding.to_span_occurrence())
        if edge_weight is not None:
            self.chunk_edge_weight.append(edge_weight)

    # Keep occurrence-related lists aligned when adding a stored span occurrence.
    def append_span_occurrence(self, span_occurrence, edge_weight=None):
        self.chunk_node_list.append(span_occurrence.chunk_node)
        self.span_occurrences.append(
            SpanOccurrence(
                chunk_node=span_occurrence.chunk_node,
                span_start=span_occurrence.span_start,
                span_end=span_occurrence.span_end,
                span_text=span_occurrence.span_text,
                surface_text=getattr(span_occurrence, "surface_text", None),
                surface_start=getattr(span_occurrence, "surface_start", None),
                surface_end=getattr(span_occurrence, "surface_end", None),
                phrase_type=getattr(span_occurrence, "phrase_type", None),
                modifier_text=getattr(span_occurrence, "modifier_text", None),
                modifier_texts=list(getattr(span_occurrence, "modifier_texts", []) or []),
                modifier_texts_norm=list(getattr(span_occurrence, "modifier_texts_norm", []) or []),
                modifier_spans=list(getattr(span_occurrence, "modifier_spans", []) or []),
                atomic_modifier_spans=list(getattr(span_occurrence, "atomic_modifier_spans", []) or []),
                sentence_id=getattr(span_occurrence, "sentence_id", None),
            )
        )
        if edge_weight is not None:
            self.chunk_edge_weight.append(edge_weight)

    # Filter parallel occurrence lists with the same index set.
    def keep_occurrence_indices(self, keep_indices):
        self.chunk_node_list = [self.chunk_node_list[i] for i in keep_indices]
        if getattr(self, "span_occurrences", None):
            self.span_occurrences = [
                self.span_occurrences[i]
                for i in keep_indices
                if i < len(self.span_occurrences)
            ]
        else:
            self.span_occurrences = []
        self.chunk_edge_weight = [
            self.chunk_edge_weight[i]
            for i in keep_indices
            if i < len(self.chunk_edge_weight)
        ]

    # Build per-chunk term-frequency and length lookup tables for this semantic node.
    def get_tf(self):
        self.tf_dict_by_chunk_id = dict(Counter([chunk_node.chunk_node_id for chunk_node in self.chunk_node_list]))
        self.chunk_len_dict_by_id = {chunk_node.chunk_node_id: chunk_node.num_tokens for chunk_node in self.chunk_node_list}

    # Compute BM25 scores for chunks connected to this semantic node.
    def get_BM25(self, avg_chunk_len,  k1=1.2, b=0.75):
        self.get_tf()
        scores = {}
        for chunk_id, tf in self.tf_dict_by_chunk_id.items():
            doc_len = self.chunk_len_dict_by_id[chunk_id]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_chunk_len)
            score = self.idf * numerator / denominator
            scores[chunk_id] = score
        self.BM25 = scores

# Backward-compatible alias for old imports and pickles.
Prototype = SemNode


def _find_description_match_span(chunk_text, token_text):
    normalized_token = token_text.strip()
    if not normalized_token:
        return None

    pattern = re.compile(rf"(?<!\w){re.escape(normalized_token)}(?!\w)", flags=re.IGNORECASE)
    match = pattern.search(chunk_text)
    if match is None:
        lowered_text = chunk_text.casefold()
        lowered_token = normalized_token.casefold()
        match_start = lowered_text.find(lowered_token)
        if match_start < 0:
            return None
        match_end = match_start + len(lowered_token)
    else:
        match_start, match_end = match.span()
    return match_start, match_end


class LiteSemRAG:
    # Initialize graph storage, model handles, retrieval settings, and runtime state.
    def __init__(self,
                 min_occurrences_for_description=20,
                 retrieve_top_k=5, chunk_size=300, device="cuda",
                 discard_no_word=False,
                 sem_description_prompt_context_mode="sentence_neighbors",
                 consensus_ratio_threshold=0.8,
                 min_description_candidates=3,
                 use_llm_candidate_filter=False,
                 llm_candidate_filter_use_api=True,
                 llm_candidate_filter_provider="openai",
                 llm_candidate_filter_api_model=None,
                 llm_candidate_filter_cache_path="cache/wikidata_definition_filter_cache.sqlite3",
                 llm_candidate_filter_max_tokens=4096,
                 use_llm_semantic_labeler=False,
                 disambiguate_query_sense=True,
                 llm_semantic_labeler_provider="openai",
                 llm_semantic_labeler_model="gpt-5.4-mini",
                 llm_semantic_labeler_api_key_file="API_KEY",
                 llm_semantic_labeler_cache_path="cache/llm_semantic_label_cache.sqlite3",
                 llm_semantic_labeler_context_word_window=20,
                 llm_semantic_labeler_max_tokens=256,
                 llm_semantic_labeler_batch_size=10,
                 fft_max_samples=10,
                 fft_d1_d2_ratio_threshold=0.8,
                 fft_knn_check_k=5,
                 fft_danger_neighbor_m=10,
                 sem_assignment_method=DEFAULT_SEM_ASSIGNMENT_METHOD,
                 anchor_fraction=ANCHOR_PROP_DEFAULTS["anchor_fraction"],
                 anchor_fft_ratio=ANCHOR_PROP_DEFAULTS["anchor_fft_ratio"],
                 anchor_min_count=ANCHOR_PROP_DEFAULTS["anchor_min_count"],
                 anchor_max_count=ANCHOR_PROP_DEFAULTS["anchor_max_count"],
                 prop_knn_k=ANCHOR_PROP_DEFAULTS["prop_knn_k"],
                 tau_conc=0.90,
                 fast_index=False,
                 phrase_audit_enabled=False,
                 phrase_audit_cache_path="cache/phrase_protection_audit.jsonl"):
        self.doc_nodes = []
        self.chunk_nodes = []
        self.token_nodes = []
        self.phrase_token_nodes = []
        self.sem_nodes = []
        self.next_doc_node_id = 0
        self.next_chunk_node_id = 0
        self.next_token_node_id = 0
        self.next_sem_node_id = 0
        # 索引期不再增量建语义节点:每个 token/phrase 的所有出现都先累积在
        # token_node.embeds_buffer 里,直到所有文档扫描完毕,finalize 时才统一跑
        # 多义判定(build_sem_node)。一个 token/phrase 至少出现这么多次才会跑描述
        # 预测 / 多义切分,低于该阈值的走 create_basic_sem_node(单节点、无描述)。
        # 设为 None 则所有 token 都只建单节点、无描述;设为 1 则所有 token 都做描述。
        self.min_occurrences_for_description = min_occurrences_for_description
        # fast index 模式:finalize 时所有 token/phrase 一律建单义 basic 节点,
        # 完全跳过 build_sem_node 的 s_mean 闸门 / FFT 采样 / 描述预测 / 多义切分。
        # 出现次数阈值、entity、force_single 都不再影响分流。索引期短语 head/modifier
        # 路由与 finalize 下游(modifier postings / IDF / BM25 / query_database /
        # phrase_index / chunk->sem 边)照常,检索可正常工作。
        self.fast_index = bool(fast_index)
        self.token_node_query = {}
        # s_mean 闸门:某 token 各次出现 embedding 的平均相似度高于该阈值 →
        # 语义集中,判为单义不切分(_is_token_polysemous_by_s_mean)。值越大越倾向多义切分。
        self.tau_conc = float(tau_conc)
        self.tau_disp = 0.78
        self.device = device
        self.query_database = None
        self.retrieve_top_k = retrieve_top_k
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.start_time = None
        self.index_session_start_time = None
        self.index_session_document_count = 0
        self.phrase_index = defaultdict(set)
        self.modifier_postings = defaultdict(Counter)
        self.nlp = None
        self.phrase_analyzer = None
        self.text_encoder = None
        self.tokenizer = None
        self.encoder_lock = threading.Lock()
        self.text_encoder_addr = "/home/xiaoyue/ProtoGraphRAG/deberta-v3-large"
        self._load_text_encoder()
        self._load_nlp()
        self.json_path = "index_documents.json"
        self.reranker = None
        self.load_reranker()
        self.sem_description_model_name = "cross-encoder/nli-deberta-v3-large"
        self.sem_description_model = None
        self._load_sem_description_model()
        self.sem_description_candidate_limit = 5
        self.llm_candidate_filter_candidate_limit = 10
        self.use_llm_candidate_filter = bool(use_llm_candidate_filter)
        self.llm_candidate_filter_use_api = bool(llm_candidate_filter_use_api)
        self.llm_candidate_filter_provider = llm_candidate_filter_provider
        self.llm_candidate_filter_api_model = llm_candidate_filter_api_model
        self.llm_candidate_filter_cache_path = llm_candidate_filter_cache_path
        # gpt-5* 是 reasoning 模型,max_completion_tokens 预算由隐藏推理 token 与可见
        # 输出共享;合并多义项 + 判 FFT 样本的 JSON 输出较长,1024 容易被截断导致
        # 解析失败 / 空 definitions(被记成 no_candidate_definitions),故默认放大。
        self.llm_candidate_filter_max_tokens = max(1, int(llm_candidate_filter_max_tokens))
        self.llm_candidate_filter = None
        self.llm_candidate_filter_total_tokens = 0
        self.llm_candidate_filter_prompt_tokens = 0
        self.llm_candidate_filter_completion_tokens = 0
        self.llm_candidate_filter_total_wall_time = 0.0
        self.llm_candidate_filter_api_wait_wall_time = 0.0
        self.use_llm_semantic_labeler = bool(use_llm_semantic_labeler)
        # 查询期多义消歧:当一个 span 命中的 token 在索引期生成了多个带描述的 sem
        # node(多义)时,复用索引期的语义判断(cross-encoder / LLM)来选择对应的
        # sem node,而不是单纯比 embedding 相似度。只有一个 sem node 时直接取用。
        self.disambiguate_query_sense = bool(disambiguate_query_sense)
        self.llm_semantic_labeler_provider = llm_semantic_labeler_provider
        self.llm_semantic_labeler_model = llm_semantic_labeler_model
        self.llm_semantic_labeler_api_key_file = llm_semantic_labeler_api_key_file
        self.llm_semantic_labeler_cache_path = llm_semantic_labeler_cache_path
        self.llm_semantic_labeler_context_word_window = int(llm_semantic_labeler_context_word_window)
        self.llm_semantic_labeler_max_tokens = int(llm_semantic_labeler_max_tokens)
        self.llm_semantic_labeler_batch_size = max(1, int(llm_semantic_labeler_batch_size))
        self.llm_semantic_label_client = None
        self.llm_semantic_label_total_tokens = 0
        self.llm_semantic_label_cache_hits = 0
        self.llm_semantic_label_cache_misses = 0
        self.fft_propagation_wall_time = 0.0
        self.fft_max_samples = max(1, int(fft_max_samples))
        self.fft_d1_d2_ratio_threshold = float(fft_d1_d2_ratio_threshold)
        self.fft_knn_check_k = max(0, int(fft_knn_check_k))
        self.fft_danger_neighbor_m = max(0, int(fft_danger_neighbor_m))
        # 语义分配方法选择:Anchor-{C,D,E,F}[-plain/mutual] [ce_fallback|re_llm]
        # 走 anchor 传播路径;其它(FFT-CE / FFT-LLM / None)沿用既有
        # _assign_sem_description_on_build。其余 anchor 数值超参取 notebook 默认值,
        # 存为实例属性、可按需覆盖。
        self.anchor_prop_params = dict(ANCHOR_PROP_DEFAULTS)
        self.anchor_prop_params["anchor_fraction"] = float(anchor_fraction)
        self.anchor_prop_params["anchor_fft_ratio"] = float(anchor_fft_ratio)
        self.anchor_prop_params["anchor_min_count"] = max(1, int(anchor_min_count))
        self.anchor_prop_params["anchor_max_count"] = max(1, int(anchor_max_count))
        self.anchor_prop_params["prop_knn_k"] = max(1, int(prop_knn_k))
        self._set_sem_assignment_method(sem_assignment_method)
        self.sem_description_batch_size = 32
        self.sem_description_use_detailed_description = False
        self.sem_description_require_detailed_description = True
        self.sem_description_exact_match_text = False
        self.sem_description_label_contains_text = True
        self.sem_description_exact_match_first = False
        self.sem_retained_embed_limit = 10
        self.sem_description_log_path = "sem_description_logs.txt"
        self.chunk_avg_len = None
        self.discard_no_word = discard_no_word
        self.sem_description_prompt_context_mode = sem_description_prompt_context_mode
        # Minimum agreement ratio (top predicted description / total samples) required
        # to accept a cross-encoder prediction as the semantic node's description.
        self.consensus_ratio_threshold = consensus_ratio_threshold
        self.min_description_candidates = max(
            1, int(min_description_candidates)
        )
        self.predicted_sem_description_logs = []
        self.deleted_merged_sem_logs = []
        self.sem_description_operation_logs = []
        self.wikidata_no_result_logs = []
        self._wikidata_no_result_keys = set()
        self.llm_semantic_label_total_tokens = 0
        self.llm_semantic_label_cache_hits = 0
        self.llm_semantic_label_cache_misses = 0
        self._sem_description_candidate_bank_cache = {}
        # token_text -> {sample_order: merged_description} produced by the combined
        # candidate-merge + FFT-sample-judgment LLM call (use_llm_candidate_filter).
        self._combined_merge_sample_sense_descriptions = {}
        self.llm_candidate_filter_result_cache = {}
        self.llm_anchor_sample_assignment_count = 0
        self.sem_build_count = 0
        self._reset_sem_build_stats()
        self.phrase_audit_enabled = bool(phrase_audit_enabled)
        self.phrase_audit_cache_path = phrase_audit_cache_path
        self._phrase_audit_session_id = None
        self.schema_version = CURRENT_SCHEMA_VERSION

    # Select the semantic-assignment method and cache the parsed anchor spec.
    # Anchor 方法名 -> 走传播路径(spec 非 None);其它名(FFT-CE/FFT-LLM/None)
    # 沿用既有 cross-encoder/LLM 的 consensus+d1/d2 路径。
    def _set_sem_assignment_method(self, method_name):
        self.sem_assignment_method = method_name
        self._anchor_method_spec = _parse_anchor_method(method_name)

    # Convenience accessor for the active anchor hyper-parameters.
    def _anchor_param(self, key):
        params = getattr(self, "anchor_prop_params", None) or {}
        return params.get(key, ANCHOR_PROP_DEFAULTS[key])

    # Recompute the average chunk length used by BM25 scoring.
    def _recompute_chunk_avg_len(self):
        self.chunk_avg_len = None
        if self.chunk_nodes:
            self.chunk_avg_len = (
                sum(chunk_node.num_tokens for chunk_node in self.chunk_nodes)
                / len(self.chunk_nodes)
            )
        return self.chunk_avg_len

    # Initialize or disable the reranker used after graph retrieval.
    def load_reranker(self):

        self.reranker = None

    # Load the cross-encoder used to predict semantic node descriptions.
    def _load_sem_description_model(self):
        from sentence_transformers import CrossEncoder

        self.sem_description_model = CrossEncoder(self.sem_description_model_name)

    # Lazily construct the LLM-backed Wikidata candidate filter.
    def _get_llm_candidate_filter(self):
        if self.llm_candidate_filter is None:
            from wikidata_definition_filter import WikidataDefinitionFilter

            self.llm_candidate_filter = WikidataDefinitionFilter(
                use_api=self.llm_candidate_filter_use_api,
                api_provider=self.llm_candidate_filter_provider,
                api_model=self.llm_candidate_filter_api_model,
                cache_path=self.llm_candidate_filter_cache_path,
                max_tokens=getattr(self, "llm_candidate_filter_max_tokens", 4096),
            )
        return self.llm_candidate_filter

    # Lazily construct the LLM client used for per-span semantic labeling.
    def _get_llm_semantic_label_client(self):
        if self.llm_semantic_label_client is None:
            from local_llm import LocalLLMClient, LocalLLMConfig

            config = LocalLLMConfig.from_env(
                provider=self.llm_semantic_labeler_provider,
                model=self.llm_semantic_labeler_model,
                api_key_file=self.llm_semantic_labeler_api_key_file,
            )
            self.llm_semantic_label_client = LocalLLMClient(config)
        return self.llm_semantic_label_client

    # Accumulate token usage from one LLM candidate-filter call.
    def _record_llm_candidate_filter_usage(self, metadata):
        metadata = metadata or {}
        usage = metadata.get("token_usage") or {}
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = usage.get("total_tokens_delta")
        if isinstance(total_tokens, int):
            self.llm_candidate_filter_total_tokens += total_tokens
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            self.llm_candidate_filter_prompt_tokens += prompt_tokens
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            self.llm_candidate_filter_completion_tokens += completion_tokens
        api_wait_wall_time = metadata.get("api_wait_wall_time")
        if isinstance(api_wait_wall_time, (int, float)):
            self.llm_candidate_filter_api_wait_wall_time += float(api_wait_wall_time)

    # Keep stored semantic-node embeddings on CPU; move them to self.device only for computation.
    def _store_sem_embed(self, embed):
        if embed is None:
            return None
        if torch.is_tensor(embed):
            return embed.detach().cpu()
        return torch.as_tensor(embed).detach().cpu()

    # Serialize access to the shared encoder when chunk encoding is submitted from worker threads.
    def _encode_chunk_thread_safe(self, chunk):
        with self.encoder_lock:
            return encode_chunk(chunk, self.text_encoder, self.tokenizer, self.device)

    # Shut down the background executor used by indexing workers.
    def shutdown(self):
        self.executor.shutdown()

    # Allocate the next integer ID for the requested node type.
    def _new_node_id(self, node_type):
        if node_type == "doc":
            pid = self.next_doc_node_id
            self.next_doc_node_id += 1
        elif node_type == "chunk":
            pid = self.next_chunk_node_id
            self.next_chunk_node_id += 1
        elif node_type == "token":
            pid = self.next_token_node_id
            self.next_token_node_id += 1
        else:
            pid = self.next_sem_node_id
            self.next_sem_node_id += 1
        return pid

    # Clear semantic-description and Wikidata lookup logs for a new finalize run.
    def _reset_sem_description_logs(self):
        self.predicted_sem_description_logs = []
        self.deleted_merged_sem_logs = []
        self.sem_description_operation_logs = []
        self.wikidata_no_result_logs = []
        self._wikidata_no_result_keys = set()

    # Clear persisted per-token LLM anchor assignment records before rebuilding sem nodes.
    def _reset_llm_anchor_sample_assignments(self):
        self.llm_anchor_sample_assignment_count = 0
        for token_node in getattr(self, "token_nodes", []) or []:
            token_node.llm_anchor_sample_assignments = []

    # Append a structured semantic-description operation log entry.
    def _log_sem_description_operation(self, event_type, **payload):
        self.sem_description_operation_logs.append(
            {
                "event_type": event_type,
                **payload,
            }
        )

    # Configure phrase protection audit logging for future indexing calls.
    def enable_phrase_audit(self, cache_path=None):
        if cache_path is not None:
            self.phrase_audit_cache_path = cache_path
        self.phrase_audit_enabled = True
        return self.phrase_audit_cache_path

    # Disable phrase protection audit logging.
    def disable_phrase_audit(self):
        self.phrase_audit_enabled = False

    # Prepare the JSONL audit cache and return this indexing session ID.
    def _start_phrase_audit_session(self):
        if not getattr(self, "phrase_audit_enabled", False):
            self._phrase_audit_session_id = None
            return None

        cache_path = getattr(
            self,
            "phrase_audit_cache_path",
            "cache/phrase_protection_audit.jsonl",
        )
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._phrase_audit_session_id = time.strftime("%Y%m%d_%H%M%S")
        return self._phrase_audit_session_id

    # Append phrase protection audit records as JSONL.
    def _append_phrase_audit_records(self, records, doc_name, chunk_node, chunk_index):
        if not getattr(self, "phrase_audit_enabled", False) or not records:
            return

        cache_path = getattr(
            self,
            "phrase_audit_cache_path",
            "cache/phrase_protection_audit.jsonl",
        )
        session_id = getattr(self, "_phrase_audit_session_id", None)
        with open(cache_path, "a", encoding="utf-8") as handle:
            for record in records:
                payload = {
                    "session_id": session_id,
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                    "chunk_node_id": getattr(chunk_node, "chunk_node_id", None),
                    **record,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # Record a deduplicated Wikidata lookup failure for later inspection.
    def _log_wikidata_no_result(self, term, stage, reason):
        log_key = (str(term), str(stage), str(reason))
        if log_key in self._wikidata_no_result_keys:
            return
        self._wikidata_no_result_keys.add(log_key)
        self.wikidata_no_result_logs.append(
            {
                "term": str(term),
                "stage": str(stage),
                "reason": str(reason),
            }
        )

    # Create and register a document node.
    def create_doc_node(self, doc_name):
        new_doc_node = DocumentNode(doc_name, self._new_node_id("doc"))
        self.doc_nodes.append(new_doc_node)
        return new_doc_node

    # Create and register a chunk node under a document node.
    def create_chunk_node(self, chunk_text, doc_node, num_tokens=None, sentence_boundaries=None):
        new_chunk_node = ChunkNode(chunk_text, self._new_node_id("chunk"), doc_node)
        if num_tokens is None:
            num_tokens = get_num_tokens(chunk_text, self.nlp)
        new_chunk_node.num_tokens = num_tokens
        new_chunk_node.sentence_boundaries = sentence_boundaries
        self.chunk_nodes.append(new_chunk_node)
        doc_node.chunk_node_list.append(new_chunk_node)
        return new_chunk_node

    # Create and register a token or phrase token node.
    def create_token_node(self, token_text, is_placeholder=False):
        node_type = 'phrase' if len(token_text.split()) >= 2 else 'token'
        new_token_node = TokenNode(token_text, self._new_node_id("token"), node_type)
        new_token_node.is_placeholder = bool(is_placeholder)
        self.token_nodes.append(new_token_node)
        if new_token_node.node_type == "phrase":
            self.phrase_token_nodes.append(new_token_node)
        self.token_node_query[token_text] = new_token_node
        return new_token_node

    # Return an existing token node or create one, optionally as a temporary placeholder.
    def _get_or_create_token_node(self, token_text, is_placeholder=False):
        token_text = normalize_text(str(token_text).strip())
        if not token_text:
            return None
        token_node = self.query_token_node(token_text)
        if token_node is None:
            return self.create_token_node(token_text, is_placeholder=is_placeholder)
        if not is_placeholder:
            token_node.is_placeholder = False
        return token_node

    # Create and register a semantic node for a token node.
    def create_sem_node(self, token_node):
        new_sem_node = SemNode(self._new_node_id("sem"), token_node)
        new_sem_node.description = self._get_initial_sem_description(token_node)
        self.sem_nodes.append(new_sem_node)
        token_node.has_semantic = True
        return new_sem_node

    # Keep semantic node IDs aligned with the current semantic-node list.
    def _reset_sem_node_ids(self):
        for index, sem_node in enumerate(self.sem_nodes):
            sem_node.sem_node_id = index
        self.next_sem_node_id = len(self.sem_nodes)

    # Return the runtime phrase analyzer used by indexing.
    def _get_phrase_analyzer(self):
        if self.phrase_analyzer is None:
            self.phrase_analyzer = PhraseAnalyzer(self.nlp)
        return self.phrase_analyzer

    # Return occurrence metadata for a routed index span.
    def _build_occurrence_metadata(
        self,
        *,
        phrase_type=None,
        surface_text=None,
        surface_start=None,
        surface_end=None,
        modifier_texts=None,
        modifier_texts_norm=None,
        modifier_spans=None,
        atomic_modifier_spans=None,
    ):
        modifier_texts = list(modifier_texts or [])
        return {
            "phrase_type": phrase_type,
            "surface_text": surface_text,
            "surface_start": surface_start,
            "surface_end": surface_end,
            "modifier_text": " ".join(modifier_texts) if modifier_texts else None,
            "modifier_texts": modifier_texts,
            "modifier_texts_norm": list(modifier_texts_norm or []),
            "modifier_spans": list(modifier_spans or []),
            "atomic_modifier_spans": list(atomic_modifier_spans or []),
        }

    # Attach compositional phrase/head/modifier relation metadata to token nodes.
    def _register_compositional_phrase_relation(self, phrase_token_node, occurrence_metadata, chunk_node):
        if not occurrence_metadata:
            return
        if occurrence_metadata.get("compositional_role") != "phrase":
            return
        head_text_norm = normalize_text(occurrence_metadata.get("head_text_norm") or "")
        if not head_text_norm:
            return

        head_node = self._get_or_create_token_node(head_text_norm, is_placeholder=True)
        modifier_texts = list(occurrence_metadata.get("modifier_texts", []) or [])
        modifier_texts_norm = [
            normalize_text(modifier)
            for modifier in list(occurrence_metadata.get("modifier_texts_norm", []) or [])
            if normalize_text(modifier)
        ]
        modifier_nodes = [
            self._get_or_create_token_node(modifier, is_placeholder=True)
            for modifier in modifier_texts_norm
        ]
        modifier_nodes = [node for node in modifier_nodes if node is not None]

        phrase_token_node.compositional_head = head_node
        phrase_token_node.compositional_head_text = occurrence_metadata.get("head_text")
        phrase_token_node.compositional_head_text_norm = head_text_norm
        phrase_token_node.compositional_modifiers = modifier_nodes
        phrase_token_node.compositional_modifier_texts = modifier_texts
        phrase_token_node.compositional_modifier_texts_norm = modifier_texts_norm

        if head_node is None:
            return
        record_key = normalize_text(phrase_token_node.token_text)
        record = head_node.headed_phrase_records.setdefault(
            record_key,
            {
                "phrase_token_node": phrase_token_node,
                "phrase_text": phrase_token_node.token_text,
                "count": 0,
                "occurrences": [],
            },
        )
        record["phrase_token_node"] = phrase_token_node
        record["phrase_text"] = phrase_token_node.token_text
        record["count"] = int(record.get("count", 0)) + 1
        record.setdefault("occurrences", []).append(
            {
                "chunk_node": chunk_node,
                "chunk_node_id": getattr(chunk_node, "chunk_node_id", None),
                "chunk_text": getattr(chunk_node, "chunk_text", None),
                "surface_text": occurrence_metadata.get("surface_text"),
                "surface_start": occurrence_metadata.get("surface_start"),
                "surface_end": occurrence_metadata.get("surface_end"),
                "head_text": occurrence_metadata.get("head_text"),
                "head_text_norm": head_text_norm,
                "head_start": occurrence_metadata.get("head_start"),
                "head_end": occurrence_metadata.get("head_end"),
                "modifier_texts": modifier_texts,
                "modifier_texts_norm": modifier_texts_norm,
            }
        )

    # Return True when a placeholder node has no indexed evidence and can be removed.
    def _is_empty_placeholder_token_node(self, token_node):
        return (
            bool(getattr(token_node, "is_placeholder", False))
            and not getattr(token_node, "has_semantic", False)
            and not getattr(token_node, "sem_node_list", [])
            and not getattr(token_node, "embeds_buffer", [])
            and not getattr(token_node, "span_occurrences", [])
        )

    # Remove modifier-only placeholder token nodes from global token indexes.
    def _cleanup_empty_placeholder_token_nodes(self):
        removed_nodes = [
            token_node
            for token_node in self.token_nodes
            if self._is_empty_placeholder_token_node(token_node)
        ]
        if not removed_nodes:
            return 0
        removed_ids = {id(token_node) for token_node in removed_nodes}

        self.token_nodes = [
            token_node for token_node in self.token_nodes
            if id(token_node) not in removed_ids
        ]
        self.phrase_token_nodes = [
            token_node for token_node in self.phrase_token_nodes
            if id(token_node) not in removed_ids
        ]
        self.token_node_query = {
            token_text: token_node
            for token_text, token_node in self.token_node_query.items()
            if id(token_node) not in removed_ids
        }

        for token_node in self.token_nodes:
            if id(getattr(token_node, "compositional_head", None)) in removed_ids:
                token_node.compositional_head = None
            token_node.compositional_modifiers = [
                modifier_node
                for modifier_node in getattr(token_node, "compositional_modifiers", []) or []
                if id(modifier_node) not in removed_ids
            ]
            for record in getattr(token_node, "headed_phrase_records", {}).values():
                if id(record.get("phrase_token_node")) in removed_ids:
                    record["phrase_token_node"] = None

        for index, token_node in enumerate(self.token_nodes):
            token_node.token_node_id = index
        self.next_token_node_id = len(self.token_nodes)
        return len(removed_nodes)

    # Route compositional phrase spans to both the full phrase and semantic head for indexing.
    def _prepare_index_phrase_spans(self, chunk_text, phrases, doc=None):
        if not phrases:
            return phrases

        analyzer = self._get_phrase_analyzer()
        if doc is None:
            doc = self.nlp(chunk_text)
        routed_phrases = []

        for phrase, start_char, end_char in phrases:
            analysis = analyzer.analyze(
                phrase,
                chunk_text=chunk_text,
                span_start=start_char,
                span_end=end_char,
                doc=doc,
            )
            surface_text = analysis.phrase_text or chunk_text[start_char:end_char]

            if (
                analysis.phrase_type == PHRASE_TYPE_COMPOSITIONAL
                and analysis.head_text_norm
                and analysis.head_start is not None
                and analysis.head_end is not None
            ):
                phrase_metadata = self._build_occurrence_metadata(
                    phrase_type=PHRASE_TYPE_COMPOSITIONAL,
                    surface_text=surface_text,
                    surface_start=start_char,
                    surface_end=end_char,
                    modifier_texts=analysis.modifier_texts,
                    modifier_texts_norm=analysis.modifier_texts_norm,
                    modifier_spans=analysis.modifier_spans,
                    atomic_modifier_spans=analysis.atomic_modifier_spans,
                )
                phrase_metadata.update(
                    {
                        "compositional_role": "phrase",
                        "head_text": analysis.head_text,
                        "head_text_norm": analysis.head_text_norm,
                        "head_start": analysis.head_start,
                        "head_end": analysis.head_end,
                    }
                )
                routed_phrases.append((phrase, start_char, end_char, phrase_metadata))
                head_metadata = dict(phrase_metadata)
                head_metadata["compositional_role"] = "head"
                routed_phrases.append(
                    (
                        analysis.head_text_norm,
                        analysis.head_start,
                        analysis.head_end,
                        head_metadata,
                    )
                )
                continue

            phrase_type = analysis.phrase_type
            if phrase_type == PHRASE_TYPE_UNKNOWN:
                phrase_type = PHRASE_TYPE_ATOMIC
            metadata = self._build_occurrence_metadata(
                phrase_type=phrase_type,
                surface_text=surface_text,
                surface_start=start_char,
                surface_end=end_char,
            )
            routed_phrases.append((phrase, start_char, end_char, metadata))

        return routed_phrases

    # Wrap an embedding with chunk and optional span metadata.
    def _make_text_embedding(
        self,
        embed,
        chunk_node,
        span_start=None,
        span_end=None,
        occurrence_metadata=None,
    ):
        occurrence_metadata = occurrence_metadata or {}
        span_text = None
        if span_start is not None and span_end is not None:
            span_text = chunk_node.chunk_text[span_start:span_end]
        return TextEmbedding(
            embed=embed,
            chunk_node=chunk_node,
            span_start=span_start,
            span_end=span_end,
            span_text=span_text,
            surface_text=occurrence_metadata.get("surface_text") or span_text,
            surface_start=occurrence_metadata.get("surface_start", span_start),
            surface_end=occurrence_metadata.get("surface_end", span_end),
            phrase_type=occurrence_metadata.get("phrase_type"),
            modifier_text=occurrence_metadata.get("modifier_text"),
            modifier_texts=list(occurrence_metadata.get("modifier_texts", []) or []),
            modifier_texts_norm=list(occurrence_metadata.get("modifier_texts_norm", []) or []),
            modifier_spans=list(occurrence_metadata.get("modifier_spans", []) or []),
            atomic_modifier_spans=list(occurrence_metadata.get("atomic_modifier_spans", []) or []),
        )

    # Record a token occurrence from a text embedding.
    def _append_token_occurrence(self, token_node, text_embedding):
        token_node.span_occurrences.append(text_embedding.to_span_occurrence())

    # Clone a text embedding while preserving its chunk and span metadata.
    def _clone_text_embedding(self, text_embedding):
        return TextEmbedding(
            embed=text_embedding.embed.clone() if torch.is_tensor(text_embedding.embed) else text_embedding.embed,
            chunk_node=text_embedding.chunk_node,
            span_start=text_embedding.span_start,
            span_end=text_embedding.span_end,
            span_text=text_embedding.span_text,
            surface_text=getattr(text_embedding, "surface_text", None),
            surface_start=getattr(text_embedding, "surface_start", None),
            surface_end=getattr(text_embedding, "surface_end", None),
            phrase_type=getattr(text_embedding, "phrase_type", None),
            modifier_text=getattr(text_embedding, "modifier_text", None),
            modifier_texts=list(getattr(text_embedding, "modifier_texts", []) or []),
            modifier_texts_norm=list(getattr(text_embedding, "modifier_texts_norm", []) or []),
            modifier_spans=list(getattr(text_embedding, "modifier_spans", []) or []),
            atomic_modifier_spans=list(getattr(text_embedding, "atomic_modifier_spans", []) or []),
        )

    # Return cloned retained embeddings, optionally downsampled.
    def _sample_text_embeddings(self, text_embeddings, max_samples=None):
        text_embeddings = list(text_embeddings)
        if max_samples is not None and max_samples > 0 and len(text_embeddings) > max_samples:
            text_embeddings = random.sample(text_embeddings, max_samples)
        return [self._clone_text_embedding(text_embedding) for text_embedding in text_embeddings]

    # Initialize the retained embedding reservoir for a semantic node.
    def _initialize_sem_retained_text_embeddings(self, sem_node, text_embeddings):
        text_embeddings = list(text_embeddings)
        sem_node.retained_text_embeddings = self._sample_text_embeddings(
            text_embeddings,
            max_samples=self.sem_retained_embed_limit,
        )
        sem_node.retained_text_embedding_source_count = len(text_embeddings)

    # Update a semantic node retained-embedding reservoir with one embedding.
    def _retain_text_embedding_for_sem(self, sem_node, text_embedding):
        if text_embedding is None or text_embedding.embed is None:
            return
        if not hasattr(sem_node, "retained_text_embeddings") or sem_node.retained_text_embeddings is None:
            sem_node.retained_text_embeddings = []
        if not hasattr(sem_node, "retained_text_embedding_source_count"):
            sem_node.retained_text_embedding_source_count = 0

        sem_node.retained_text_embedding_source_count += 1
        cloned_text_embedding = self._clone_text_embedding(text_embedding)
        if len(sem_node.retained_text_embeddings) < self.sem_retained_embed_limit:
            sem_node.retained_text_embeddings.append(cloned_text_embedding)
            return

        replace_index = random.randint(0, sem_node.retained_text_embedding_source_count - 1)
        if replace_index < self.sem_retained_embed_limit:
            sem_node.retained_text_embeddings[replace_index] = cloned_text_embedding

    # Attach a text occurrence and optional edge weight to a semantic node.
    def _append_sem_occurrence(self, sem_node, text_embedding, edge_weight=None):
        sem_node.append_text_embedding_occurrence(text_embedding, edge_weight=edge_weight)
        self._retain_text_embedding_for_sem(sem_node, text_embedding)

    # Attach a span occurrence to a semantic node and retain its embedding when available.
    def _append_span_occurrence_to_sem(
        self,
        sem_node,
        span_occurrence,
        edge_weight=None,
        retained_text_embedding=None,
    ):
        sem_node.append_span_occurrence(span_occurrence, edge_weight=edge_weight)
        if retained_text_embedding is not None:
            self._retain_text_embedding_for_sem(sem_node, retained_text_embedding)
            if getattr(sem_node, "pending_embed_rebuild", False):
                sem_node.chunk_node_embed.append(retained_text_embedding.embed)

    # Return the initial description assigned to a new semantic node.
    def _get_initial_sem_description(self, token_node):
        return None

    # Create one semantic node from all buffered embeddings for a token node.
    def create_basic_sem_node(self, token_node):
        new_sem_node = self.create_sem_node(token_node)
        new_sem_node.chunk_node_embed = ([i.embed for i in token_node.embeds_buffer])
        new_sem_node.embed = self._store_sem_embed(average_embeds(new_sem_node.chunk_node_embed))
        new_sem_node.chunk_node_list = [k.chunk_node for k in token_node.embeds_buffer]
        new_sem_node.span_occurrences = [k.to_span_occurrence() for k in token_node.embeds_buffer]
        self._initialize_sem_retained_text_embeddings(new_sem_node, token_node.embeds_buffer)
        new_sem_node.chunk_edge_weight = sem_embed_sim(new_sem_node).cpu().tolist()
        new_sem_node.chunk_node_embed.clear()

        token_node.sem_node_list.append(new_sem_node)

    # Build one sem node from a given list of text embeddings and run description assignment.
    def _build_sem_node_from_cluster(self, token_node, text_embeddings):
        self.sem_build_count += 1
        new_sem_node = self.create_sem_node(token_node)
        embeds = [te.embed for te in text_embeddings]
        new_sem_node.embed = self._store_sem_embed(average_embeds(embeds))
        for te in text_embeddings:
            new_sem_node.chunk_node_list.append(te.chunk_node)
            new_sem_node.span_occurrences.append(te.to_span_occurrence())
            new_sem_node.chunk_node_embed.append(te.embed)
        self._initialize_sem_retained_text_embeddings(new_sem_node, text_embeddings)
        new_sem_node.chunk_edge_weight = sem_embed_sim(new_sem_node).cpu().tolist()
        final_sem_nodes = self._assign_sem_description_on_build(
            new_sem_node,
            list(text_embeddings),
        )
        token_node.sem_node_list.extend(final_sem_nodes)

    # Reset the per-session counters that track who reaches the full sem-build path.
    def _reset_sem_build_stats(self):
        # 已构建的语义节点总数(build_sem_node 每产出一个节点 +1)。
        self.sem_build_count = 0
        # 进入 build_sem_node 的 token/span 数(= finalize 时出现次数达到
        # min_occurrences_for_description、足够走完整建节点路径的 token/phrase)。
        self.sem_stats_passed_occurrence = 0
        # 其中被判为 entity / 原子短语(force_single_semantic)而直接单义的,不计入 S_mean 统计。
        self.sem_stats_entity_single = 0
        # 通过出现次数闸门、且非 entity 的 token 中,因 s_mean 过高(语义集中)被判单义而筛掉的数量。
        self.sem_stats_smean_filtered = 0
        # 通过 s_mean 检测、进入聚类/描述消歧路径的数量(多义切分的候选)。
        self.sem_stats_smean_passed = 0
        # 上述候选里最终真正切出多个语义节点的数量。
        self.sem_stats_multi_sense = 0

    # Build semantic nodes from buffered token embeddings.
    def build_sem_node(self, token_node):
        self.sem_stats_passed_occurrence += 1
        if getattr(token_node, "force_single_semantic", False):
            # 被判为 entity / 原子短语:默认单义,不参与 S_mean / 多义统计。
            self.sem_stats_entity_single += 1
            self.create_basic_sem_node(token_node)
        elif token_node.node_type == "token" and not self.semantic_type_cls(token_node):
            # s_mean 过高 → 各次出现语义一致 → 被筛成单义。
            self.sem_stats_smean_filtered += 1
            self.create_basic_sem_node(token_node)
        else:
            # 通过 s_mean(或非 token 的短语),进入描述驱动的多义切分路径。
            self.sem_stats_smean_passed += 1
            sem_count_before = len(token_node.sem_node_list)
            self._build_sem_node_from_cluster(token_node, list(token_node.embeds_buffer))
            if len(token_node.sem_node_list) - sem_count_before > 1:
                self.sem_stats_multi_sense += 1
        token_node.embeds_buffer.clear()

    # Print how many token/span reached / passed each stage of the sem-build funnel.
    def _print_sem_build_stats(self):
        passed = getattr(self, "sem_stats_passed_occurrence", 0)
        entity = getattr(self, "sem_stats_entity_single", 0)
        smean_filtered = getattr(self, "sem_stats_smean_filtered", 0)
        smean_passed = getattr(self, "sem_stats_smean_passed", 0)
        multi_sense = getattr(self, "sem_stats_multi_sense", 0)
        # 参与 S_mean 判定的总数(已剔除 entity / 原子短语)。
        smean_candidates = smean_filtered + smean_passed
        min_occ = getattr(self, "min_occurrences_for_description", None)
        print("=" * 60)
        print(f"sem-build funnel (min_occurrences_for_description={min_occ}):")
        print(
            f"  1. 通过出现次数闸门进入完整建节点路径: {passed} "
            f"(其中 entity/原子短语默认单义: {entity})"
        )
        print(
            f"  2. 参与 S_mean 判定(非 entity): {smean_candidates} "
            f"-> 因 S_mean 过高被筛为单义: {smean_filtered}"
        )
        print(
            f"  3. 通过 S_mean 进入消歧路径: {smean_passed} "
            f"-> 最终切出多个语义: {multi_sense}"
        )
        print("=" * 60)

    # Print an elapsed-time progress message for the current operation.
    def log_time(self, msg):
        print(f"[{time.perf_counter() - self.start_time:.4f}s] {msg}")

    # Start the wall-clock timer for one indexing session.
    def _start_index_session_timer(
        self,
        indexed_document_count,
        start_time=None,
    ):
        start_time = time.perf_counter() if start_time is None else start_time
        self.start_time = start_time
        self.index_session_start_time = start_time
        self.index_session_document_count = indexed_document_count
        self._reset_sem_build_stats()
        self.llm_candidate_filter_total_tokens = 0
        self.llm_candidate_filter_prompt_tokens = 0
        self.llm_candidate_filter_completion_tokens = 0
        self.llm_candidate_filter_total_wall_time = 0.0
        self.llm_candidate_filter_api_wait_wall_time = 0.0
        self.llm_semantic_label_total_tokens = 0
        self.llm_semantic_label_cache_hits = 0
        self.llm_semantic_label_cache_misses = 0
        self.fft_propagation_wall_time = 0.0

    # Build the index-session running-info lines (total/avg time, LLM token usage,
    # wall times). Returns an empty list when no index session has been started.
    # Shared by the console summary (_print_index_session_timing) and the saved
    # sem-description log so both show the same end-of-index report.
    def _format_index_session_timing(self):
        if self.index_session_start_time is None:
            return []

        lines = []
        elapsed = time.perf_counter() - self.index_session_start_time
        lines.append(f"Index + finalize elapsed time: {elapsed:.2f}s")
        lines.append(f"Indexed document count: {self.index_session_document_count}")
        if self.index_session_document_count > 0:
            avg_seconds = elapsed / self.index_session_document_count
            lines.append(f"Average time per indexed document: {avg_seconds:.4f}s")
        if getattr(self, "use_llm_candidate_filter", False):
            lines.append(
                "LLM candidate filter tokens: "
                f"total={self.llm_candidate_filter_total_tokens}, "
                f"prompt={self.llm_candidate_filter_prompt_tokens}, "
                f"completion={self.llm_candidate_filter_completion_tokens}"
            )
            lines.append(
                "LLM candidate filter wall time: "
                f"total={self.llm_candidate_filter_total_wall_time:.2f}s, "
                f"api_wait={self.llm_candidate_filter_api_wait_wall_time:.2f}s"
            )
        if getattr(self, "use_llm_semantic_labeler", False):
            lines.append(
                "LLM semantic labeler tokens: "
                f"total={self.llm_semantic_label_total_tokens}, "
                f"cache_hits={self.llm_semantic_label_cache_hits}, "
                f"cache_misses={self.llm_semantic_label_cache_misses}"
            )
        lines.append(f"FFT propagation wall time: {self.fft_propagation_wall_time:.2f}s")
        return lines

    # Print total index-through-finalize time and average time per indexed document.
    def _print_index_session_timing(self):
        for line in self._format_index_session_timing():
            print(line)

    # Create a throttled console progress updater for long loops.
    def _make_progress_updater(self, label, total, min_interval=0.2, show_remaining=False):
        last_update_time = [0.0]

        # Print progress when enough time has elapsed or a forced update is requested.
        def update(processed, force=False):
            now = time.perf_counter()
            if not force and processed < total and (now - last_update_time[0]) < min_interval:
                return
            last_update_time[0] = now
            # Right-align `remaining` to the width of `total` so a shrinking value
            # never leaves stale trailing digits on terminals that ignore \033[K
            # (e.g. Jupyter output cells, which honor \r but not erase-to-EOL).
            remaining = max(total - processed, 0)
            remaining_text = (
                f" | remaining: {remaining:>{len(str(total))}}" if show_remaining else ""
            )
            print(
                f"\r\033[K{label}: {processed}/{total}{remaining_text}",
                end="",
                flush=True,
            )

        return update

    # Run farthest-first traversal while accumulating total FFT sampling time.
    def _run_farthest_first_traversal(self, *args, **kwargs):
        fft_start = time.perf_counter()
        try:
            return farthest_first_traversal(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - fft_start
            self.fft_propagation_wall_time = (
                getattr(self, "fft_propagation_wall_time", 0.0) + elapsed
            )

    # Compute document-frequency and IDF values for a token node and its semantic nodes.
    def assign_idf(self, token_node):
        N = len(self.chunk_nodes)
        token_chunk_ids = set()
        for sem_node in token_node.sem_node_list:
            sem_chunk_ids = {chunk_node.chunk_node_id for chunk_node in sem_node.chunk_node_list}
            sem_node.df = len(sem_chunk_ids)
            sem_node.idf = math.log((N + 1) / (sem_node.df + 1)) + 1.0
            token_chunk_ids.update(sem_chunk_ids)
        token_node.df = len(token_chunk_ids)
        token_node.idf = math.log((N + 1) / (token_node.df + 1)) + 1.0

    # Decide whether a token node should be split into multiple semantic clusters.
    def semantic_type_cls(self, token_node):
        # 是否对该 token 做多义切分,仅由 embedding 的集中度决定:
        # s_mean 高 → 各次出现语义一致 → 单义节点(返回 False);
        # s_mean 低 → 语义分散 → 走切分路径(返回 True)。
        # 注:此前还有一个基于 df 的"太常见就不切分"闸门,但 df 在索引期
        # 恒为 0(只在 finalize 的 assign_idf 里赋值),该分支从未生效;且即便
        # 生效,它会把"高频多义词"(如 bank)误判为单义,与目标相悖,故已移除。
        s_mean = get_s_mean([i.embed for i in token_node.embeds_buffer])
        # 持久化到 token_node,供 finalize 之后的统计/检视使用(embeds_buffer 随后被清空,
        # 原始全量出现序列不再保留,只能在此刻把这个值存下来)。
        token_node.s_mean = float(s_mean)
        return not (s_mean > self.tau_conc)

    # Look up a token node by its surface text.
    def query_token_node(self, text):
        return self.token_node_query.get(text, None)

    # Populate missing attributes when loading graphs saved by older code versions.
    def _ensure_backward_compatible_attrs(self):
        if not hasattr(self, "index_session_start_time"):
            self.index_session_start_time = None
        if not hasattr(self, "index_session_document_count"):
            self.index_session_document_count = getattr(
                self,
                "index_session_record_count",
                None,
            )
            if self.index_session_document_count is None:
                self.index_session_document_count = getattr(
                    self,
                    "index_session_sample_count",
                    0,
                ) or 0
        if hasattr(self, "index_session_record_count"):
            del self.index_session_record_count
        if hasattr(self, "index_session_sample_count"):
            del self.index_session_sample_count
        if not hasattr(self, "fast_index"):
            self.fast_index = False
        if not hasattr(self, "llm_semantic_labeler_batch_size"):
            self.llm_semantic_labeler_batch_size = 10
        if not hasattr(self, "llm_anchor_sample_assignment_count"):
            self.llm_anchor_sample_assignment_count = 0
        for token_node in getattr(self, "token_nodes", []) or []:
            if not hasattr(token_node, "llm_anchor_sample_assignments"):
                token_node.llm_anchor_sample_assignments = []
        if not getattr(self, "llm_anchor_sample_assignment_count", 0):
            self.llm_anchor_sample_assignment_count = sum(
                len(getattr(token_node, "llm_anchor_sample_assignments", []) or [])
                for token_node in getattr(self, "token_nodes", []) or []
            )
        if getattr(self, "schema_version", None) == CURRENT_SCHEMA_VERSION:
            return
        if not hasattr(self, "phrase_analyzer"):
            self.phrase_analyzer = None
        if not hasattr(self, "modifier_postings"):
            self.modifier_postings = defaultdict(Counter)
        elif not isinstance(self.modifier_postings, defaultdict):
            self.modifier_postings = defaultdict(
                Counter,
                {
                    key: Counter(value)
                    for key, value in self.modifier_postings.items()
                },
            )
        if not hasattr(self, "sem_nodes") and hasattr(self, "proto_nodes"):
            self.sem_nodes = self.proto_nodes
        if not hasattr(self, "next_sem_node_id") and hasattr(self, "next_proto_node_id"):
            self.next_sem_node_id = self.next_proto_node_id
        if not hasattr(self, "sem_retained_embed_limit") and hasattr(self, "proto_retained_embed_limit"):
            self.sem_retained_embed_limit = self.proto_retained_embed_limit
        if not hasattr(self, "sem_description_prompt_context_mode") and hasattr(self, "proto_description_prompt_context_mode"):
            self.sem_description_prompt_context_mode = self.proto_description_prompt_context_mode
        if not hasattr(self, "sem_description_candidate_limit") and hasattr(self, "proto_description_candidate_limit"):
            self.sem_description_candidate_limit = self.proto_description_candidate_limit
        if not hasattr(self, "sem_description_batch_size") and hasattr(self, "proto_description_batch_size"):
            self.sem_description_batch_size = self.proto_description_batch_size
        if not hasattr(self, "sem_description_use_detailed_description") and hasattr(self, "proto_description_use_detailed_description"):
            self.sem_description_use_detailed_description = self.proto_description_use_detailed_description
        if not hasattr(self, "sem_description_require_detailed_description") and hasattr(self, "proto_description_require_detailed_description"):
            self.sem_description_require_detailed_description = self.proto_description_require_detailed_description
        if not hasattr(self, "sem_description_exact_match_text") and hasattr(self, "proto_description_exact_match_text"):
            self.sem_description_exact_match_text = self.proto_description_exact_match_text
        if not hasattr(self, "sem_description_label_contains_text") and hasattr(self, "proto_description_label_contains_text"):
            self.sem_description_label_contains_text = self.proto_description_label_contains_text
        if not hasattr(self, "sem_description_exact_match_first") and hasattr(self, "proto_description_exact_match_first"):
            self.sem_description_exact_match_first = self.proto_description_exact_match_first
        if not hasattr(self, "predicted_sem_description_logs") and hasattr(self, "predicted_proto_description_logs"):
            self.predicted_sem_description_logs = self.predicted_proto_description_logs
        if not hasattr(self, "deleted_merged_sem_logs") and hasattr(self, "deleted_merged_proto_logs"):
            self.deleted_merged_sem_logs = self.deleted_merged_proto_logs
        if not hasattr(self, "sem_description_operation_logs") and hasattr(self, "proto_description_operation_logs"):
            self.sem_description_operation_logs = self.proto_description_operation_logs
        if not hasattr(self, "sem_description_log_path") and hasattr(self, "proto_description_log_path"):
            self.sem_description_log_path = self.proto_description_log_path.replace("proto", "sem")
        if not hasattr(self, "sem_description_model_name") and hasattr(self, "proto_description_model_name"):
            self.sem_description_model_name = self.proto_description_model_name
        if not hasattr(self, "sem_description_model") and hasattr(self, "proto_description_model"):
            self.sem_description_model = self.proto_description_model
        if not hasattr(self, "discard_no_word"):
            self.discard_no_word = False
        if not hasattr(self, "encoder_lock") or self.encoder_lock is None:
            self.encoder_lock = threading.Lock()
        if not hasattr(self, "sem_description_prompt_context_mode"):
            self.sem_description_prompt_context_mode = "sentence_neighbors"
        if not hasattr(self, "sem_description_candidate_limit"):
            self.sem_description_candidate_limit = 5
        if not hasattr(self, "llm_candidate_filter_candidate_limit"):
            self.llm_candidate_filter_candidate_limit = 10
        if not hasattr(self, "use_llm_candidate_filter"):
            self.use_llm_candidate_filter = False
        if not hasattr(self, "llm_candidate_filter_use_api"):
            self.llm_candidate_filter_use_api = True
        if not hasattr(self, "llm_candidate_filter_provider"):
            self.llm_candidate_filter_provider = "openai"
        if not hasattr(self, "llm_candidate_filter_api_model"):
            self.llm_candidate_filter_api_model = None
        if not hasattr(self, "llm_candidate_filter_cache_path"):
            self.llm_candidate_filter_cache_path = "cache/wikidata_definition_filter_cache.sqlite3"
        if not hasattr(self, "llm_candidate_filter_max_tokens"):
            self.llm_candidate_filter_max_tokens = 4096
        if not hasattr(self, "llm_candidate_filter"):
            self.llm_candidate_filter = None
        if not hasattr(self, "llm_candidate_filter_total_tokens"):
            self.llm_candidate_filter_total_tokens = 0
        if not hasattr(self, "llm_candidate_filter_prompt_tokens"):
            self.llm_candidate_filter_prompt_tokens = 0
        if not hasattr(self, "llm_candidate_filter_completion_tokens"):
            self.llm_candidate_filter_completion_tokens = 0
        if not hasattr(self, "llm_candidate_filter_total_wall_time"):
            self.llm_candidate_filter_total_wall_time = 0.0
        if not hasattr(self, "llm_candidate_filter_api_wait_wall_time"):
            self.llm_candidate_filter_api_wait_wall_time = 0.0
        if not hasattr(self, "use_llm_semantic_labeler"):
            self.use_llm_semantic_labeler = False
        if not hasattr(self, "llm_semantic_labeler_provider"):
            self.llm_semantic_labeler_provider = "openai"
        if not hasattr(self, "llm_semantic_labeler_model"):
            self.llm_semantic_labeler_model = "gpt-5.4-mini"
        if not hasattr(self, "llm_semantic_labeler_api_key_file"):
            self.llm_semantic_labeler_api_key_file = "API_KEY"
        if not hasattr(self, "llm_semantic_labeler_cache_path"):
            self.llm_semantic_labeler_cache_path = "cache/llm_semantic_label_cache.sqlite3"
        if not hasattr(self, "llm_semantic_labeler_context_word_window"):
            self.llm_semantic_labeler_context_word_window = 20
        if not hasattr(self, "llm_semantic_labeler_max_tokens"):
            self.llm_semantic_labeler_max_tokens = 256
        if not hasattr(self, "llm_semantic_labeler_batch_size"):
            self.llm_semantic_labeler_batch_size = 10
        if not hasattr(self, "llm_semantic_label_client"):
            self.llm_semantic_label_client = None
        if not hasattr(self, "llm_semantic_label_total_tokens"):
            self.llm_semantic_label_total_tokens = 0
        if not hasattr(self, "llm_semantic_label_cache_hits"):
            self.llm_semantic_label_cache_hits = 0
        if not hasattr(self, "llm_semantic_label_cache_misses"):
            self.llm_semantic_label_cache_misses = 0
        if not hasattr(self, "fft_propagation_wall_time"):
            self.fft_propagation_wall_time = 0.0
        if not hasattr(self, "fft_max_samples"):
            self.fft_max_samples = 10
        if not hasattr(self, "fft_d1_d2_ratio_threshold"):
            self.fft_d1_d2_ratio_threshold = 0.8
        if not hasattr(self, "fft_knn_check_k"):
            self.fft_knn_check_k = 5
        if not hasattr(self, "fft_danger_neighbor_m"):
            self.fft_danger_neighbor_m = 10
        if not hasattr(self, "anchor_prop_params"):
            self.anchor_prop_params = dict(ANCHOR_PROP_DEFAULTS)
        else:
            for key, value in ANCHOR_PROP_DEFAULTS.items():
                self.anchor_prop_params.setdefault(key, value)
        if not hasattr(self, "sem_assignment_method"):
            # 旧 pickle 没有该字段:保持其历史行为(FFT 路径),不强行切到 anchor。
            self._set_sem_assignment_method("FFT-CE")
        elif not hasattr(self, "_anchor_method_spec"):
            self._set_sem_assignment_method(self.sem_assignment_method)
        if not hasattr(self, "sem_description_batch_size"):
            self.sem_description_batch_size = 32
        if not hasattr(self, "sem_description_use_detailed_description"):
            self.sem_description_use_detailed_description = False
        if not hasattr(self, "sem_description_require_detailed_description"):
            self.sem_description_require_detailed_description = True
        if not hasattr(self, "sem_description_exact_match_text"):
            self.sem_description_exact_match_text = False
        if not hasattr(self, "sem_description_label_contains_text"):
            self.sem_description_label_contains_text = True
        if not hasattr(self, "sem_description_exact_match_first"):
            self.sem_description_exact_match_first = False
        if not hasattr(self, "predicted_sem_description_logs"):
            self.predicted_sem_description_logs = []
        if not hasattr(self, "deleted_merged_sem_logs"):
            self.deleted_merged_sem_logs = []
        if not hasattr(self, "sem_description_operation_logs"):
            self.sem_description_operation_logs = []
        if not hasattr(self, "sem_description_log_path"):
            self.sem_description_log_path = "sem_description_logs.txt"
        if not hasattr(self, "wikidata_no_result_logs"):
            self.wikidata_no_result_logs = []
        if not hasattr(self, "_wikidata_no_result_keys"):
            self._wikidata_no_result_keys = set()
        if not hasattr(self, "_sem_description_candidate_bank_cache"):
            self._sem_description_candidate_bank_cache = {}
        if not hasattr(self, "_combined_merge_sample_sense_descriptions"):
            self._combined_merge_sample_sense_descriptions = {}
        if not hasattr(self, "llm_candidate_filter_result_cache"):
            self.llm_candidate_filter_result_cache = {}
        if not hasattr(self, "llm_anchor_sample_assignment_count"):
            self.llm_anchor_sample_assignment_count = 0
        if not hasattr(self, "phrase_audit_enabled"):
            self.phrase_audit_enabled = False
        if not hasattr(self, "phrase_audit_cache_path"):
            self.phrase_audit_cache_path = "cache/phrase_protection_audit.jsonl"
        if not hasattr(self, "_phrase_audit_session_id"):
            self._phrase_audit_session_id = None
        if not hasattr(self, "sem_description_model_name"):
            self.sem_description_model_name = "cross-encoder/nli-deberta-v3-large"
        if not hasattr(self, "sem_description_model"):
            self.sem_description_model = None
        if not hasattr(self, "disambiguate_query_sense"):
            self.disambiguate_query_sense = True
        if not hasattr(self, "consensus_ratio_threshold"):
            self.consensus_ratio_threshold = 0.8
        if not hasattr(self, "min_occurrences_for_description"):
            self.min_occurrences_for_description = 20
        if not hasattr(self, "min_description_candidates"):
            legacy_min_candidates = getattr(self, "single_cluster_min_description_candidates", 3)
            self.min_description_candidates = max(1, int(legacy_min_candidates))
        if not hasattr(self, "sem_build_count"):
            self.sem_build_count = getattr(self, "single_cluster_build_count", 0)
        for token_node in getattr(self, "token_nodes", []):
            self._ensure_token_node_backward_compatible_attrs(token_node)
            for text_embedding in getattr(token_node, "embeds_buffer", []):
                self._ensure_occurrence_metadata_attrs(text_embedding)
            for span_occurrence in getattr(token_node, "span_occurrences", []):
                self._ensure_occurrence_metadata_attrs(span_occurrence)
        for chunk_node in getattr(self, "chunk_nodes", []):
            if not hasattr(chunk_node, "sem_node_list") and hasattr(chunk_node, "proto_node_list"):
                chunk_node.sem_node_list = chunk_node.proto_node_list
            # schema 7: precomputed sentence boundaries; absent on older pickles.
            if not hasattr(chunk_node, "sentence_boundaries"):
                chunk_node.sentence_boundaries = None
        for sem_node in getattr(self, "sem_nodes", []):
            self._ensure_sem_node_backward_compatible_attrs(sem_node)
            for text_embedding in sem_node.retained_text_embeddings:
                self._ensure_occurrence_metadata_attrs(text_embedding)
            for span_occurrence in getattr(sem_node, "span_occurrences", []):
                self._ensure_occurrence_metadata_attrs(span_occurrence)
        self.schema_version = CURRENT_SCHEMA_VERSION

    # Populate occurrence metadata fields introduced for modifier-aware indexing.
    def _ensure_occurrence_metadata_attrs(self, occurrence):
        if not hasattr(occurrence, "span_start"):
            occurrence.span_start = None
        if not hasattr(occurrence, "span_end"):
            occurrence.span_end = None
        if not hasattr(occurrence, "span_text"):
            occurrence.span_text = None
        if not hasattr(occurrence, "surface_text"):
            occurrence.surface_text = getattr(occurrence, "span_text", None)
        if not hasattr(occurrence, "surface_start"):
            occurrence.surface_start = getattr(occurrence, "span_start", None)
        if not hasattr(occurrence, "surface_end"):
            occurrence.surface_end = getattr(occurrence, "span_end", None)
        if not hasattr(occurrence, "phrase_type"):
            occurrence.phrase_type = None
        if not hasattr(occurrence, "modifier_text"):
            occurrence.modifier_text = None
        if not hasattr(occurrence, "modifier_texts"):
            occurrence.modifier_texts = []
        if not hasattr(occurrence, "modifier_texts_norm"):
            occurrence.modifier_texts_norm = []
        if not hasattr(occurrence, "modifier_spans"):
            occurrence.modifier_spans = []
        if not hasattr(occurrence, "atomic_modifier_spans"):
            occurrence.atomic_modifier_spans = []
        # schema 7: index-time sentence index; None on older pickles triggers the
        # query-time char-offset fallback in _nodes_share_sentence.
        if not hasattr(occurrence, "sentence_id"):
            occurrence.sentence_id = None

    # Populate missing token fields introduced after older saved graph versions.
    def _ensure_token_node_backward_compatible_attrs(self, token_node):
        if not hasattr(token_node, "sem_node_list") and hasattr(token_node, "proto_node_list"):
            token_node.sem_node_list = token_node.proto_node_list
        if not hasattr(token_node, "has_semantic") and hasattr(token_node, "has_prototype"):
            token_node.has_semantic = token_node.has_prototype
        if not hasattr(token_node, "is_multi_semantic"):
            token_node.is_multi_semantic = None
        if not hasattr(token_node, "descriptions"):
            token_node.descriptions = None
        if not hasattr(token_node, "wikidata_info_loaded"):
            token_node.wikidata_info_loaded = None
        if not hasattr(token_node, "span_occurrences"):
            token_node.span_occurrences = []
        if not hasattr(token_node, "is_placeholder"):
            token_node.is_placeholder = False
        if not hasattr(token_node, "force_single_semantic"):
            token_node.force_single_semantic = False
        if not hasattr(token_node, "compositional_head"):
            token_node.compositional_head = None
        if not hasattr(token_node, "compositional_head_text"):
            token_node.compositional_head_text = None
        if not hasattr(token_node, "compositional_head_text_norm"):
            token_node.compositional_head_text_norm = None
        if not hasattr(token_node, "compositional_modifiers"):
            token_node.compositional_modifiers = []
        if not hasattr(token_node, "compositional_modifier_texts"):
            token_node.compositional_modifier_texts = []
        if not hasattr(token_node, "compositional_modifier_texts_norm"):
            token_node.compositional_modifier_texts_norm = []
        if not hasattr(token_node, "headed_phrase_records"):
            token_node.headed_phrase_records = {}
        # schema 6:finalize 时持久化的 embedding 集中度;旧 pickle 没有,记为 None。
        if not hasattr(token_node, "s_mean"):
            token_node.s_mean = None
        # schema 8:Anchor LLM 样本语义分配记录;旧 pickle 没有,记为空列表。
        if not hasattr(token_node, "llm_anchor_sample_assignments"):
            token_node.llm_anchor_sample_assignments = []
        # 索引期 anomaly 机制已移除:丢弃旧 pickle 残留的 anomaly_section 属性。
        if hasattr(token_node, "anomaly_section"):
            del token_node.anomaly_section

    # Populate missing semantic-node fields introduced after older saved graph versions.
    def _ensure_sem_node_backward_compatible_attrs(self, sem_node):
        if not hasattr(sem_node, "sem_node_id") and hasattr(sem_node, "proto_node_id"):
            sem_node.sem_node_id = sem_node.proto_node_id
        # anomaly 阈值路由已移除:丢弃旧 pickle 残留的 anomaly_threshold 属性。
        if hasattr(sem_node, "anomaly_threshold"):
            del sem_node.anomaly_threshold
        if not hasattr(sem_node, "span_occurrences"):
            sem_node.span_occurrences = []
        if not hasattr(sem_node, "retained_text_embeddings") or sem_node.retained_text_embeddings is None:
            sem_node.retained_text_embeddings = []
        if not hasattr(sem_node, "retained_text_embedding_source_count"):
            sem_node.retained_text_embedding_source_count = len(sem_node.retained_text_embeddings)
        if not hasattr(sem_node, "pending_embed_rebuild"):
            sem_node.pending_embed_rebuild = False

    # Run span extraction with optional cleanup and debug output.
    def debug_extract_important_spans(
        self,
        chunk,
        min_tokens=2,
        discard_no_word=None,
        debug_mode=True,
        clean_input=False,
    ):
        self._ensure_backward_compatible_attrs()

        if self.nlp is None:
            self._load_nlp()

        if discard_no_word is None:
            discard_no_word = getattr(self, "discard_no_word", False)

        if clean_input:
            chunk = clean_text(chunk)

        return extract_important_spans(
            chunk,
            self.nlp,
            min_tokens=min_tokens,
            discard_no_word=discard_no_word,
            debug_mode=debug_mode,
        )

    # Build the normalized semantic-node embedding matrix used for similarity search.
    def build_query_database(self):
        if not self.sem_nodes:
            self.query_database = None
            raise ValueError("Cannot build query database without semantic nodes.")
        missing_embed_ids = [
            sem_node.sem_node_id
            for sem_node in self.sem_nodes
            if sem_node.embed is None
        ]
        if missing_embed_ids:
            raise ValueError(
                "Cannot build query database because semantic nodes are missing embeddings: "
                f"{missing_embed_ids}"
            )
        embeds_list = [sem_node.embed for sem_node in self.sem_nodes]
        database = torch.stack(embeds_list).to(self.device)
        self.query_database = F.normalize(database, dim=1)

    # Rebuild modifier-head postings from final semantic-node span occurrences.
    def build_modifier_postings(self):
        self.modifier_postings = defaultdict(Counter)
        for sem_node in self.sem_nodes:
            head = normalize_text(getattr(sem_node.token_node, "token_text", ""))
            if not head:
                continue
            for occurrence in getattr(sem_node, "span_occurrences", []) or []:
                for modifier in self._get_posting_modifier_units_norm(occurrence):
                    if modifier:
                        self.modifier_postings[(head, modifier)][sem_node.sem_node_id] += 1
        return self.modifier_postings

    # Return modifier postings as plain dictionaries for inspection or export.
    def get_modifier_postings(self):
        return {
            key: dict(value)
            for key, value in self.modifier_postings.items()
        }

    # Return modifier units for a compositional phrase analysis.
    def _get_query_modifier_units(self, analysis):
        modifier_records = self._get_query_modifier_unit_records(analysis)
        return (
            [record["text"] for record in modifier_records],
            [record["text_norm"] for record in modifier_records],
        )

    # Return modifier units with offsets so each modifier can become a query group member.
    def _get_query_modifier_unit_records(self, analysis):
        atomic_spans = list(getattr(analysis, "atomic_modifier_spans", []) or [])
        modifier_spans = list(getattr(analysis, "modifier_spans", []) or [])
        modifier_texts = list(getattr(analysis, "modifier_texts", []) or [])
        modifier_texts_norm = list(getattr(analysis, "modifier_texts_norm", []) or [])

        atomic_ranges = []
        modifier_records = []

        for atomic_span in atomic_spans:
            text = str(atomic_span.get("text", "")).strip()
            text_norm = normalize_text(atomic_span.get("text_norm") or text)
            if not text_norm:
                continue
            start = atomic_span.get("start")
            end = atomic_span.get("end")
            if start is not None and end is not None:
                atomic_ranges.append((start, end))
            modifier_records.append(
                {
                    "text": text or text_norm,
                    "text_norm": text_norm,
                    "start": start,
                    "end": end,
                }
            )

        for idx, text_norm in enumerate(modifier_texts_norm):
            text_norm = normalize_text(text_norm)
            if not text_norm:
                continue
            span = modifier_spans[idx] if idx < len(modifier_spans) else None
            if span is not None:
                start, end = span
                if any(start >= atomic_start and end <= atomic_end for atomic_start, atomic_end in atomic_ranges):
                    continue
            else:
                start = None
                end = None
            text = modifier_texts[idx] if idx < len(modifier_texts) else text_norm
            modifier_records.append(
                {
                    "text": text,
                    "text_norm": text_norm,
                    "start": start,
                    "end": end,
                }
            )

        deduped_records = []
        seen = set()
        for record in modifier_records:
            text_norm = record["text_norm"]
            if text_norm in seen:
                continue
            seen.add(text_norm)
            deduped_records.append(record)
        return deduped_records

    # Return True when a token span is fully covered by any query phrase span.
    def _query_span_is_covered(self, start_char, end_char, covered_spans):
        return any(start_char >= span_start and end_char <= span_end for span_start, span_end in covered_spans)

    # Build a query-unit record used by exact, fuzzy, and similarity matching.
    def _make_query_unit(
        self,
        *,
        token_type,
        search_key,
        embedding_span_text,
        embedding_start,
        embedding_end,
        query_surface=None,
        surface_start=None,
        surface_end=None,
        phrase_type=None,
        query_modifiers=None,
        query_modifiers_norm=None,
        query_group_id=None,
        query_group_role=None,
        query_member_key=None,
        query_group_member_keys=None,
        query_role_weight=None,
    ):
        query_surface = query_surface or search_key
        return {
            "token_type": token_type,
            "token": search_key,
            "search_key": search_key,
            "embedding_span_text": embedding_span_text,
            "embedding_start": embedding_start,
            "embedding_end": embedding_end,
            "query_surface": query_surface,
            "query_surface_norm": normalize_text(query_surface),
            "surface_start": surface_start if surface_start is not None else embedding_start,
            "surface_end": surface_end if surface_end is not None else embedding_end,
            "phrase_type": phrase_type,
            "query_modifiers": list(query_modifiers or []),
            "query_modifiers_norm": [
                normalize_text(modifier)
                for modifier in list(query_modifiers_norm or [])
                if normalize_text(modifier)
            ],
            "query_group_id": query_group_id,
            "query_group_role": query_group_role,
            "query_member_key": normalize_text(query_member_key or search_key),
            "query_group_member_keys": [
                normalize_text(member_key)
                for member_key in list(query_group_member_keys or [])
                if normalize_text(member_key)
            ],
            "query_role_weight": query_role_weight,
        }

    # True when a query phrase/entity cannot be matched as a whole: neither an exact
    # phrase/token node nor a conjunctive word-intersection (fuzzy) candidate exists.
    # In that case its discriminative sub-tokens were swallowed by the entity span and
    # are missing from extract_important_tokens(), so the whole-phrase channel is empty.
    def _phrase_whole_match_misses(self, search_key):
        if self.query_token_node(search_key) is not None:
            return False
        return not self.phrase_word_intersection_query(search_key)

    # Sub-token fallback for a whole-phrase miss: strip leading articles, then emit one
    # single-token query unit per span token that (a) passes the index-time token filter
    # and (b) has an exact token node in the index. This recovers discriminative tokens
    # (e.g. "discovery" inside "star trek: discovery", "DJ"/"single" inside "a DJ fresh
    # single") that NER greedily folded into an unindexed entity span. Spans already
    # emitted are tracked in emitted_token_spans to dedup against the standalone-token loop.
    def _build_subtoken_fallback_units(self, query_text, doc, start_char, end_char, emitted_token_spans):
        span_tokens = [
            token for token in doc
            if token.idx >= start_char and token.idx + len(token.text) <= end_char
        ]
        while span_tokens and span_tokens[0].lower_ in {"a", "an", "the"}:
            span_tokens = span_tokens[1:]

        units = []
        for token in span_tokens:
            if not _is_valid_token(token):
                continue
            search_key = _normalize_token_text(token)
            if not search_key or self.query_token_node(search_key) is None:
                continue
            t_start = token.idx
            t_end = token.idx + len(token.text)
            if (t_start, t_end) in emitted_token_spans:
                continue
            emitted_token_spans.add((t_start, t_end))
            units.append(
                self._make_query_unit(
                    token_type="token",
                    search_key=search_key,
                    embedding_span_text=search_key,
                    embedding_start=t_start,
                    embedding_end=t_end,
                    query_surface=query_text[t_start:t_end],
                    surface_start=t_start,
                    surface_end=t_end,
                    phrase_type=PHRASE_TYPE_SINGLE_TOKEN,
                )
            )
        return units

    # Clean a query and extract token, phrase, and entity embeddings for matching.
    def _prepare_query_tokens(self, query_text, print_important_tokens=False, expand_compositional=False):
        query_text = clean_text(query_text)
        token_embeddings, offsets = encode_text(query_text, self.text_encoder, self.tokenizer, self.device)
        important_phrases, num_ents = extract_important_phrases(query_text, self.nlp)
        important_tokens = extract_important_tokens(query_text, self.nlp)
        tokens_for_processing = []
        covered_compositional_spans = []
        # Char spans emitted by the sub-token fallback below; consulted by the
        # standalone-token loop so a recovered sub-token is not also emitted twice.
        emitted_token_spans = set()
        analyzer = self._get_phrase_analyzer()
        doc = self.nlp(query_text)

        phrase_items = (
            [("ent", phrase, start_char, end_char) for phrase, start_char, end_char in important_phrases[:num_ents]] +
            [("phrase", phrase, start_char, end_char) for phrase, start_char, end_char in important_phrases[num_ents:]]
        )
        for token_type, phrase, start_char, end_char in phrase_items:
            analysis = analyzer.analyze(
                phrase,
                chunk_text=query_text,
                span_start=start_char,
                span_end=end_char,
                doc=doc,
            )
            query_surface = analysis.phrase_text or query_text[start_char:end_char]
            if (
                analysis.phrase_type == PHRASE_TYPE_COMPOSITIONAL
                and analysis.head_text_norm
                and analysis.head_start is not None
                and analysis.head_end is not None
            ):
                modifier_records = self._get_query_modifier_unit_records(analysis)
                modifier_units = [record["text"] for record in modifier_records]
                modifier_units_norm = [record["text_norm"] for record in modifier_records]
                if expand_compositional:
                    group_id = f"{start_char}:{end_char}:{analysis.phrase_text_norm}"
                    group_member_keys = [analysis.head_text_norm] + modifier_units_norm
                    tokens_for_processing.append(
                        self._make_query_unit(
                            token_type=token_type,
                            search_key=analysis.phrase_text_norm,
                            embedding_span_text=analysis.phrase_text_norm,
                            embedding_start=start_char,
                            embedding_end=end_char,
                            query_surface=query_surface,
                            surface_start=start_char,
                            surface_end=end_char,
                            phrase_type=PHRASE_TYPE_COMPOSITIONAL,
                            query_modifiers=modifier_units,
                            query_modifiers_norm=modifier_units_norm,
                            query_group_id=group_id,
                            query_group_role="phrase",
                            query_member_key=analysis.phrase_text_norm,
                            query_group_member_keys=group_member_keys,
                            query_role_weight=COMPOSITIONAL_QUERY_ROLE_WEIGHT["phrase"],
                        )
                    )
                    tokens_for_processing.append(
                        self._make_query_unit(
                            token_type="token",
                            search_key=analysis.head_text_norm,
                            embedding_span_text=analysis.head_text_norm,
                            embedding_start=analysis.head_start,
                            embedding_end=analysis.head_end,
                            query_surface=analysis.head_text,
                            surface_start=analysis.head_start,
                            surface_end=analysis.head_end,
                            phrase_type=PHRASE_TYPE_SINGLE_TOKEN,
                            query_group_id=group_id,
                            query_group_role="head",
                            query_member_key=analysis.head_text_norm,
                            query_group_member_keys=group_member_keys,
                            query_role_weight=COMPOSITIONAL_QUERY_ROLE_WEIGHT["head"],
                        )
                    )
                    for modifier_record in modifier_records:
                        if modifier_record["start"] is None or modifier_record["end"] is None:
                            continue
                        modifier_token_type = (
                            "phrase" if count_words(modifier_record["text_norm"]) >= 2 else "token"
                        )
                        modifier_phrase_type = (
                            PHRASE_TYPE_ATOMIC
                            if modifier_token_type == "phrase"
                            else PHRASE_TYPE_SINGLE_TOKEN
                        )
                        tokens_for_processing.append(
                            self._make_query_unit(
                                token_type=modifier_token_type,
                                search_key=modifier_record["text_norm"],
                                embedding_span_text=modifier_record["text_norm"],
                                embedding_start=modifier_record["start"],
                                embedding_end=modifier_record["end"],
                                query_surface=modifier_record["text"],
                                surface_start=modifier_record["start"],
                                surface_end=modifier_record["end"],
                                phrase_type=modifier_phrase_type,
                                query_group_id=group_id,
                                query_group_role="modifier",
                                query_member_key=modifier_record["text_norm"],
                                query_group_member_keys=group_member_keys,
                                query_role_weight=COMPOSITIONAL_QUERY_ROLE_WEIGHT["modifier"],
                            )
                        )
                    covered_compositional_spans.append((start_char, end_char))
                    continue

                tokens_for_processing.append(
                    self._make_query_unit(
                        token_type=token_type,
                        search_key=analysis.head_text_norm,
                        embedding_span_text=analysis.head_text_norm,
                        embedding_start=analysis.head_start,
                        embedding_end=analysis.head_end,
                        query_surface=query_surface,
                        surface_start=start_char,
                        surface_end=end_char,
                        phrase_type=PHRASE_TYPE_COMPOSITIONAL,
                        query_modifiers=modifier_units,
                        query_modifiers_norm=modifier_units_norm,
                    )
                )
                covered_compositional_spans.append((start_char, end_char))
                continue

            phrase_type = analysis.phrase_type
            if phrase_type == PHRASE_TYPE_UNKNOWN:
                phrase_type = PHRASE_TYPE_ATOMIC
            tokens_for_processing.append(
                self._make_query_unit(
                    token_type=token_type,
                    search_key=phrase,
                    embedding_span_text=phrase,
                    embedding_start=start_char,
                    embedding_end=end_char,
                    query_surface=query_surface,
                    surface_start=start_char,
                    surface_end=end_char,
                    phrase_type=phrase_type,
                )
            )
            # Sub-token fallback: if this atomic phrase/entity has no whole-phrase match
            # in the index, recover its discriminative sub-tokens as standalone units.
            if self._phrase_whole_match_misses(phrase):
                tokens_for_processing.extend(
                    self._build_subtoken_fallback_units(
                        query_text, doc, start_char, end_char, emitted_token_spans
                    )
                )

        # Order contract: phrase/entity units (built above) MUST be appended before these
        # single-token units. _resolve_query_matches() dedups a phrase's sub-tokens by scanning
        # tokens_in_phrase as it walks this list in order, so a phrase has to be seen before its
        # sub-tokens for the dedup to fire. Do not reorder these two loops.
        for token, start_char, end_char in important_tokens:
            if self._query_span_is_covered(start_char, end_char, covered_compositional_spans):
                continue
            if (start_char, end_char) in emitted_token_spans:
                continue
            tokens_for_processing.append(
                self._make_query_unit(
                    token_type="token",
                    search_key=token,
                    embedding_span_text=token,
                    embedding_start=start_char,
                    embedding_end=end_char,
                    query_surface=query_text[start_char:end_char],
                    surface_start=start_char,
                    surface_end=end_char,
                    phrase_type=PHRASE_TYPE_SINGLE_TOKEN,
                )
            )

        if print_important_tokens:
            print(f"important ents: {[text for text, _, _ in important_phrases[:num_ents]]}")
            print(f"important phrases: {[text for text, _, _ in important_phrases[num_ents:]]}")
            print(f"important tokens: {[text for text, _, _ in important_tokens]}")

        return query_text, token_embeddings, offsets, tokens_for_processing

    # Disambiguate a multi-sense query span by reusing the index-time semantic
    # judgment instead of embedding similarity.
    #
    # If the matched token built several described sem nodes during indexing (i.e.
    # it is genuinely multi-sense), score the query context against those senses'
    # candidate definitions with the same cross-encoder / LLM judge used at index
    # time, and return the sem node whose description wins. Returns None to signal
    # the caller should fall back to embedding-similarity matching (single sense,
    # missing descriptions, no model, or an inconclusive judgment).
    def _disambiguate_query_sem_node_by_description(self, token_node, query_text, token_text, match_span):
        if not getattr(self, "disambiguate_query_sense", False):
            return None
        sem_nodes = getattr(token_node, "sem_node_list", []) or []
        if len(sem_nodes) < 2:
            return None

        # Map each described sense to its sem node. Duplicate descriptions are
        # merged at finalize, so collisions should not occur; keep the first.
        description_to_sem_node = {}
        for sem_node in sem_nodes:
            description = (getattr(sem_node, "description", None) or "").strip()
            if description:
                description_to_sem_node.setdefault(description, sem_node)
        if len(description_to_sem_node) < 2:
            return None

        use_llm = bool(getattr(self, "use_llm_semantic_labeler", False))
        if not use_llm and self.sem_description_model is None:
            return None

        candidate_bank = self._load_sem_description_candidate_bank(
            sem_nodes[0], self._sem_description_candidate_bank_cache
        )
        if not candidate_bank:
            return None
        # Restrict the judgment to the senses that actually became sem nodes so the
        # winning description always maps back to a node.
        filtered_bank = [
            candidate
            for candidate in candidate_bank
            if (candidate.get("description") or "").strip() in description_to_sem_node
        ]
        if len({(candidate.get("description") or "").strip() for candidate in filtered_bank}) < 2:
            return None

        prompt_info = self._build_sem_node_description_prompt(
            query_text, token_text, match_span=match_span
        )
        if prompt_info is None:
            prompt_info = self._build_fallback_sem_description_prompt(query_text, token_text)
        if prompt_info is None:
            return None

        if use_llm:
            top_candidate = self._predict_description_from_prompt_with_llm(
                token_text, prompt_info, filtered_bank
            )
        else:
            pairs = [
                (prompt_info["prompt_text"], candidate["hypothesis"])
                for candidate in filtered_bank
            ]
            raw_scores = self.sem_description_model.predict(
                pairs,
                batch_size=min(self.sem_description_batch_size, len(pairs)),
                show_progress_bar=False,
            )
            scores = extract_cross_encoder_scores(raw_scores, self.sem_description_model)
            ranked = sorted(
                zip(filtered_bank, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            top_candidate = ranked[0][0] if ranked else None

        if not top_candidate:
            return None
        description = (top_candidate.get("description") or "").strip()
        return description_to_sem_node.get(description)

    # Resolve exact, fuzzy, and embedding-based semantic node matches for a query.
    def _resolve_query_matches(
        self,
        query_text,
        search_mode="broad",
        print_important_tokens=False,
        expand_compositional=False,
    ):
        query_text, token_embeddings, offsets, tokens_for_processing = self._prepare_query_tokens(
            query_text,
            print_important_tokens=print_important_tokens,
            expand_compositional=expand_compositional,
        )
        query_tokens = []
        # The single live phrase/sub-token dedup. tokens_in_phrase is filled when a phrase/entity
        # is matched (below) and is consulted to skip the phrase's standalone sub-tokens. This
        # relies on _prepare_query_tokens emitting phrase units before their sub-token units.
        # Note: only exact (token_node present) phrases register here, so a purely fuzzy phrase
        # keeps its sub-tokens on purpose (complementary recall) -- see _collect_query_cooccurrence_sems.
        tokens_in_phrase = []
        resolved_matches = []

        for query_unit in tokens_for_processing:
            token_type = query_unit["token_type"]
            token = query_unit["search_key"]
            if (
                query_unit["query_group_id"] is None
                and token_type == "token"
                and token in tokens_in_phrase
            ):
                continue

            token_embed = get_embed_by_offest(
                token_embeddings,
                offsets,
                (
                    query_unit["embedding_span_text"],
                    query_unit["embedding_start"],
                    query_unit["embedding_end"],
                ),
            )
            token_node = self.query_token_node(token)
            exact_sem_node = None
            fuzzy_sem_nodes = []
            retrieved_sem = None
            semantic_weight = None

            if token_node is not None:
                surface_start = query_unit.get("surface_start")
                surface_end = query_unit.get("surface_end")
                match_span = (
                    (surface_start, surface_end)
                    if surface_start is not None and surface_end is not None
                    else None
                )
                exact_sem_node = self._disambiguate_query_sem_node_by_description(
                    token_node,
                    query_text,
                    token,
                    match_span,
                )
                if exact_sem_node is None:
                    max_sem_index = self.get_max_sim_sem(token_node, token_embed)
                    if max_sem_index is not None:
                        exact_sem_node = token_node.sem_node_list[max_sem_index]
                if token_type in {"phrase", "ent"} and search_mode != "broad":
                    tokens_in_phrase.extend(token.split(" "))

            if query_unit["phrase_type"] == PHRASE_TYPE_COMPOSITIONAL and query_unit["query_group_id"] is None:
                fuzzy_query_list = []
            else:
                fuzzy_query_list = self.phrase_word_intersection_query(token)
            if fuzzy_query_list:
                fuzzy_sem_nodes = self.max_cosine_sem_nodes(token_embed, fuzzy_query_list, k=2)

            if exact_sem_node is None and not fuzzy_sem_nodes:
                sem_node_indices, weights = self.query_by_sim([token_embed])
                retrieved_sem = self.sem_nodes[sem_node_indices[0]]
                semantic_weight = weights[0]

            if exact_sem_node is not None and token_type in {"phrase", "ent"} and search_mode == "broad":
                tokens_in_phrase.extend(token.split(" "))

            query_tokens.append(token)
            resolved_matches.append({
                "token_type": token_type,
                "token": token,
                "query_unit": query_unit,
                "query_surface": query_unit["query_surface"],
                "query_surface_norm": query_unit["query_surface_norm"],
                "phrase_type": query_unit["phrase_type"],
                "query_modifiers": list(query_unit["query_modifiers"]),
                "query_modifiers_norm": list(query_unit["query_modifiers_norm"]),
                "query_group_id": query_unit["query_group_id"],
                "query_group_role": query_unit["query_group_role"],
                "query_member_key": query_unit["query_member_key"],
                "query_group_member_keys": list(query_unit["query_group_member_keys"]),
                "query_role_weight": query_unit["query_role_weight"],
                "token_embed": token_embed,
                "exact_sem_node": exact_sem_node,
                "fuzzy_sem_nodes": fuzzy_sem_nodes,
                "retrieved_sem": retrieved_sem,
                "semantic_weight": semantic_weight,
            })

        return query_text, query_tokens, resolved_matches

    # Return the semantic-node candidates associated with one resolved query match.
    def _get_match_sem_nodes(self, match):
        sem_nodes = []
        if match.get("exact_sem_node") is not None:
            sem_nodes.append(match["exact_sem_node"])
        sem_nodes.extend(match.get("fuzzy_sem_nodes") or [])
        if match.get("retrieved_sem") is not None:
            sem_nodes.append(match["retrieved_sem"])

        deduped = []
        seen = set()
        for sem_node in sem_nodes:
            key = id(sem_node)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(sem_node)
        return deduped

    # Normalize all modifier strings stored on an occurrence.
    def _get_occurrence_modifier_units_norm(self, occurrence):
        modifier_units = []
        modifier_units.extend(getattr(occurrence, "modifier_texts_norm", []) or [])
        modifier_text = getattr(occurrence, "modifier_text", None)
        if modifier_text:
            modifier_units.append(modifier_text)
        for atomic_span in getattr(occurrence, "atomic_modifier_spans", []) or []:
            if isinstance(atomic_span, dict):
                modifier_units.append(atomic_span.get("text_norm") or atomic_span.get("text") or "")

        deduped = []
        seen = set()
        for modifier in modifier_units:
            modifier_norm = normalize_text(str(modifier).strip())
            if not modifier_norm or modifier_norm in seen:
                continue
            seen.add(modifier_norm)
            deduped.append(modifier_norm)
        return deduped

    # Normalize modifier strings used as modifier-posting keys.
    def _get_posting_modifier_units_norm(self, occurrence):
        modifier_units = list(getattr(occurrence, "modifier_texts_norm", []) or [])
        if not modifier_units and getattr(occurrence, "modifier_text", None):
            modifier_units.append(occurrence.modifier_text)
        for atomic_span in getattr(occurrence, "atomic_modifier_spans", []) or []:
            if isinstance(atomic_span, dict):
                modifier_units.append(atomic_span.get("text_norm") or atomic_span.get("text") or "")

        deduped = []
        seen = set()
        for modifier in modifier_units:
            modifier_norm = normalize_text(str(modifier).strip())
            if not modifier_norm or modifier_norm in seen:
                continue
            seen.add(modifier_norm)
            deduped.append(modifier_norm)
        return deduped

    # Score one occurrence against query modifiers without filtering non-matches.
    def _modifier_boost_for_occurrence(self, occurrence, match):
        query_modifiers = [
            normalize_text(modifier)
            for modifier in match.get("query_modifiers_norm", [])
            if normalize_text(modifier)
        ]
        if not query_modifiers:
            return 0.0

        boost = 0.0
        query_surface = match.get("query_surface_norm") or normalize_text(match.get("query_surface", ""))
        occurrence_surface = normalize_text(
            getattr(occurrence, "surface_text", None)
            or getattr(occurrence, "span_text", None)
            or ""
        )
        if query_surface and occurrence_surface == query_surface:
            boost = max(boost, 0.5)

        occurrence_modifiers = self._get_occurrence_modifier_units_norm(occurrence)
        occurrence_modifier_set = set(occurrence_modifiers)
        if occurrence_modifier_set and all(modifier in occurrence_modifier_set for modifier in query_modifiers):
            boost = max(boost, 0.3)
        else:
            occurrence_modifier_tokens = set()
            for modifier in occurrence_modifiers:
                occurrence_modifier_tokens.update(modifier.split())
            query_modifier_tokens = set()
            for modifier in query_modifiers:
                query_modifier_tokens.update(modifier.split())

            if occurrence_modifier_tokens and query_modifier_tokens:
                overlap_ratio = len(query_modifier_tokens & occurrence_modifier_tokens) / len(query_modifier_tokens)
                if overlap_ratio >= 1.0:
                    boost = max(boost, 0.3)
                elif overlap_ratio > 0:
                    boost = max(boost, 0.1 + 0.1 * overlap_ratio)

        return boost

    # Compute the best modifier boost per chunk for the semantic nodes selected by the query.
    def _score_chunk_modifier_boosts(self, resolved_matches):
        chunk_boosts = defaultdict(float)
        for match in resolved_matches:
            if match.get("phrase_type") != PHRASE_TYPE_COMPOSITIONAL:
                continue
            if not match.get("query_modifiers_norm"):
                continue
            for sem_node in self._get_match_sem_nodes(match):
                span_occurrences = list(getattr(sem_node, "span_occurrences", []) or [])
                for occurrence in span_occurrences:
                    chunk_node = getattr(occurrence, "chunk_node", None)
                    if chunk_node is None:
                        continue
                    boost = self._modifier_boost_for_occurrence(occurrence, match)
                    if boost > chunk_boosts[chunk_node.chunk_node_id]:
                        chunk_boosts[chunk_node.chunk_node_id] = boost
        return dict(chunk_boosts)

    # Apply modifier boosts to scored chunk candidates while preserving base order on ties.
    def _apply_modifier_boost_to_scored_chunks(self, scored_chunks, modifier_boosts):
        boosted_chunks = [
            (chunk_id, score * (1.0 + modifier_boosts.get(chunk_id, 0.0)))
            for chunk_id, score in scored_chunks
        ]
        return sorted(boosted_chunks, key=lambda item: item[1], reverse=True)

    # Rerank chunk id lists by modifier boost with stable fallback to the original order.
    def _apply_modifier_boost_to_chunk_ids(self, chunk_ids, modifier_boosts):
        return sorted(
            chunk_ids,
            key=lambda chunk_id: modifier_boosts.get(chunk_id, 0.0),
            reverse=True,
        )

    # Rerank grouped BM25 chunk lists by modifier boost with stable fallback order.
    def _apply_modifier_boost_to_ranked_groups(self, ranked_groups, modifier_boosts):
        return [
            self._apply_modifier_boost_to_chunk_ids(group, modifier_boosts)
            for group in ranked_groups
        ]

    # Retrieve and rerank chunks for a query using broad semantic-node matching.
    def broad_search_query(self, query_text, top_k=10,candidate=30):
        query_text, _, resolved_matches = self._resolve_query_matches(query_text)
        tokens_in_phrase = []
        retrieved_sem_nodes = []
        for match in resolved_matches:
            token_type = match["token_type"]
            token = match["token"]
            if token_type == "token" and token in tokens_in_phrase:
                continue
            if match["exact_sem_node"] is not None:
                retrieved_sem_nodes.append(match["exact_sem_node"])
                if token_type in {"phrase", "ent"}:
                    tokens_in_phrase.extend(token.split(" "))
            if match["fuzzy_sem_nodes"]:
                retrieved_sem_nodes.extend(match["fuzzy_sem_nodes"])
            elif match["retrieved_sem"] is not None:
                retrieved_sem_nodes.append(match["retrieved_sem"])
        retrieved_chunk_ids = []
        for sem_node in retrieved_sem_nodes:
            for chunk_node in sem_node.chunk_node_list:
                retrieved_chunk_ids.append(chunk_node.chunk_node_id)
        retrieved_chunk_ids = list(dict.fromkeys(retrieved_chunk_ids))
        modifier_boosts = self._score_chunk_modifier_boosts(resolved_matches)
        if modifier_boosts:
            retrieved_chunk_ids = self._apply_modifier_boost_to_chunk_ids(
                retrieved_chunk_ids,
                modifier_boosts,
            )
        candidate_chunk_ids = retrieved_chunk_ids[:candidate]
        retrieved_chunks = self.chunk_id2text(candidate_chunk_ids)
        if self.reranker is None:
            return retrieved_chunks[:top_k], candidate_chunk_ids[:top_k]
        rerank_chunks, rerank_chunk_index= self.reranker.rerank(query_text, retrieved_chunks, top_k=top_k)
        rerank_chunk_ids = [candidate_chunk_ids[i] for i in rerank_chunk_index]

        return rerank_chunks, rerank_chunk_ids

    # Build the co-occurrence node infos (low/high level sem-info tuples) from resolved matches.
    #
    # Shared by multi_level_query (legacy) and chunk_cooccur_query. Returns the low-level and
    # high-level sem-info lists plus token-debug lists; optionally prints the token breakdown.
    #
    # Phrase/sub-token dedup is NOT repeated here: _resolve_query_matches() already drops the
    # sub-tokens of an exactly-matched phrase before they reach resolved_matches. A phrase that
    # only matched fuzzily is resolved against a different (longer) sem node than its sub-tokens,
    # so those sub-tokens are deliberately kept as complementary recall signals.
    def _collect_query_cooccurrence_sems(self, resolved_matches, print_important_tokens):
        query_tokens = []
        low_level_tokens = []
        high_level_tokens = []
        low_level_sems = []
        high_level_sems = []

        def make_sem_info(sem_node, node_level, node_query_weight, match):
            if match.get("query_group_id") is None:
                return (sem_node, node_level, node_query_weight)
            return (
                sem_node,
                node_level,
                node_query_weight,
                match.get("query_group_id"),
                match.get("query_group_role"),
                match.get("query_member_key"),
                tuple(match.get("query_group_member_keys") or ()),
                match.get("query_role_weight"),
            )

        for match in resolved_matches:
            token_type = match["token_type"]
            token = match["token"]
            query_tokens.append(token)
            exact_match = match["exact_sem_node"] is not None
            fuzzy_match = len(match["fuzzy_sem_nodes"]) > 0
            if exact_match:
                max_sem_node = match["exact_sem_node"]
                low_level_tokens.append(token + f"(exact matched)")
                if token_type == 'ent':
                    low_level_sems.append(make_sem_info(max_sem_node, 0, 1, match))
                else:
                    low_level_sems.append(make_sem_info(max_sem_node, 1, 1, match))
            if fuzzy_match:
                for sem_node in match["fuzzy_sem_nodes"][:1]:
                    weight = count_words(token) / count_words(sem_node.token_node.token_text)
                    if exact_match:
                        if token_type == 'ent':
                            high_level_sems.append(make_sem_info(sem_node, 1, weight, match))
                        else:
                            high_level_sems.append(make_sem_info(sem_node, 2, weight, match))
                        high_level_tokens.append(
                            sem_node.token_node.token_text + f"(partial matched)")
                    else:
                        if token_type == 'ent':
                            low_level_sems.append(make_sem_info(sem_node, 1, weight, match))
                        else:
                            low_level_sems.append(make_sem_info(sem_node, 2, weight, match))
                        low_level_tokens.append(
                            sem_node.token_node.token_text + f"(partial matched)")
            if not (exact_match or fuzzy_match):
                low_level_sems.append(
                    make_sem_info(match["retrieved_sem"], 3, match["semantic_weight"], match)
                )
                low_level_tokens.append(match["retrieved_sem"].token_node.token_text + f"(sim matched)")

                high_level_tokens.append(['N/A'])

        if print_important_tokens:
            print(f"query tokens: {[text for text in query_tokens]}")
            print(f"low level tokens: {[text for text in low_level_tokens]}")
            print(f"high level tokens: {[text for text in high_level_tokens]}")

        return low_level_sems, high_level_sems

    # Query-time IDF-based generic-word pruning (borrows NoLLMRAG's IS threshold idea).
    #
    # NoLLMRAG keeps only query tokens whose importance score exceeds `tau * max_score`
    # (relative, per-query) so that generic predicates (person/drink/state) are dropped from
    # keyword matching entirely instead of merely down-weighted. We reuse the already-persisted
    # `token_node.idf` as the discriminativeness signal and apply the same relative cut, with two
    # safety rails tuned for a system that has no strong full-sentence dense fallback:
    #   * gate: only prune when max_idf >= `min_max_idf` (there is a genuinely discriminative token
    #     -- a real peak); an all-generic query has no peak and is left untouched.
    #   * keep-top: always retain the `keep_top` highest-idf sems regardless of the threshold, so
    #     the relative cut can never starve the pair-boost mechanism (which needs >=2 discriminative
    #     tokens in the same chunk to fire).
    # Compositional query-group members (tuple len > 3) are never pruned: a phrase head/modifier is
    # inherently discriminative and removing one would distort grouped-chunk scoring.
    #
    # `idf_prune_tau is None` disables pruning entirely (identical to the pre-change behavior).
    # Returns (low_level_sems, high_level_sems, dropped_tokens) where dropped_tokens is for debug.
    def _idf_prune_query_sems(
        self,
        low_level_sems,
        high_level_sems,
        idf_prune_tau,
        min_max_idf=None,
        keep_top=2,
        print_important_tokens=False,
    ):
        if idf_prune_tau is None:
            return low_level_sems, high_level_sems, []

        def is_groupable(sem_info):
            # tuple len > 3 carries query_group_id et al. -> compositional member, never prune.
            return len(sem_info) <= 3

        prunable = [
            s for s in (low_level_sems + high_level_sems) if is_groupable(s)
        ]
        if not prunable:
            return low_level_sems, high_level_sems, []

        def sem_idf(sem_info):
            return float(getattr(sem_info[0].token_node, "idf", 0.0) or 0.0)

        max_idf = max(sem_idf(s) for s in prunable)
        # Gate: no clearly discriminative token -> leave the query untouched.
        if min_max_idf is not None and max_idf < min_max_idf:
            return low_level_sems, high_level_sems, []

        threshold = idf_prune_tau * max_idf
        # Force-keep the top-`keep_top` prunable sems by idf (identity set), then keep anything
        # above threshold. Rank by idf desc; ties broken by token text for determinism.
        ranked = sorted(
            prunable,
            key=lambda s: (-sem_idf(s), s[0].token_node.token_text),
        )
        keep_ids = {id(s) for s in ranked[: max(keep_top, 1)]}

        def keep(sem_info):
            if not is_groupable(sem_info):
                return True
            if id(sem_info) in keep_ids:
                return True
            return sem_idf(sem_info) > threshold

        dropped_tokens = [
            s[0].token_node.token_text
            for s in prunable
            if id(s) not in keep_ids and sem_idf(s) <= threshold
        ]
        new_low = [s for s in low_level_sems if keep(s)]
        new_high = [s for s in high_level_sems if keep(s)]

        if print_important_tokens and dropped_tokens:
            print(
                f"[idf_prune] tau={idf_prune_tau} max_idf={max_idf:.3f} "
                f"thr={threshold:.3f} dropped={dropped_tokens}"
            )
        return new_low, new_high, dropped_tokens

    # Chunk-level bounded co-occurrence boosting query.
    #
    # ChunkScore(c) = BaseEvidence(c) * (1 + λ * PairBoost(c)), where PairBoost(c) averages the
    # top-k pair boosts and a pair boost only fires when a candidate chunk contains BOTH endpoints
    # of a co-occurrence edge. No connected/isolated quota — every candidate ranks by ChunkScore.
    # The pair-boost pass is restricted to the top `candidate_pool_size` chunks by base evidence.
    #
    # Returns (chunk_texts, chunk_ids, debug_info). debug_info is a structured dict: params,
    # resolved_matches, token_weight_map, all_scored_chunks (full sorted), and top_chunks.
    def chunk_cooccur_query(
        self,
        query_text,
        top_k_chunk=10,
        top_k_pair_boosts=COOCCUR_TOP_K_PAIR_BOOSTS,
        lambda_boost=COOCCUR_LAMBDA,
        candidate_pool_size=COOCCUR_CANDIDATE_POOL_SIZE,
        print_important_tokens=True,
        search_mode='broad',
        idf_prune_tau=None,
        idf_prune_min_max_idf=None,
        idf_prune_keep_top=2,
    ):
        # ---- Parameter validation (fail loud instead of silently degrading) ----
        # bool is a subclass of int, so reject it explicitly to avoid True being read as 1.
        if isinstance(top_k_chunk, bool) or not isinstance(top_k_chunk, int) or top_k_chunk < 1:
            raise ValueError(f"top_k_chunk must be a positive int, got {top_k_chunk!r}")
        if (
            isinstance(top_k_pair_boosts, bool)
            or not isinstance(top_k_pair_boosts, int)
            or top_k_pair_boosts < 1
        ):
            # >=1 enforced on purpose: 0/negative would silently switch off the pair boost.
            raise ValueError(
                f"top_k_pair_boosts must be a positive int, got {top_k_pair_boosts!r}"
            )
        if isinstance(lambda_boost, bool) or not isinstance(lambda_boost, (int, float)) or lambda_boost < 0:
            # negative lambda would turn co-occurrence evidence into a penalty.
            raise ValueError(f"lambda_boost must be a non-negative number, got {lambda_boost!r}")
        if candidate_pool_size is not None and (
            isinstance(candidate_pool_size, bool)
            or not isinstance(candidate_pool_size, int)
            or candidate_pool_size < 1
        ):
            raise ValueError(
                f"candidate_pool_size must be a positive int or None, got {candidate_pool_size!r}"
            )
        if search_mode not in {"broad", "precise"}:
            raise ValueError(f"search_mode must be 'broad' or 'precise', got {search_mode!r}")

        query_text, _, resolved_matches = self._resolve_query_matches(
            query_text,
            search_mode=search_mode,
            print_important_tokens=print_important_tokens,
            expand_compositional=True,
        )
        low_level_sems, high_level_sems = self._collect_query_cooccurrence_sems(
            resolved_matches, print_important_tokens
        )
        low_level_sems, high_level_sems, _ = self._idf_prune_query_sems(
            low_level_sems,
            high_level_sems,
            idf_prune_tau=idf_prune_tau,
            min_max_idf=idf_prune_min_max_idf,
            keep_top=idf_prune_keep_top,
            print_important_tokens=print_important_tokens,
        )

        graph = CoOccurrenceGraph(low_level_sems + high_level_sems)
        pair_edge_weight = graph.build_pair_edges()
        base_evidence, token_weight_map = graph.assign_base_evidence()
        chunk_to_nodes = graph.nodes_by_chunk()

        # Candidate-pool cutoff: restrict the pair-boost pass to the top chunks by base evidence.
        # Chunks outside the pool keep pair_boost=0 (final_score == base) and still compete in the
        # final ranking, but we skip their O(n^2) pair combinations. None => score every chunk.
        if candidate_pool_size is not None and len(base_evidence) > candidate_pool_size:
            pool_chunk_ids = {
                chunk_node_id
                for chunk_node_id, _ in sorted(
                    base_evidence.items(), key=lambda kv: kv[1], reverse=True
                )[:candidate_pool_size]
            }
        else:
            pool_chunk_ids = None

        sentence_boundary_cache = {}
        scored_chunks = []
        for chunk_node_id, base in base_evidence.items():
            if pool_chunk_ids is not None and chunk_node_id not in pool_chunk_ids:
                scored_chunks.append({
                    "chunk_id": chunk_node_id,
                    "base_evidence": base,
                    "pair_boost": 0.0,
                    "final_score": base,
                    "top_pairs": [],
                })
                continue
            nodes = chunk_to_nodes.get(chunk_node_id, [])
            pair_details = []
            for node_a, node_b in combinations(nodes, 2):
                # Group-level collapse: skip pairs whose two nodes were split from the SAME
                # compositional query phrase (same non-None query_group_id). Their co-occurrence
                # inside a chunk is expected by construction, not cross-concept interaction, and
                # the "real phrase" signal is already carried by the full phrase node in
                # assign_base_evidence. Cross-group pairs and token pairs are unaffected.
                if (
                    node_a.query_group_id is not None
                    and node_a.query_group_id == node_b.query_group_id
                ):
                    continue
                key = frozenset((id(node_a.sem_node), id(node_b.sem_node)))
                global_edge_weight = pair_edge_weight.get(key)
                if global_edge_weight is None:
                    continue
                local_evidence = self._local_evidence_level(
                    node_a, node_b, chunk_node_id, sentence_boundary_cache
                )
                # Endpoint reliability: node_level_weight already folds match level
                # (exact/fuzzy/semantic-fallback), compositional role, and the per-match
                # query weight (cosine / phrase coverage). Multiply by the geometric mean
                # so a pair anchored on a weak semantic fallback or a low-weight modifier
                # cannot boost a chunk as much as a pair of strong exact matches.
                endpoint_reliability = (
                    node_a.node_level_weight * node_b.node_level_weight
                ) ** 0.5
                pair_boost = global_edge_weight * local_evidence * endpoint_reliability
                pair_details.append({
                    "semantic_node_1": node_a.sem_node.token_node.token_text,
                    "semantic_node_2": node_b.sem_node.token_node.token_text,
                    "global_edge_weight": global_edge_weight,
                    "local_evidence_level": local_evidence,
                    "endpoint_reliability": endpoint_reliability,
                    "pair_boost": pair_boost,
                })

            pair_details.sort(key=lambda detail: detail["pair_boost"], reverse=True)
            top_pairs = pair_details[:top_k_pair_boosts]
            if top_pairs:
                pair_boost_value = sum(d["pair_boost"] for d in top_pairs) / len(top_pairs)
            else:
                pair_boost_value = 0.0
            final_score = base * (1 + lambda_boost * pair_boost_value)
            scored_chunks.append({
                "chunk_id": chunk_node_id,
                "base_evidence": base,
                "pair_boost": pair_boost_value,
                "final_score": final_score,
                "top_pairs": top_pairs,
            })

        scored_chunks.sort(key=lambda record: record["final_score"], reverse=True)
        top_chunks = scored_chunks[:top_k_chunk]
        retrieved_chunk_ids = [record["chunk_id"] for record in top_chunks]

        if print_important_tokens:
            print(token_weight_map)
            for record in top_chunks:
                print(
                    f"chunk {record['chunk_id']}: base={record['base_evidence']:.4f}, "
                    f"pair_boost={record['pair_boost']:.4f}, final={record['final_score']:.4f}, "
                    f"pairs={record['top_pairs']}"
                )

        # Structured debug payload (third return value). Keeps the full picture so notebooks /
        # eval scripts can explain why a chunk did or did not reach top-k: base-vs-final ranking,
        # which tokens contributed base evidence, and the candidate-pool size that was applied.
        # `top_chunks` stays accessible under the same key for existing callers.
        debug_info = {
            "params": {
                "top_k_chunk": top_k_chunk,
                "top_k_pair_boosts": top_k_pair_boosts,
                "lambda_boost": lambda_boost,
                "candidate_pool_size": candidate_pool_size,
                "search_mode": search_mode,
            },
            "resolved_matches": resolved_matches,
            "token_weight_map": token_weight_map,
            "candidate_pool_applied": pool_chunk_ids is not None,
            "num_scored_chunks": len(scored_chunks),
            "all_scored_chunks": scored_chunks,
            "top_chunks": top_chunks,
        }

        return self.chunk_id2text(retrieved_chunk_ids), retrieved_chunk_ids, debug_info

    # Two-stage bridge retrieval for multi-hop (bridge) questions — Implementation Plan One.
    #
    # Thin wrapper that delegates to multihop_reasoning.multihop_bridge_query so RAG_graph.py
    # does not keep growing. Reuses chunk_cooccur_query() for first-hop recall, then extracts
    # bridge semantic nodes from the first-hop chunks and runs a bridge-driven second hop.
    # Returns (chains, retrieved_chunk_ids, debug_info); does NOT touch index/finalize/schema
    # or the existing chunk_cooccur_query() path (single-hop eval scripts are unaffected).
    def multihop_bridge_query(self, query_text, *args, **kwargs):
        from multihop_reasoning import multihop_bridge_query
        return multihop_bridge_query(self, query_text, *args, **kwargs)

    # Chunk-level bounded boosting: local evidence level for a co-occurring node pair.
    #
    # Two tiers (chunk-side verified, highest applicable wins):
    #   - same sentence (0.7): both nodes have an occurrence inside the same chunk sentence,
    #     decided from index-time sentence ids (schema 7) or the chunk's stored sentence
    #     boundaries. This is real chunk-side positional evidence, not query-side grouping.
    #   - same chunk (0.4): they only share the chunk.
    #
    # There is deliberately no "phrase" tier keyed on query_group_id: that only encodes that the
    # two nodes were split from one compositional query phrase, which says nothing about whether
    # they actually form a phrase inside THIS chunk. The "real phrase" signal is carried instead by
    # the full phrase node matching exactly in assign_base_evidence (weight 1.0, with same-group
    # head/modifier suppressed there), so re-boosting it here would double-count / inflate chunks
    # where the two words merely co-occur apart.
    def _local_evidence_level(self, node_a, node_b, chunk_node_id, sentence_boundary_cache=None):
        if sentence_boundary_cache is not None and self._nodes_share_sentence(
            node_a.sem_node, node_b.sem_node, chunk_node_id, sentence_boundary_cache
        ):
            return COOCCUR_LOCAL_EVIDENCE["sentence"]
        return COOCCUR_LOCAL_EVIDENCE["chunk"]

    # Collect the span occurrences of a sem node within one chunk.
    def _sem_node_occurrences_in_chunk(self, sem_node, chunk_node_id):
        occurrences = []
        for occurrence in getattr(sem_node, "span_occurrences", []) or []:
            chunk_node = getattr(occurrence, "chunk_node", None)
            if chunk_node is None or chunk_node.chunk_node_id != chunk_node_id:
                continue
            occurrences.append(occurrence)
        return occurrences

    # Resolve the ChunkNode a sem node occurs in, by chunk id.
    def _chunk_node_for_sem_node(self, sem_node, chunk_node_id):
        for chunk_node in getattr(sem_node, "chunk_node_list", []) or []:
            if chunk_node.chunk_node_id == chunk_node_id:
                return chunk_node
        return None

    # Sentence-boundary char offsets for a chunk, cached by chunk id.
    #
    # Prefer the index-time spaCy boundaries (ChunkNode.sentence_boundaries, schema 7: sorted
    # sent.end_char offsets). Fall back to a .!? scan only for pre-schema-7 chunks that never
    # stored them — that scan misjudges abbreviations / decimals / numbering and is last resort.
    def _chunk_sentence_boundaries(self, chunk_node, chunk_node_id, cache):
        if chunk_node_id in cache:
            return cache[chunk_node_id]
        boundaries = list(getattr(chunk_node, "sentence_boundaries", None) or [])
        if not boundaries:
            chunk_text = getattr(chunk_node, "chunk_text", "") or ""
            boundaries = [idx for idx, ch in enumerate(chunk_text) if ch in ".!?"]
        cache[chunk_node_id] = boundaries
        return boundaries

    # Sentence index of a char position = number of sentence boundaries strictly before it.
    @staticmethod
    def _sentence_index_of(position, boundaries):
        lo, hi = 0, len(boundaries)
        while lo < hi:
            mid = (lo + hi) // 2
            if boundaries[mid] < position:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # True when both sem nodes have an occurrence inside the same chunk sentence.
    #
    # Fast path: index-time sentence ids (schema 7) — pure int-set intersection.
    # Fallback: derive sentence index from span char-offsets, using the chunk's stored
    # sentence boundaries when available and a .!? scan only as a last resort, for old
    # pickles whose occurrences predate the precomputed sentence ids.
    def _nodes_share_sentence(self, sem_a, sem_b, chunk_node_id, cache):
        occ_a = self._sem_node_occurrences_in_chunk(sem_a, chunk_node_id)
        occ_b = self._sem_node_occurrences_in_chunk(sem_b, chunk_node_id)
        if not occ_a or not occ_b:
            return False

        sids_a = {o.sentence_id for o in occ_a if getattr(o, "sentence_id", None) is not None}
        sids_b = {o.sentence_id for o in occ_b if getattr(o, "sentence_id", None) is not None}
        if sids_a and sids_b:
            return bool(sids_a & sids_b)

        chunk_node = self._chunk_node_for_sem_node(sem_a, chunk_node_id)
        if chunk_node is None:
            return False
        boundaries = self._chunk_sentence_boundaries(chunk_node, chunk_node_id, cache)
        if not boundaries:
            return False
        starts_a = [o.span_start for o in occ_a if getattr(o, "span_start", None) is not None]
        starts_b = [o.span_start for o in occ_b if getattr(o, "span_start", None) is not None]
        if not starts_a or not starts_b:
            return False
        sentences_a = {self._sentence_index_of(start, boundaries) for start in starts_a}
        sentences_b = {self._sentence_index_of(start, boundaries) for start in starts_b}
        return bool(sentences_a & sentences_b)

    # Retrieve chunks using exact, fuzzy, and semantic matches organized by match level.
    def multi_level_query(self, query_text, top_k_chunk=10, top_k_each_isolated_chunk=2, isolate_chunk_ratio=0.2, isolate_retrieve_mode='sequential',print_important_tokens=True, search_mode='broad'):
        query_text, query_tokens, resolved_matches = self._resolve_query_matches(
            query_text,
            search_mode=search_mode,
            print_important_tokens=print_important_tokens,
            expand_compositional=True,
        )
        low_level_sems, high_level_sems = self._collect_query_cooccurrence_sems(
            resolved_matches, print_important_tokens
        )

        co_occurrence_graph = CoOccurrenceGraph(low_level_sems + high_level_sems)
        co_occurrence_graph.build_edges()
        co_occurrence_graph.assign_chunk_weight(self.chunk_avg_len,print_important_tokens)
        co_occurrence_graph.rank_sem_node()
        co_occurrence_graph.rank_chunk_by_BM25()

        isolated_quota = math.floor(top_k_chunk * isolate_chunk_ratio)
        connected_quota = top_k_chunk - isolated_quota
        connected_candidates = [
            chunk_id for chunk_id, _ in co_occurrence_graph.weighted_chunk_node_list
        ]
        retrieved_connected_chunk = connected_candidates[:connected_quota]

        isolate_chunk_extractor = ListBatchExtractor(
            co_occurrence_graph.ranked_chunk_BM25,
            mode=isolate_retrieve_mode,
            k=top_k_each_isolated_chunk,
            exclude_list=retrieved_connected_chunk,
        )
        retrieved_isolated_chunk = isolate_chunk_extractor.extract(isolated_quota, [])

        if len(retrieved_connected_chunk) < connected_quota:
            retrieved_isolated_chunk = isolate_chunk_extractor.extract(
                top_k_chunk - len(retrieved_connected_chunk),
                retrieved_isolated_chunk,
            )
        elif len(retrieved_isolated_chunk) < isolated_quota:
            isolated_chunk_ids = set(retrieved_isolated_chunk)
            retrieved_connected_chunk = [
                chunk_id
                for chunk_id in connected_candidates
                if chunk_id not in isolated_chunk_ids
            ][:top_k_chunk - len(retrieved_isolated_chunk)]

        retrieved_chunk_ids = []
        seen_chunk_ids = set()
        for chunk_id in retrieved_connected_chunk + retrieved_isolated_chunk:
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            retrieved_chunk_ids.append(chunk_id)
            if len(retrieved_chunk_ids) >= top_k_chunk:
                break

        return (
            self.chunk_id2text(retrieved_chunk_ids),
            retrieved_chunk_ids,
            co_occurrence_graph,
        )

    # Return the highest-weight unique chunks connected to a semantic node.
    def get_top_k_chunks_for_sem_node(self, sem_node, top_k=None, retrieve_all=False):
        top_k = self.retrieve_top_k if top_k is None else top_k
        sorted_indices = sorted(
            range(len(sem_node.chunk_edge_weight)),
            key=lambda i: sem_node.chunk_edge_weight[i],
            reverse=True
        )
        result = []
        result_chunk_id = []
        seen = set()
        for idx in sorted_indices:
            node = sem_node.chunk_node_list[idx]
            if node not in seen:
                result.append(node.chunk_text)
                result_chunk_id.append(node.chunk_node_id)
                seen.add(node)
            if len(result) == top_k and not retrieve_all:
                break
        return result, result_chunk_id

    # Find the nearest semantic-node embeddings for query embeddings.
    def query_by_sim(self, query_embeds):
        if self.query_database is None:
            raise ValueError("Query database is empty; finalize or build the query database before querying.")
        query_tensor = torch.stack(query_embeds).to(self.device)
        query_tensor = F.normalize(query_tensor, dim=1)
        sims = torch.matmul(self.query_database, query_tensor.T)
        best_scores, best_indices = torch.max(sims, dim=0)

        return best_indices.tolist(), best_scores.tolist()

    # Finalize graph metadata, semantic descriptions, retrieval indexes, and document output.
    def finalize(self):
        """Finalize after indexing.

        Required ordering: ensure token nodes own semantic nodes, merge/split semantic
        nodes by description, assign IDF, compute BM25, then rebuild query and chunk
        indexes from the final semantic-node set.
        """
        self._reset_sem_description_logs()
        self.log_time("Finalize started.")
        if not self.chunk_nodes:
            raise ValueError("Cannot finalize an empty graph: no chunk nodes have been indexed.")
        self._recompute_chunk_avg_len()
        self.log_time("Computed average chunk length.")
        removed_placeholder_count = self._cleanup_empty_placeholder_token_nodes()
        if removed_placeholder_count:
            self.log_time(f"Removed {removed_placeholder_count} empty placeholder token nodes.")
        self.finalize_token_nodes()
        self.log_time("Finished token node finalization.")
        self._print_sem_build_stats()

        if not self.sem_nodes:
            raise ValueError("Cannot finalize graph: no semantic nodes were created.")
        self.build_modifier_postings()
        self.log_time("Finished building modifier postings.")
        for token_node in self.token_nodes:
            self.assign_idf(token_node)
        self.log_time("Finished assigning token and sem IDF.")
        self.get_sem_BM25()
        self.log_time("Finished computing sem BM25.")
        self.build_query_database()
        self.log_time("Finished building query database.")
        self.build_phrase_query()
        self.log_time("Finished building phrase query index.")
        self.build_chunk2sem_edge()
        self.log_time("Finished building chunk-to-sem edges.")
        self.save_doc_to_json()
        log_path = self._save_sem_description_logs_to_timestamped_file()
        self.log_time(f"Saved sem description logs to {log_path}.")
        self.log_time("Finalizing completed.")
        self._print_index_session_timing()

    # Persist sem description logs to a timestamped file under <project_root>/logs/.
    def _save_sem_description_logs_to_timestamped_file(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(logs_dir, f"sem_description_{timestamp}.log")
        return self.save_sem_description_logs(output_path=output_path, as_html=False)

    # Build semantic nodes for every token node from its fully accumulated embeddings.
    #
    # 索引期只累积 embedding,从不建语义节点;此处每个 token 的 buffer 里就是它的
    # 全部出现次数。分两趟处理:先建潜在多义节点(出现次数 >= 阈值,走完整的
    # build_sem_node:描述预测 / 多义切分,通常是最耗时的一段),再建剩余的廉价
    # 基础节点(create_basic_sem_node:单节点、无描述)。
    def finalize_token_nodes(self):
        if not self.token_nodes:
            return

        # The combined merge stores sample_order -> description per token; clear it
        # so a fresh finalize re-runs the combined LLM call against current samples.
        self._combined_merge_sample_sense_descriptions = {}
        self._reset_llm_anchor_sample_assignments()

        min_occ = getattr(self, "min_occurrences_for_description", None)
        fast_index = getattr(self, "fast_index", False)

        # Pass 0: split nodes into the expensive multi-sense candidates and the
        # cheap description-less remainder. A node is a multi-sense candidate when
        # its accumulated occurrence count reaches min_occurrences_for_description.
        # fast index 模式下永远不产生多义候选:所有 token/phrase 一律建单义 basic 节点。
        multi_sense_candidates = []
        basic_only = []
        for token_node in self.token_nodes:
            if token_node.has_semantic:
                continue
            occurrence_count = len(token_node.embeds_buffer)
            if not fast_index and min_occ is not None and occurrence_count >= min_occ:
                multi_sense_candidates.append(token_node)
            else:
                basic_only.append(token_node)

        if fast_index:
            print(
                f"finalize_token_nodes [fast_index]: all {len(basic_only)} node(s) "
                "built as single-sense basic node(s); polysemy analysis skipped."
            )
        else:
            print(
                f"finalize_token_nodes: {len(multi_sense_candidates)} potential "
                f"multi-sense node(s) (occurrences >= min_occurrences_for_description="
                f"{min_occ}), {len(basic_only)} basic node(s)."
            )

        # Pass 1: build the potential multi-sense nodes first. This is the
        # expensive stage (s_mean gate -> FFT sampling -> per-sample description
        # prediction) and usually dominates finalize time.
        total_candidates = len(multi_sense_candidates)
        if total_candidates:
            print("  Building potential multi-sense nodes (most time-consuming stage)...")
            progress_update = self._make_progress_updater(
                "multi_sense_node_build",
                total_candidates,
                show_remaining=True,
            )
            progress_update(0, force=True)
            for index, token_node in enumerate(multi_sense_candidates, start=1):
                self.build_sem_node(token_node)  # clears its own embeds_buffer
                progress_update(index)
            progress_update(total_candidates, force=True)
            print()

        # Pass 2: build the remaining cheap, description-less basic nodes.
        total_basic = len(basic_only)
        if total_basic:
            progress_update = self._make_progress_updater(
                "basic_node_build",
                total_basic,
                show_remaining=True,
            )
            progress_update(0, force=True)
            for index, token_node in enumerate(basic_only, start=1):
                self.create_basic_sem_node(token_node)
                token_node.embeds_buffer.clear()
                progress_update(index)
            progress_update(total_basic, force=True)
            print()

    # Sample unique chunks connected to a semantic node.
    def _sample_sem_chunk_nodes(self, sem_node, max_samples=10):
        unique_chunk_nodes = []
        unique_embeds = []
        seen_chunk_ids = set()
        chunk_embeds = list(getattr(sem_node, "chunk_node_embed", []) or [])
        for idx, chunk_node in enumerate(sem_node.chunk_node_list):
            if chunk_node.chunk_node_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_node.chunk_node_id)
            unique_chunk_nodes.append(chunk_node)
            if idx < len(chunk_embeds):
                unique_embeds.append(chunk_embeds[idx])
        if max_samples is None or max_samples <= 0:
            return unique_chunk_nodes
        if len(unique_chunk_nodes) <= max_samples:
            return unique_chunk_nodes
        if len(unique_embeds) == len(unique_chunk_nodes):
            selected_idx = self._run_farthest_first_traversal(unique_embeds, max_samples)
            return [unique_chunk_nodes[i] for i in selected_idx]
        return unique_chunk_nodes[:max_samples]

    # Sample span occurrences for semantic-description prediction.
    def _sample_sem_span_occurrences(self, sem_node, max_samples=10):
        span_occurrences = list(getattr(sem_node, "span_occurrences", []))
        if span_occurrences:
            if max_samples is None or max_samples <= 0:
                return span_occurrences
            if len(span_occurrences) <= max_samples:
                return span_occurrences
            chunk_embeds = list(getattr(sem_node, "chunk_node_embed", []) or [])
            if len(chunk_embeds) == len(span_occurrences):
                selected_idx = self._run_farthest_first_traversal(chunk_embeds, max_samples)
                return [span_occurrences[i] for i in selected_idx]
            return span_occurrences[:max_samples]

        fallback_chunk_nodes = self._sample_sem_chunk_nodes(sem_node, max_samples=max_samples)
        return [SpanOccurrence(chunk_node=chunk_node) for chunk_node in fallback_chunk_nodes]

    # Find the nearest sentence boundary to the left of a character index.
    def _find_left_boundary_for_description(self, text, index):
        return max(
            text.rfind(".", 0, index),
            text.rfind("!", 0, index),
            text.rfind("?", 0, index),
        )

    # Find the nearest sentence boundary to the right of a character index.
    def _find_right_boundary_for_description(self, text, index):
        right_candidates = [
            text.find(".", index),
            text.find("!", index),
            text.find("?", index),
        ]
        right_candidates = [idx for idx in right_candidates if idx != -1]
        return len(text) if not right_candidates else min(right_candidates) + 1

    # Build description-prompt context and local match coordinates from text bounds.
    def _build_description_context_from_bounds(self, chunk_text, match_span, context_start, context_end):
        match_start, match_end = match_span
        context_raw = chunk_text[context_start:context_end]

        if not context_raw.strip():
            context_start = max(0, match_start - 120)
            context_end = min(len(chunk_text), match_end + 120)
            context_raw = chunk_text[context_start:context_end]

        left_trim = len(context_raw) - len(context_raw.lstrip())
        context_text = context_raw.strip()
        local_start = match_start - context_start - left_trim
        local_end = match_end - context_start - left_trim
        local_start = max(0, min(local_start, len(context_text)))
        local_end = max(local_start, min(local_end, len(context_text)))

        return {
            "context_text": context_text,
            "matched_text": context_text[local_start:local_end],
            "local_span": (local_start, local_end),
        }

    # Locate a token text span inside a chunk for description prompting.

    # Extract the sentence containing the target token for description prediction.
    def _extract_sentence_context_for_description(self, chunk_text, token_text, match_span=None):
        if match_span is None:
            match_span = _find_description_match_span(chunk_text, token_text)
        if match_span is None:
            return None
        match_start, match_end = match_span
        left_boundary = self._find_left_boundary_for_description(chunk_text, match_start)
        context_start = 0 if left_boundary == -1 else left_boundary + 1
        context_end = self._find_right_boundary_for_description(chunk_text, match_end)
        return self._build_description_context_from_bounds(
            chunk_text,
            match_span,
            context_start,
            context_end,
        )

    # Extract the target sentence plus neighboring sentence context for description prediction.
    def _extract_neighbor_sentence_context_for_description(self, chunk_text, token_text, match_span=None):
        if match_span is None:
            match_span = _find_description_match_span(chunk_text, token_text)
        if match_span is None:
            return None
        match_start, match_end = match_span
        left_boundary = self._find_left_boundary_for_description(chunk_text, match_start)
        context_start = 0 if left_boundary == -1 else left_boundary + 1
        context_end = self._find_right_boundary_for_description(chunk_text, match_end)

        if context_start > 0:
            previous_boundary = self._find_left_boundary_for_description(chunk_text, max(0, context_start - 1))
            context_start = 0 if previous_boundary == -1 else previous_boundary + 1

        if context_end < len(chunk_text):
            context_end = self._find_right_boundary_for_description(chunk_text, context_end)

        return self._build_description_context_from_bounds(
            chunk_text,
            match_span,
            context_start,
            context_end,
        )

    # Use the full chunk as context for semantic-description prediction.
    def _extract_full_context_for_description(self, chunk_text, token_text, match_span=None):
        if match_span is None:
            match_span = _find_description_match_span(chunk_text, token_text)
        if match_span is None:
            return None
        return self._build_description_context_from_bounds(
            chunk_text,
            match_span,
            0,
            len(chunk_text),
        )

    # Select the configured context extraction strategy for description prediction.
    def _extract_prompt_context_for_description(self, chunk_text, token_text, match_span=None):
        mode = self.sem_description_prompt_context_mode
        if mode == "sentence":
            return self._extract_sentence_context_for_description(chunk_text, token_text, match_span=match_span)
        if mode == "sentence_neighbors":
            return self._extract_neighbor_sentence_context_for_description(chunk_text, token_text, match_span=match_span)
        if mode == "full_text":
            return self._extract_full_context_for_description(chunk_text, token_text, match_span=match_span)
        raise ValueError(
            f"Unsupported sem_description_prompt_context_mode={mode!r}. "
            "Use 'sentence', 'sentence_neighbors', or 'full_text'."
        )

    # Build a prompt asking what the token means in a specific chunk context.
    def _build_sem_node_description_prompt(self, chunk_text, token_text, match_span=None):
        context_info = self._extract_prompt_context_for_description(
            chunk_text,
            token_text,
            match_span=match_span,
        )
        if context_info is None:
            return None

        prompt_text = (
            f"Context: {context_info['context_text']}\n"
            f"Target word: {token_text.strip()}\n\n"
            f'Question: What does "{token_text.strip()}" mean in this context?'
        )
        return {
            **context_info,
            "prompt_text": prompt_text,
        }

    # Build a full-chunk fallback prompt when span context cannot be found.
    def _build_fallback_sem_description_prompt(self, chunk_text, token_text):
        context_text = chunk_text.strip()
        if not context_text:
            return None
        prompt_text = (
            f"Context: {context_text}\n"
            f"Target word: {token_text.strip()}\n\n"
            f'Question: What does "{token_text.strip()}" mean in this context?'
        )
        return {
            "context_text": context_text,
            "matched_text": token_text.strip(),
            "local_span": None,
            "prompt_text": prompt_text,
        }

    # Load and cache Wikidata definition candidates for a semantic node token.
    # Cache key is token_text alone: once a token has been looked up (with any
    # parameter values), later calls return the cached bank without re-querying
    # Wikidata or consulting the arguments again.
    def _load_sem_description_candidate_bank(self, sem_node, candidate_bank_cache):
        token_text = sem_node.token_node.token_text
        cached_candidate_bank = candidate_bank_cache.get(token_text)
        if cached_candidate_bank is not None:
            return cached_candidate_bank

        max_candidate_count = max(1, len(sem_node.token_node.sem_node_list) + 1)
        max_candidate_count = max(
            max_candidate_count,
            self.min_description_candidates,
        )
        use_llm_filter = bool(getattr(self, "use_llm_candidate_filter", False))
        if use_llm_filter:
            max_candidate_count = self.llm_candidate_filter_candidate_limit
        filter_start = time.perf_counter() if use_llm_filter else None
        try:
            candidates_df, definition_column = load_wikidata_definition_candidates(
                token_text,
                use_detailed_description=self.sem_description_use_detailed_description,
                limit=(
                    self.llm_candidate_filter_candidate_limit
                    if use_llm_filter
                    else self.sem_description_candidate_limit
                ),
                target_candidate_count=max_candidate_count,
                use_llm_filter=use_llm_filter,
                llm_filter=(
                    self._get_llm_candidate_filter()
                    if use_llm_filter
                    else None
                ),
                llm_filter_use_api=self.llm_candidate_filter_use_api,
                llm_filter_use_cache=False,
                llm_filter_write_cache=False,
            )
        except ValueError as exc:
            candidate_bank_cache[token_text] = []
            self._log_wikidata_no_result(
                token_text,
                "sem_description",
                getattr(exc, "empty_reason", "no_candidate_definitions"),
            )
            return []
        finally:
            if filter_start is not None:
                self.llm_candidate_filter_total_wall_time += time.perf_counter() - filter_start

        candidate_bank = build_wikidata_candidate_bank(candidates_df, definition_column=definition_column)
        candidate_bank_cache[token_text] = candidate_bank
        llm_metadata = None
        if use_llm_filter and not candidates_df.empty and "llm_filter_metadata" in candidates_df:
            llm_metadata = candidates_df["llm_filter_metadata"].iloc[0]
            self._record_llm_candidate_filter_usage(llm_metadata)
            self.llm_candidate_filter_result_cache[token_text] = {
                "candidate_bank": candidate_bank,
                "definitions": [
                    {
                        "entity_id": candidate.get("entity_id"),
                        "label": candidate.get("label"),
                        "description": candidate.get("description"),
                        "definition": candidate.get("definition"),
                    }
                    for candidate in candidate_bank
                ],
                "metadata": llm_metadata,
            }
        if candidate_bank:
            self._log_sem_description_operation(
                (
                    "llm_filtered_wikidata_candidate_lookup"
                    if use_llm_filter
                    else "wikidata_candidate_lookup"
                ),
                token_text=token_text,
                candidate_limit=(
                    self.llm_candidate_filter_candidate_limit
                    if use_llm_filter
                    else self.sem_description_candidate_limit
                ),
                max_candidate_count=max_candidate_count,
                candidate_count=len(candidate_bank),
                llm_token_usage=(llm_metadata or {}).get("token_usage") if llm_metadata else None,
            )
            return candidate_bank

        self._log_wikidata_no_result(
            token_text,
            "sem_description",
            "empty_candidate_bank",
        )
        return []

    # Build the fft_dataset_samples payload sent to the combined merge LLM call.
    def _build_fft_samples_payload(self, prepared_samples):
        fft_samples = []
        for sample_index, _span_occurrence, chunk_node, prompt_info in prepared_samples:
            doc_node = getattr(chunk_node, "doc_node", None)
            title = getattr(doc_node, "doc_name", None) if doc_node is not None else None
            fft_samples.append(
                {
                    "sample_order": sample_index,
                    "title": title,
                    "matched_text": prompt_info.get("matched_text") or "",
                    "context_text": prompt_info.get("context_text") or "",
                }
            )
        return fft_samples

    # Merge candidate senses and judge FFT samples in a single LLM call.
    # Used when use_llm_candidate_filter is enabled. Caches the merged candidate
    # bank (by token) and the sample_order -> merged-description map so the split
    # path reuses both without issuing a second LLM call. Mirrors the combined
    # prompt of wikidata_llm_candidate_merge_experiment.ipynb.
    def _run_combined_merge_for_samples(self, sem_node, fft_samples):
        token_text = sem_node.token_node.token_text
        cached_map = self._combined_merge_sample_sense_descriptions.get(token_text)
        if cached_map is not None:
            return (
                self._sem_description_candidate_bank_cache.get(token_text, []),
                cached_map,
            )

        max_candidate_count = self.llm_candidate_filter_candidate_limit
        filter_start = time.perf_counter()
        try:
            candidates_df, definition_column = load_wikidata_definition_candidates(
                token_text,
                use_detailed_description=self.sem_description_use_detailed_description,
                limit=self.llm_candidate_filter_candidate_limit,
                target_candidate_count=max_candidate_count,
                use_llm_filter=True,
                llm_filter=self._get_llm_candidate_filter(),
                llm_filter_use_api=self.llm_candidate_filter_use_api,
                fft_samples=fft_samples,
            )
        except ValueError as exc:
            self._sem_description_candidate_bank_cache[token_text] = []
            self._combined_merge_sample_sense_descriptions[token_text] = {}
            self._log_wikidata_no_result(
                token_text,
                "sem_description",
                getattr(exc, "empty_reason", "no_candidate_definitions"),
            )
            return [], {}
        finally:
            self.llm_candidate_filter_total_wall_time += time.perf_counter() - filter_start

        candidate_bank = build_wikidata_candidate_bank(
            candidates_df, definition_column=definition_column
        )
        self._sem_description_candidate_bank_cache[token_text] = candidate_bank

        llm_metadata = None
        sample_sense_descriptions = {}
        if not candidates_df.empty and "llm_filter_metadata" in candidates_df:
            llm_metadata = candidates_df["llm_filter_metadata"].iloc[0]
            self._record_llm_candidate_filter_usage(llm_metadata)
            raw_map = (llm_metadata or {}).get("sample_sense_descriptions") or {}
            for order, description in raw_map.items():
                try:
                    sample_sense_descriptions[int(order)] = description
                except (TypeError, ValueError):
                    continue
            self.llm_candidate_filter_result_cache[token_text] = {
                "candidate_bank": candidate_bank,
                "definitions": [
                    {
                        "entity_id": candidate.get("entity_id"),
                        "label": candidate.get("label"),
                        "description": candidate.get("description"),
                        "definition": candidate.get("definition"),
                    }
                    for candidate in candidate_bank
                ],
                "metadata": llm_metadata,
            }
        self._combined_merge_sample_sense_descriptions[token_text] = sample_sense_descriptions

        if candidate_bank:
            self._log_sem_description_operation(
                "combined_candidate_merge_with_samples",
                token_text=token_text,
                candidate_limit=self.llm_candidate_filter_candidate_limit,
                candidate_count=len(candidate_bank),
                fft_sample_count=len(fft_samples or []),
                matched_sample_count=len(sample_sense_descriptions),
                llm_token_usage=(llm_metadata or {}).get("token_usage") if llm_metadata else None,
            )
        else:
            self._log_wikidata_no_result(
                token_text,
                "sem_description",
                "empty_candidate_bank",
            )
        return candidate_bank, sample_sense_descriptions

    # Predict a semantic node description by scoring sampled contexts against candidates.
    def _predict_sem_description_from_samples(
        self,
        sem_node,
        model,
        candidate_bank_cache,
        max_samples=10,
    ):
        token_text = sem_node.token_node.token_text
        combined = bool(getattr(self, "use_llm_candidate_filter", False))

        description_vote_map = {}
        sample_prediction_records = []
        sample_span_occurrences = self._sample_sem_span_occurrences(sem_node, max_samples=max_samples)
        prepared_samples = []
        for sample_index, span_occurrence in enumerate(sample_span_occurrences, start=1):
            chunk_node = span_occurrence.chunk_node
            prompt_info = self._build_sem_node_description_prompt(
                chunk_node.chunk_text,
                token_text,
                match_span=span_occurrence.get_span_tuple(),
            )
            if prompt_info is None:
                prompt_info = self._build_fallback_sem_description_prompt(
                    chunk_node.chunk_text,
                    token_text,
                )
            if prompt_info is None:
                continue
            prepared_samples.append((sample_index, span_occurrence, chunk_node, prompt_info))

        if combined:
            # A single LLM call merges the candidate senses and judges each FFT
            # sample. Samples the LLM marks unsupported/ambiguous get no candidate
            # here, so they drop out of the consensus vote and are later handled by
            # the same embedding-based path as ordinary non-sampled occurrences.
            fft_samples = self._build_fft_samples_payload(prepared_samples)
            candidate_bank, sample_sense_descriptions = self._run_combined_merge_for_samples(
                sem_node, fft_samples
            )
            if not candidate_bank:
                self._log_wikidata_no_result(
                    token_text,
                    "sem_description",
                    "empty_candidate_bank",
                )
                return None
            description_to_candidate = {
                candidate["description"]: candidate for candidate in candidate_bank
            }
            top_candidates = []
            for sample_index, _span_occurrence, _chunk_node, _prompt_info in prepared_samples:
                description = sample_sense_descriptions.get(sample_index)
                candidate = description_to_candidate.get(description) if description else None
                if candidate is not None:
                    candidate = {
                        **candidate,
                        "score": None,
                        "prediction_method": "llm_candidate_merge",
                    }
                top_candidates.append(candidate)
        else:
            candidate_bank = self._load_sem_description_candidate_bank(sem_node, candidate_bank_cache)
            if not candidate_bank:
                self._log_wikidata_no_result(
                    token_text,
                    "sem_description",
                    "empty_candidate_bank",
                )
                return None
            if getattr(self, "use_llm_semantic_labeler", False):
                top_candidates = self._predict_descriptions_from_prompt_infos_with_llm(
                    token_text,
                    [prompt_info for _sample_index, _span_occurrence, _chunk_node, prompt_info in prepared_samples],
                    candidate_bank,
                )
            else:
                top_candidates = []
                for _sample_index, _span_occurrence, _chunk_node, prompt_info in prepared_samples:
                    pairs = [(prompt_info["prompt_text"], candidate["hypothesis"]) for candidate in candidate_bank]
                    raw_scores = model.predict(
                        pairs,
                        batch_size=min(self.sem_description_batch_size, len(pairs)),
                        show_progress_bar=False,
                    )
                    scores = extract_cross_encoder_scores(raw_scores, model)
                    ranked_candidates = sorted(
                        [
                            {
                                **candidate,
                                "score": float(score),
                            }
                            for candidate, score in zip(candidate_bank, scores)
                        ],
                        key=lambda item: item["score"],
                        reverse=True,
                    )
                    top_candidates.append(ranked_candidates[0] if ranked_candidates else None)

        for (sample_index, span_occurrence, chunk_node, prompt_info), top_candidate in zip(
            prepared_samples,
            top_candidates,
        ):
            if top_candidate is not None:
                description = top_candidate["description"]
                state = description_vote_map.setdefault(
                    description,
                    {
                        "candidate": top_candidate,
                        "description": description,
                        "count": 0,
                        "score_sum": 0.0,
                        "score_count": 0,
                    },
                )
                raw_score = top_candidate.get("score")
                score = 0.0 if raw_score is None else self._safe_float(raw_score)
                state["count"] += 1
                state["score_sum"] += score
                state["score_count"] += 1
                sample_prediction_records.append(
                    {
                        "span_occurrence": span_occurrence,
                        "sample_index": sample_index,
                        "chunk_node_id": chunk_node.chunk_node_id,
                        "span_start": span_occurrence.span_start,
                        "span_end": span_occurrence.span_end,
                        "context_text": prompt_info["context_text"],
                        "matched_text": prompt_info["matched_text"],
                        "predicted_entity_id": top_candidate["entity_id"],
                        "predicted_label": top_candidate["label"],
                        "predicted_description": top_candidate["description"],
                        "predicted_definition": top_candidate["definition"],
                        "prediction_score": top_candidate.get("score"),
                        "prediction_method": top_candidate.get("prediction_method", "cross_encoder"),
                        "llm_reason": top_candidate.get("llm_reason"),
                        "llm_cache_hit": top_candidate.get("llm_cache_hit"),
                    }
                )

        if not sample_prediction_records or not description_vote_map:
            return None
        aggregated_candidates = sorted(
            (
                {
                    **state["candidate"],
                    "description": state["description"],
                    "count": int(state["count"]),
                    "score_sum": float(state["score_sum"]),
                    "score_count": int(state["score_count"]),
                    "score_mean": float(state["score_sum"] / state["score_count"]),
                }
                for state in description_vote_map.values()
                if state["score_count"] > 0
            ),
            key=lambda item: (item["count"], item["score_mean"], item["score_sum"]),
            reverse=True,
        )
        top_aggregated_candidate = aggregated_candidates[0]
        public_sample_prediction_logs = [
            {key: value for key, value in record.items() if key != "span_occurrence"}
            for record in sample_prediction_records
        ]
        return {
            "description": top_aggregated_candidate["description"],
            "predicted_entity_id": top_aggregated_candidate["entity_id"],
            "predicted_label": top_aggregated_candidate["label"],
            "predicted_definition": top_aggregated_candidate["definition"],
            "prediction_score_mean": top_aggregated_candidate["score_mean"],
            "sample_predictions": public_sample_prediction_logs,
            "sample_prediction_records": sample_prediction_records,
            "sample_count": len(sample_prediction_records),
            "description_counts": {
                item["description"]: item["count"]
                for item in aggregated_candidates
            },
            "top_description_count": top_aggregated_candidate["count"],
            "top_description_ratio": (
                top_aggregated_candidate["count"] / len(sample_prediction_records)
            ),
        }

    # Run LLM semantic judgments for prepared prompt/context records in shared prompts.
    def _predict_descriptions_from_prompt_infos_with_llm(
        self,
        token_text,
        prompt_infos,
        candidate_bank,
    ):
        from llm_semantic_labeler import choose_wikidata_candidates_with_llm_batch

        if not prompt_infos:
            return []

        client = self._get_llm_semantic_label_client()
        total_before = getattr(client, "total_tokens", 0)
        records = [
            {
                "span_text": token_text,
                "context_text": prompt_info["context_text"],
                "matched_text": prompt_info.get("matched_text") or token_text,
                "local_span": prompt_info.get("local_span"),
            }
            for prompt_info in prompt_infos
        ]
        results = choose_wikidata_candidates_with_llm_batch(
            records=records,
            candidate_bank=candidate_bank,
            cache_path=self.llm_semantic_labeler_cache_path,
            context_word_window=self.llm_semantic_labeler_context_word_window,
            client=client,
            max_tokens=self.llm_semantic_labeler_max_tokens,
            max_batch_size=self.llm_semantic_labeler_batch_size,
        )
        total_after = getattr(client, "total_tokens", total_before)
        self.llm_semantic_label_total_tokens += max(0, int(total_after - total_before))
        for result in results:
            if result.get("cache_hit"):
                self.llm_semantic_label_cache_hits += 1
            else:
                self.llm_semantic_label_cache_misses += 1

        candidates = []
        for result in results:
            candidate = dict(result["selected_candidate"])
            candidate["score"] = None
            candidate["prediction_method"] = "llm_semantic_label"
            candidate["llm_reason"] = result.get("reason")
            candidate["llm_cache_hit"] = bool(result.get("cache_hit"))
            candidates.append(candidate)
        return candidates

    # Run one LLM semantic judgment for one prompt/context.
    def _predict_description_from_prompt_with_llm(self, token_text, prompt_info, candidate_bank):
        candidates = self._predict_descriptions_from_prompt_infos_with_llm(
            token_text,
            [prompt_info],
            candidate_bank,
        )
        return candidates[0] if candidates else None

    # Run cross-encoder prediction for a single span occurrence; return the top candidate or None.
    def _predict_description_for_span_occurrence(
        self,
        token_text,
        span_occurrence,
        candidate_bank,
        model,
    ):
        chunk_node = span_occurrence.chunk_node
        prompt_info = self._build_sem_node_description_prompt(
            chunk_node.chunk_text,
            token_text,
            match_span=span_occurrence.get_span_tuple(),
        )
        if prompt_info is None:
            prompt_info = self._build_fallback_sem_description_prompt(
                chunk_node.chunk_text,
                token_text,
            )
        if prompt_info is None:
            return None
        pairs = [
            (prompt_info["prompt_text"], candidate["hypothesis"])
            for candidate in candidate_bank
        ]
        raw_scores = model.predict(
            pairs,
            batch_size=min(self.sem_description_batch_size, len(pairs)),
            show_progress_bar=False,
        )
        scores = extract_cross_encoder_scores(raw_scores, model)
        ranked_candidates = sorted(
            [
                {**candidate, "score": float(score)}
                for candidate, score in zip(candidate_bank, scores)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        if not ranked_candidates:
            return None
        return ranked_candidates[0]

    # Run single-prompt LLM prediction for a single span occurrence.
    def _predict_description_for_span_occurrence_with_llm(
        self,
        token_text,
        span_occurrence,
        candidate_bank,
    ):
        chunk_node = span_occurrence.chunk_node
        prompt_info = self._build_sem_node_description_prompt(
            chunk_node.chunk_text,
            token_text,
            match_span=span_occurrence.get_span_tuple(),
        )
        if prompt_info is None:
            prompt_info = self._build_fallback_sem_description_prompt(
                chunk_node.chunk_text,
                token_text,
            )
        if prompt_info is None:
            return None
        return self._predict_description_from_prompt_with_llm(
            token_text,
            prompt_info,
            candidate_bank,
        )

    # Pick the medoid index from a list of embedding tensors using cosine distance.
    def _compute_medoid_index(self, embeds):
        if not embeds:
            return None
        if len(embeds) == 1:
            return 0
        stacked = torch.stack([F.normalize(embed, p=2, dim=0) for embed in embeds])
        sims = stacked @ stacked.T
        dists = (1.0 - sims).clamp(min=0.0)
        total = dists.sum(dim=1)
        return int(torch.argmin(total).item())

    # Return nearest text-embedding indices by cosine distance inside one FFT cluster.
    def _nearest_cluster_text_embedding_indices(self, cluster_text_embeddings, target_idx, count):
        if count <= 0 or len(cluster_text_embeddings) <= 1:
            return []
        target_embed = cluster_text_embeddings[target_idx].embed
        target_vec = F.normalize(target_embed, p=2, dim=0).unsqueeze(0)
        embed_stack = torch.stack(
            [F.normalize(te.embed, p=2, dim=0) for te in cluster_text_embeddings]
        )
        dists = (1.0 - (target_vec @ embed_stack.T).squeeze(0)).clamp(min=0.0)
        dists[target_idx] = float("inf")
        neighbor_count = min(int(count), len(cluster_text_embeddings) - 1)
        return [int(idx.item()) for idx in torch.argsort(dists)[:neighbor_count]]

    # Create a new sem node initialized from a subset of cluster text embeddings (build-time path).
    def _create_sem_node_from_cluster_subset(self, token_node, description, text_embeddings):
        new_sem_node = self.create_sem_node(token_node)
        new_sem_node.description = description
        new_sem_node.chunk_node_list = [te.chunk_node for te in text_embeddings]
        new_sem_node.span_occurrences = [te.to_span_occurrence() for te in text_embeddings]
        new_sem_node.chunk_node_embed = [te.embed for te in text_embeddings]
        new_sem_node.embed = self._store_sem_embed(average_embeds(new_sem_node.chunk_node_embed))
        self._initialize_sem_retained_text_embeddings(new_sem_node, text_embeddings)
        new_sem_node.chunk_edge_weight = sem_embed_sim(new_sem_node).cpu().tolist()
        new_sem_node.chunk_node_embed.clear()
        return new_sem_node

    # Build a recorder snapshot of the cluster going into description assignment.
    # Captures every text-embedding's span/context/embedding, the FFT-selected sample
    # indices, and identifies the cluster by token / sem_node id. Designed for offline
    # analysis notebooks; never called when `_fft_call_recorder` is unset.
    def _build_fft_recorder_snapshot(self, sem_node, cluster_text_embeddings):
        token_node = sem_node.token_node
        token_text = token_node.token_text
        all_text_embeddings = []
        embeds_for_fft = []
        for idx, te in enumerate(cluster_text_embeddings):
            embed = te.embed
            embed_np = (
                embed.detach().cpu().float().numpy()
                if torch.is_tensor(embed)
                else np.asarray(embed, dtype=np.float32)
            )
            chunk_node = te.chunk_node
            span_start = te.span_start
            span_end = te.span_end
            span_text = te.span_text
            if span_text is None and span_start is not None and span_end is not None:
                span_text = chunk_node.chunk_text[span_start:span_end]
            context_text = None
            if span_start is not None and span_end is not None:
                ctx_lo = max(0, span_start - 120)
                ctx_hi = min(len(chunk_node.chunk_text), span_end + 120)
                context_text = chunk_node.chunk_text[ctx_lo:ctx_hi]
            all_text_embeddings.append({
                "te_index": idx,
                "chunk_node_id": chunk_node.chunk_node_id,
                "span_start": span_start,
                "span_end": span_end,
                "span_text": span_text,
                "context_text": context_text,
                "embedding": embed_np,
            })
            embeds_for_fft.append(embed_np)

        sampled_indices = []
        if embeds_for_fft and len(embeds_for_fft) > self.fft_max_samples:
            sampled_indices = list(self._run_farthest_first_traversal(embeds_for_fft, self.fft_max_samples))
        elif embeds_for_fft:
            sampled_indices = list(range(len(embeds_for_fft)))

        candidate_bank_snapshot = []
        if self.sem_description_model is not None or getattr(self, "use_llm_semantic_labeler", False):
            try:
                bank = self._load_sem_description_candidate_bank(
                    sem_node, self._sem_description_candidate_bank_cache
                )
            except Exception:
                bank = []
            for cand in bank or []:
                candidate_bank_snapshot.append({
                    "entity_id": cand.get("entity_id"),
                    "label": cand.get("label"),
                    "description": cand.get("description"),
                    "definition": cand.get("definition"),
                    "definition_source": cand.get("definition_source"),
                })

        return {
            "token_text": token_text,
            "token_node_id": getattr(token_node, "token_node_id", None),
            "sem_node_id": sem_node.sem_node_id,
            "fft_max_samples": self.fft_max_samples,
            "consensus_ratio_threshold": self.consensus_ratio_threshold,
            "fft_d1_d2_ratio_threshold": self.fft_d1_d2_ratio_threshold,
            "fft_knn_check_k": self.fft_knn_check_k,
            "fft_danger_neighbor_m": self.fft_danger_neighbor_m,
            "all_text_embeddings": all_text_embeddings,
            "sampled_indices": sampled_indices,
            "candidate_bank": candidate_bank_snapshot,
        }

    # Assign a description to a freshly-built sem node; may split it into multiple sem nodes.
    #
    # Called before `chunk_node_embed.clear()` so that full cluster embeddings are still
    # available for medoid / d1-d2 based sample classification.
    # Returns the list of sem nodes that should be attached to `token_node.sem_node_list`.
    # Each returned sem node has its `chunk_node_embed` cleared.
    def _assign_sem_description_on_build(self, sem_node, cluster_text_embeddings):
        if getattr(self, "_anchor_method_spec", None) is not None:
            return self._assign_sem_description_anchor_propagation(
                sem_node, cluster_text_embeddings
            )
        token_node = sem_node.token_node
        recorder = getattr(self, "_fft_call_recorder", None)
        record_snapshot = (
            self._build_fft_recorder_snapshot(sem_node, cluster_text_embeddings)
            if recorder is not None
            else None
        )

        def _emit(path_taken, final_assignments=None, prediction_result=None, extra=None):
            if recorder is None or record_snapshot is None:
                return
            payload = dict(record_snapshot)
            payload["path_taken"] = path_taken
            payload["final_assignments"] = final_assignments or []
            if prediction_result is not None:
                payload["sample_predictions"] = prediction_result.get("sample_predictions", [])
                payload["prediction_top_description"] = prediction_result.get("description")
                payload["top_description_ratio"] = prediction_result.get("top_description_ratio")
                payload["top_description_count"] = prediction_result.get("top_description_count")
                payload["sample_count"] = prediction_result.get("sample_count")
                payload["description_counts"] = prediction_result.get("description_counts")
            if extra:
                payload.update(extra)
            try:
                recorder(payload)
            except Exception as exc:
                print(f"[fft_recorder] hook raised {exc!r}; skipping record")

        # In combined-merge mode (use_llm_candidate_filter) the candidate bank and
        # the per-FFT-sample sense judgments come from one LLM call issued inside
        # _predict_sem_description_from_samples, so there is no separate sample-less
        # candidate-bank load here. A single merged sense collapses naturally via
        # the consensus path below.

        if (
            self.sem_description_model is None
            and not getattr(self, "use_llm_semantic_labeler", False)
            and not getattr(self, "use_llm_candidate_filter", False)
        ):
            _emit("no_model")
            sem_node.chunk_node_embed.clear()
            return [sem_node]

        prediction_result = self._predict_sem_description_from_samples(
            sem_node,
            model=self.sem_description_model,
            candidate_bank_cache=self._sem_description_candidate_bank_cache,
            max_samples=self.fft_max_samples,
        )
        if prediction_result is None:
            _emit("no_prediction")
            sem_node.chunk_node_embed.clear()
            return [sem_node]

        if (
            prediction_result["sample_count"] > 0
            and prediction_result["top_description_ratio"] >= self.consensus_ratio_threshold
        ):
            predicted_description = prediction_result["description"]
            sem_node.description = predicted_description
            self.predicted_sem_description_logs.append(
                {
                    "sem_node_id": sem_node.sem_node_id,
                    "token_text": token_node.token_text,
                    "description": predicted_description,
                    "predicted_entity_id": prediction_result.get("predicted_entity_id"),
                    "predicted_label": prediction_result.get("predicted_label"),
                    "predicted_definition": prediction_result.get("predicted_definition"),
                    "prediction_score_mean": prediction_result.get("prediction_score_mean"),
                    "chunk_count": len(sem_node.chunk_node_list),
                    "sample_count": prediction_result.get("sample_count"),
                    "top_description_count": prediction_result.get("top_description_count"),
                    "top_description_ratio": prediction_result.get("top_description_ratio"),
                    "sample_predictions": prediction_result["sample_predictions"],
                }
            )
            self._log_sem_description_operation(
                "assign_description_by_consensus",
                token_text=token_node.token_text,
                sem_node_id=sem_node.sem_node_id,
                description=predicted_description,
                sample_count=prediction_result.get("sample_count"),
                top_description_count=prediction_result.get("top_description_count"),
                top_description_ratio=prediction_result.get("top_description_ratio"),
            )
            consensus_assignments = [
                {
                    "te_index": idx,
                    "assigned_description": predicted_description,
                    "provenance": "consensus",
                }
                for idx in range(len(cluster_text_embeddings))
            ]
            _emit("consensus", final_assignments=consensus_assignments, prediction_result=prediction_result)
            sem_node.chunk_node_embed.clear()
            return [sem_node]

        # Consensus failed: run d1/d2 medoid split using full cluster embeddings.
        candidate_bank = self._load_sem_description_candidate_bank(
            sem_node,
            self._sem_description_candidate_bank_cache,
        )

        sample_records = prediction_result["sample_prediction_records"]
        description_groups = defaultdict(list)
        for record in sample_records:
            description_groups[record["predicted_description"]].append(record)

        valid_classes = {
            description: records
            for description, records in description_groups.items()
            if len(records) >= 2
        }
        if len(valid_classes) < 2:
            fallback_description = prediction_result["description"]
            sem_node.description = fallback_description
            self.predicted_sem_description_logs.append(
                {
                    "sem_node_id": sem_node.sem_node_id,
                    "token_text": token_node.token_text,
                    "description": fallback_description,
                    "predicted_entity_id": prediction_result.get("predicted_entity_id"),
                    "predicted_label": prediction_result.get("predicted_label"),
                    "predicted_definition": prediction_result.get("predicted_definition"),
                    "prediction_score_mean": prediction_result.get("prediction_score_mean"),
                    "chunk_count": len(sem_node.chunk_node_list),
                    "sample_count": prediction_result.get("sample_count"),
                    "top_description_count": prediction_result.get("top_description_count"),
                    "top_description_ratio": prediction_result.get("top_description_ratio"),
                    "sample_predictions": prediction_result["sample_predictions"],
                }
            )
            self._log_sem_description_operation(
                "assign_description_by_fallback_top_vote",
                token_text=token_node.token_text,
                sem_node_id=sem_node.sem_node_id,
                description=fallback_description,
                sample_count=prediction_result.get("sample_count"),
                top_description_count=prediction_result.get("top_description_count"),
                top_description_ratio=prediction_result.get("top_description_ratio"),
                valid_class_count=len(valid_classes),
                description_counts=prediction_result.get("description_counts"),
            )
            fallback_assignments = [
                {
                    "te_index": idx,
                    "assigned_description": fallback_description,
                    "provenance": "fallback_top_vote",
                }
                for idx in range(len(cluster_text_embeddings))
            ]
            _emit(
                "fallback_top_vote",
                final_assignments=fallback_assignments,
                prediction_result=prediction_result,
                extra={
                    "valid_class_count": len(valid_classes),
                    "description_counts": prediction_result.get("description_counts"),
                },
            )
            sem_node.chunk_node_embed.clear()
            return [sem_node]

        embedding_lookup = {
            self._get_span_occurrence_key(te.to_span_occurrence()): te
            for te in cluster_text_embeddings
        }
        te_to_index = {id(te): idx for idx, te in enumerate(cluster_text_embeddings)}
        prediction_cache = {
            self._get_span_occurrence_key(record["span_occurrence"]): record
            for record in sample_records
        }

        def _prediction_record_from_candidate(te, top_candidate):
            span_occurrence = te.to_span_occurrence()
            return {
                "span_occurrence": span_occurrence,
                "chunk_node_id": te.chunk_node.chunk_node_id,
                "span_start": te.span_start,
                "span_end": te.span_end,
                "context_text": None,
                "matched_text": te.span_text,
                "predicted_entity_id": top_candidate.get("entity_id"),
                "predicted_label": top_candidate.get("label"),
                "predicted_description": top_candidate.get("description"),
                "predicted_definition": top_candidate.get("definition"),
                "prediction_score": top_candidate.get("score"),
                "prediction_method": top_candidate.get("prediction_method", "cross_encoder"),
                "llm_reason": top_candidate.get("llm_reason"),
                "llm_cache_hit": top_candidate.get("llm_cache_hit"),
            }

        def _predict_missing_records_for_tes(text_embeddings):
            pending = []
            pending_keys = set()
            for te in text_embeddings:
                key = self._get_span_occurrence_key(te.to_span_occurrence())
                if key in prediction_cache or key in pending_keys:
                    continue
                pending.append((key, te))
                pending_keys.add(key)
            if not pending:
                return

            if getattr(self, "use_llm_semantic_labeler", False):
                prompt_infos = []
                llm_pending = []
                for key, te in pending:
                    span_occurrence = te.to_span_occurrence()
                    chunk_node = span_occurrence.chunk_node
                    prompt_info = self._build_sem_node_description_prompt(
                        chunk_node.chunk_text,
                        token_node.token_text,
                        match_span=span_occurrence.get_span_tuple(),
                    )
                    if prompt_info is None:
                        prompt_info = self._build_fallback_sem_description_prompt(
                            chunk_node.chunk_text,
                            token_node.token_text,
                        )
                    if prompt_info is None:
                        continue
                    prompt_infos.append(prompt_info)
                    llm_pending.append((key, te))
                top_candidates = self._predict_descriptions_from_prompt_infos_with_llm(
                    token_node.token_text,
                    prompt_infos,
                    candidate_bank,
                )
                for (key, te), top_candidate in zip(llm_pending, top_candidates):
                    if top_candidate is None:
                        continue
                    prediction_cache[key] = _prediction_record_from_candidate(te, top_candidate)
                return

            for key, te in pending:
                top_candidate = self._predict_description_for_span_occurrence(
                    token_node.token_text,
                    te.to_span_occurrence(),
                    candidate_bank,
                    self.sem_description_model,
                )
                if top_candidate is not None:
                    prediction_cache[key] = _prediction_record_from_candidate(te, top_candidate)

        def _get_or_predict_record(te):
            key = self._get_span_occurrence_key(te.to_span_occurrence())
            cached = prediction_cache.get(key)
            if cached is not None:
                return cached
            _predict_missing_records_for_tes([te])
            return prediction_cache.get(key)

        final_groups = defaultdict(list)
        class_medoid_embeds = {}
        sample_assignments = []
        danger_indices = set()
        safety_checks = []
        safe_class_tes = defaultdict(list)

        if getattr(self, "use_llm_semantic_labeler", False):
            neighbor_tes_to_judge = []
            for records in valid_classes.values():
                for record in records:
                    key = self._get_span_occurrence_key(record["span_occurrence"])
                    te = embedding_lookup.get(key)
                    if te is None:
                        continue
                    seed_idx = te_to_index.get(id(te))
                    if seed_idx is None:
                        continue
                    for neighbor_idx in self._nearest_cluster_text_embedding_indices(
                        cluster_text_embeddings,
                        seed_idx,
                        self.fft_knn_check_k,
                    ):
                        neighbor_tes_to_judge.append(cluster_text_embeddings[neighbor_idx])
            _predict_missing_records_for_tes(neighbor_tes_to_judge)

        for description, records in valid_classes.items():
            for record in records:
                key = self._get_span_occurrence_key(record["span_occurrence"])
                te = embedding_lookup.get(key)
                if te is None:
                    continue
                seed_idx = te_to_index.get(id(te))
                if seed_idx is None:
                    continue
                neighbor_indices = self._nearest_cluster_text_embedding_indices(
                    cluster_text_embeddings,
                    seed_idx,
                    self.fft_knn_check_k,
                )
                neighbor_descriptions = []
                is_safe = bool(neighbor_indices)
                for neighbor_idx in neighbor_indices:
                    neighbor_te = cluster_text_embeddings[neighbor_idx]
                    neighbor_record = _get_or_predict_record(neighbor_te)
                    neighbor_description = (
                        neighbor_record.get("predicted_description")
                        if neighbor_record is not None
                        else None
                    )
                    neighbor_descriptions.append(neighbor_description)
                    if neighbor_description != description:
                        is_safe = False

                danger_neighbor_indices = []
                if is_safe:
                    safe_class_tes[description].append(te)
                else:
                    danger_indices.add(seed_idx)
                    danger_neighbor_indices = self._nearest_cluster_text_embedding_indices(
                        cluster_text_embeddings,
                        seed_idx,
                        self.fft_danger_neighbor_m,
                    )
                    danger_indices.update(danger_neighbor_indices)

                safety_checks.append(
                    {
                        "seed_index": seed_idx,
                        "description": description,
                        "is_safe": bool(is_safe),
                        "knn_neighbor_indices": neighbor_indices,
                        "knn_neighbor_descriptions": neighbor_descriptions,
                        "danger_neighbor_indices": danger_neighbor_indices,
                    }
                )

        safe_class_tes = {
            description: [
                te for te in tes
                if te_to_index.get(id(te)) not in danger_indices
            ]
            for description, tes in safe_class_tes.items()
        }
        safe_class_tes = {
            description: tes
            for description, tes in safe_class_tes.items()
            if tes
        }

        assigned_keys = set()
        for description, class_tes in safe_class_tes.items():
            medoid_idx = self._compute_medoid_index([te.embed for te in class_tes])
            class_medoid_embeds[description] = class_tes[medoid_idx].embed
            final_groups[description].extend(class_tes)
            for te in class_tes:
                assigned_keys.add(self._get_span_occurrence_key(te.to_span_occurrence()))
                sample_assignments.append({
                    "provenance": "class_seed",
                    "description": description,
                    "text_embedding": te,
                    "safety": "safe",
                })

        medoid_descriptions = list(class_medoid_embeds.keys())
        medoid_stack = None
        if medoid_descriptions:
            medoid_stack = torch.stack(
                [F.normalize(class_medoid_embeds[d], p=2, dim=0) for d in medoid_descriptions]
            )

        unclassified_records = []
        for te in cluster_text_embeddings:
            key = self._get_span_occurrence_key(te.to_span_occurrence())
            if key in assigned_keys:
                continue
            te_idx = te_to_index.get(id(te))
            cached_record = prediction_cache.get(key)
            singleton_description = (
                cached_record.get("predicted_description")
                if cached_record is not None
                else None
            )
            is_danger_zone = te_idx in danger_indices
            unclassified_records.append((te, singleton_description, is_danger_zone))

        model_judgment_records = []
        if len(medoid_descriptions) == 1:
            only_desc = medoid_descriptions[0]
            for te, singleton_description, is_danger_zone in unclassified_records:
                if is_danger_zone:
                    model_judgment_records.append((te, singleton_description, None, None, None))
                    continue
                final_groups[only_desc].append(te)
                sample_assignments.append({
                    "provenance": "single_safe_class",
                    "description": only_desc,
                    "text_embedding": te,
                    "singleton_class_description": singleton_description,
                })
        elif len(medoid_descriptions) >= 2:
            for te, singleton_description, is_danger_zone in unclassified_records:
                if is_danger_zone:
                    model_judgment_records.append((te, singleton_description, None, None, None))
                    continue
                sample_vec = F.normalize(te.embed, p=2, dim=0).unsqueeze(0)
                sims = (sample_vec @ medoid_stack.T).squeeze(0)
                dists = (1.0 - sims).clamp(min=0.0)
                sorted_dists, sorted_idx = torch.sort(dists)
                d1 = float(sorted_dists[0].item())
                d2 = float(sorted_dists[1].item())
                ratio = (d1 / d2) if d2 > 0 else float("inf")
                if d2 > 0 and ratio <= self.fft_d1_d2_ratio_threshold:
                    chosen_desc = medoid_descriptions[int(sorted_idx[0].item())]
                    final_groups[chosen_desc].append(te)
                    sample_assignments.append({
                        "provenance": "d1_d2",
                        "description": chosen_desc,
                        "text_embedding": te,
                        "d1": d1,
                        "d2": d2,
                        "ratio": ratio,
                        "singleton_class_description": singleton_description,
                    })
                else:
                    model_judgment_records.append((te, singleton_description, d1, d2, ratio))
        else:
            for te, singleton_description, _is_danger_zone in unclassified_records:
                model_judgment_records.append((te, singleton_description, None, None, None))

        if getattr(self, "use_llm_semantic_labeler", False):
            _predict_missing_records_for_tes(
                [te for te, _singleton_description, _d1, _d2, _ratio in model_judgment_records]
            )

        for te, singleton_description, d1, d2, ratio in model_judgment_records:
            prediction_record = _get_or_predict_record(te)
            if prediction_record is None:
                if medoid_stack is None:
                    continue
                sample_vec = F.normalize(te.embed, p=2, dim=0).unsqueeze(0)
                sims = (sample_vec @ medoid_stack.T).squeeze(0)
                dists = (1.0 - sims).clamp(min=0.0)
                nearest_idx = int(torch.argmin(dists).item())
                chosen_desc = medoid_descriptions[nearest_idx]
                final_groups[chosen_desc].append(te)
                sample_assignments.append({
                    "provenance": "medoid_fallback",
                    "description": chosen_desc,
                    "text_embedding": te,
                    "nearest_medoid_distance": float(dists[nearest_idx].item()),
                    "d1": d1,
                    "d2": d2,
                    "ratio": ratio,
                    "singleton_class_description": singleton_description,
                })
            else:
                final_groups[prediction_record["predicted_description"]].append(te)
                sample_assignments.append({
                    "provenance": "model_judgment",
                    "description": prediction_record["predicted_description"],
                    "text_embedding": te,
                    "predicted_entity_id": prediction_record.get("predicted_entity_id"),
                    "predicted_label": prediction_record.get("predicted_label"),
                    "predicted_definition": prediction_record.get("predicted_definition"),
                    "score": prediction_record.get("prediction_score"),
                    "prediction_method": prediction_record.get("prediction_method", "cross_encoder"),
                    "llm_reason": prediction_record.get("llm_reason"),
                    "llm_cache_hit": prediction_record.get("llm_cache_hit"),
                    "d1": d1,
                    "d2": d2,
                    "ratio": ratio,
                    "singleton_class_description": singleton_description,
                })

        provenance_counts = Counter(entry["provenance"] for entry in sample_assignments)
        split_event_payload = {
            "token_text": token_node.token_text,
            "source_sem_node_id": sem_node.sem_node_id,
            "source_chunk_count": len(sem_node.chunk_node_list),
            "cluster_sample_count": len(cluster_text_embeddings),
            "initial_prediction_sample_count": len(sample_records),
            "valid_class_count": len(valid_classes),
            "safe_class_count": len(safe_class_tes),
            "safe_seed_count": sum(len(tes) for tes in safe_class_tes.values()),
            "danger_seed_count": sum(1 for item in safety_checks if not item["is_safe"]),
            "danger_zone_count": len(danger_indices),
            "singleton_class_count": len(description_groups) - len(valid_classes),
            "class_seed_sample_count": provenance_counts.get("class_seed", 0),
            "single_safe_class_assigned_count": provenance_counts.get("single_safe_class", 0),
            "d1_d2_assigned_count": provenance_counts.get("d1_d2", 0),
            "model_judgment_count": provenance_counts.get("model_judgment", 0),
            "medoid_fallback_count": provenance_counts.get("medoid_fallback", 0),
            "description_counts": {d: len(tes) for d, tes in final_groups.items()},
            "initial_class_sizes": {d: len(r) for d, r in description_groups.items()},
            "safe_class_sizes": {d: len(tes) for d, tes in safe_class_tes.items()},
            "fft_max_samples": self.fft_max_samples,
            "fft_d1_d2_ratio_threshold": self.fft_d1_d2_ratio_threshold,
            "fft_knn_check_k": self.fft_knn_check_k,
            "fft_danger_neighbor_m": self.fft_danger_neighbor_m,
            "safety_checks": safety_checks,
        }
        self._log_sem_description_operation(
            "split_sem_node_by_sample_labels",
            **split_event_payload,
        )
        self.deleted_merged_sem_logs.append(
            {
                "deleted_sem_node_id": sem_node.sem_node_id,
                "kept_sem_node_id": None,
                "token_text": token_node.token_text,
                "description": "split_by_sample_label",
                "deleted_chunk_count": len(sem_node.chunk_node_list),
            }
        )

        try:
            self.sem_nodes.remove(sem_node)
        except ValueError:
            pass
        sem_node.chunk_node_embed.clear()

        new_sem_nodes = []
        description_to_sem_node = {}
        for description, tes in final_groups.items():
            if not tes:
                continue
            new_node = self._create_sem_node_from_cluster_subset(
                token_node,
                description,
                tes,
            )
            new_sem_nodes.append(new_node)
            description_to_sem_node[description] = new_node
            self._log_sem_description_operation(
                "create_split_sem_node",
                token_text=token_node.token_text,
                source_sem_node_id=sem_node.sem_node_id,
                sem_node_id=new_node.sem_node_id,
                description=description,
                sample_count=len(tes),
            )

        for entry in sample_assignments:
            target_sem = description_to_sem_node.get(entry["description"])
            if target_sem is None:
                continue
            te = entry["text_embedding"]
            chunk_node = te.chunk_node
            log_payload = {
                "token_text": token_node.token_text,
                "source_sem_node_id": sem_node.sem_node_id,
                "target_sem_node_id": target_sem.sem_node_id,
                "description": entry["description"],
                "chunk_node_id": chunk_node.chunk_node_id,
                "span_start": te.span_start,
                "span_end": te.span_end,
                "span_text": te.span_text,
            }
            provenance = entry["provenance"]
            if provenance == "class_seed":
                event_type = "assign_sample_as_class_seed"
            elif provenance == "d1_d2":
                event_type = "assign_sample_by_d1_d2"
                log_payload["d1"] = entry["d1"]
                log_payload["d2"] = entry["d2"]
                log_payload["ratio"] = entry["ratio"]
                log_payload["singleton_class_description"] = entry["singleton_class_description"]
            elif provenance == "single_safe_class":
                event_type = "assign_sample_by_single_safe_class"
                log_payload["singleton_class_description"] = entry["singleton_class_description"]
            elif provenance == "model_judgment":
                event_type = "assign_sample_by_model_judgment"
                log_payload["predicted_entity_id"] = entry["predicted_entity_id"]
                log_payload["predicted_label"] = entry["predicted_label"]
                log_payload["predicted_definition"] = entry["predicted_definition"]
                log_payload["score"] = entry["score"]
                log_payload["d1"] = entry["d1"]
                log_payload["d2"] = entry["d2"]
                log_payload["ratio"] = entry["ratio"]
                log_payload["singleton_class_description"] = entry["singleton_class_description"]
            else:
                event_type = "assign_sample_by_medoid_fallback"
                log_payload["nearest_medoid_distance"] = entry["nearest_medoid_distance"]
                log_payload["d1"] = entry["d1"]
                log_payload["d2"] = entry["d2"]
                log_payload["ratio"] = entry["ratio"]
                log_payload["singleton_class_description"] = entry["singleton_class_description"]
            self._log_sem_description_operation(event_type, **log_payload)

        token_node.has_semantic = bool(new_sem_nodes)
        self._reset_sem_node_ids()

        if recorder is not None and record_snapshot is not None:
            te_to_index = {id(te): idx for idx, te in enumerate(cluster_text_embeddings)}
            split_assignments = []
            for entry in sample_assignments:
                te = entry["text_embedding"]
                te_idx = te_to_index.get(id(te))
                if te_idx is None:
                    continue
                split_assignments.append({
                    "te_index": te_idx,
                    "assigned_description": entry["description"],
                    "provenance": entry["provenance"],
                })
            _emit(
                "split",
                final_assignments=split_assignments,
                prediction_result=prediction_result,
                extra={"split_event": split_event_payload},
            )

        return new_sem_nodes

    # ------------------------------------------------------------------
    # Anchor-传播 语义分配(C/D/E/F 及其 plain/mutual + ce_fallback/re_llm 变体)
    # 移植自 jupyter_notebooks/hotpotqa_anchor_propagation_compare.ipynb 的
    # run_anchor_propagation:少量 anchor 经 LLM 候选合并标注,在 (mutual-)kNN 图上
    # 传播,CE 作"反对票"。anchor 标签复用现有 use_llm_candidate_filter 的合并路径;
    # 无 LLM 时退化为 cross-encoder top-1。
    # ------------------------------------------------------------------

    # 把一个 text-embedding 的向量转成 1-D float32 numpy。
    def _te_embed_to_numpy(self, text_embedding):
        embed = text_embedding.embed
        if isinstance(embed, torch.Tensor):
            return embed.detach().to(torch.float32).cpu().numpy()
        return np.asarray(embed, dtype=np.float32)

    # 采样 anchor 位置:fft_ratio 用 FFT 覆盖边界/稀有点,其余 random。
    def _sample_anchor_positions(self, embeddings):
        n_records = len(embeddings)
        if n_records == 0:
            return []
        fraction = float(self._anchor_param("anchor_fraction"))
        fft_ratio = float(self._anchor_param("anchor_fft_ratio"))
        min_count = int(self._anchor_param("anchor_min_count"))
        max_count = int(self._anchor_param("anchor_max_count"))
        random_state = int(self._anchor_param("anchor_random_state"))

        n_anchor = max(min_count, int(round(fraction * n_records)))
        # 封顶:高频词的 occurrence 很多,fraction 会让 anchor 线性膨胀,导致一次性塞给
        # LLM 判定的样本过多(准确率下降 + 成本上升)。用 anchor_max_count 截断。
        n_anchor = min(n_anchor, n_records, max_count)
        n_fft = min(n_anchor, int(round(fft_ratio * n_anchor)))
        n_rand = n_anchor - n_fft

        fft_positions = []
        if n_fft > 0:
            fft_positions = [
                int(p)
                for p in self._run_farthest_first_traversal(
                    embeddings, n_fft, start="random", random_state=random_state
                )
            ]
        fft_set = set(fft_positions)
        remaining = [p for p in range(n_records) if p not in fft_set]
        rand_positions = []
        if n_rand > 0 and remaining:
            rng = np.random.default_rng(random_state)
            take = min(n_rand, len(remaining))
            rand_positions = [
                int(p) for p in rng.choice(remaining, size=take, replace=False)
            ]
        return sorted(fft_set | set(rand_positions))

    # 为给定 occurrence 构造描述 prompt(主 prompt 失败时退化 fallback)。
    def _build_occurrence_prompt_info(self, token_text, span_occurrence):
        chunk_node = span_occurrence.chunk_node
        prompt_info = self._build_sem_node_description_prompt(
            chunk_node.chunk_text,
            token_text,
            match_span=span_occurrence.get_span_tuple(),
        )
        if prompt_info is None:
            prompt_info = self._build_fallback_sem_description_prompt(
                chunk_node.chunk_text,
                token_text,
            )
        return prompt_info

    # Return raw LLM sample judgments from the combined candidate-merge call.
    def _combined_sample_judgments_by_order(self, token_text):
        cached_result = getattr(self, "llm_candidate_filter_result_cache", {}).get(token_text) or {}
        metadata = cached_result.get("metadata") or {}
        raw_judgments = metadata.get("sample_judgments") or []
        judgments_by_order = {}
        for item in raw_judgments:
            if not isinstance(item, dict):
                continue
            try:
                sample_order = int(item.get("sample_order"))
            except (TypeError, ValueError):
                continue
            judgments_by_order[sample_order] = dict(item)
        return judgments_by_order

    # Find the candidate-bank row matching a description assigned by an LLM sample judgment.
    def _candidate_by_description(self, candidate_bank, description):
        if not description:
            return None
        for candidate in candidate_bank or []:
            if candidate.get("description") == description:
                return candidate
        return None

    # Persist one LLM-labeled Anchor sample under its token node.
    def _record_llm_anchor_sample_assignment(
        self,
        token_node,
        sem_node,
        *,
        source,
        sample_order,
        anchor_position,
        span_occurrence,
        prompt_info,
        description=None,
        top_candidate=None,
        raw_judgment=None,
        accepted_as_anchor_label=False,
    ):
        if not hasattr(token_node, "llm_anchor_sample_assignments"):
            token_node.llm_anchor_sample_assignments = []

        chunk_node = span_occurrence.chunk_node
        doc_node = getattr(chunk_node, "doc_node", None)
        if description is None and top_candidate is not None:
            description = top_candidate.get("description")
        record = {
            "record_index": len(token_node.llm_anchor_sample_assignments),
            "global_record_index": getattr(self, "llm_anchor_sample_assignment_count", 0),
            "token_text": token_node.token_text,
            "token_node_id": token_node.token_node_id,
            "source_sem_node_id": sem_node.sem_node_id if sem_node is not None else None,
            "sem_assignment_method": self.sem_assignment_method,
            "source": source,
            "sample_role": "anchor",
            "sample_order": sample_order,
            "anchor_position": anchor_position,
            "accepted_as_anchor_label": bool(accepted_as_anchor_label),
            "assigned_description": description,
            "chunk_node_id": chunk_node.chunk_node_id,
            "doc_name": getattr(doc_node, "doc_name", None),
            "span_start": span_occurrence.span_start,
            "span_end": span_occurrence.span_end,
            "span_text": span_occurrence.span_text,
            "matched_text": (prompt_info or {}).get("matched_text"),
            "context_text": (prompt_info or {}).get("context_text"),
        }

        if top_candidate is not None:
            record.update(
                {
                    "predicted_entity_id": top_candidate.get("entity_id"),
                    "predicted_label": top_candidate.get("label"),
                    "predicted_definition": top_candidate.get("definition"),
                    "prediction_score": top_candidate.get("score"),
                    "prediction_method": top_candidate.get("prediction_method", source),
                    "llm_reason": top_candidate.get("llm_reason"),
                    "llm_cache_hit": top_candidate.get("llm_cache_hit"),
                }
            )
        if raw_judgment is not None:
            record.update(
                {
                    "judgment": raw_judgment.get("judgment"),
                    "sense_id": raw_judgment.get("sense_id"),
                    "confidence": raw_judgment.get("confidence"),
                    "reason": raw_judgment.get("reason"),
                    "raw_sample_judgment": dict(raw_judgment),
                }
            )

        token_node.llm_anchor_sample_assignments.append(record)
        self.llm_anchor_sample_assignment_count = (
            getattr(self, "llm_anchor_sample_assignment_count", 0) + 1
        )
        return record

    # 标注 anchor 子集 + 取候选库。返回 (candidate_bank, {pos: description})。
    def _label_anchor_positions(self, sem_node, records, anchor_positions):
        token_text = sem_node.token_node.token_text
        prepared = []
        for sample_index, pos in enumerate(anchor_positions, start=1):
            span_occurrence = records[pos].to_span_occurrence()
            prompt_info = self._build_occurrence_prompt_info(token_text, span_occurrence)
            if prompt_info is None:
                continue
            prepared.append((sample_index, pos, span_occurrence, prompt_info))
        if not prepared:
            return [], {}

        anchor_labels = {}
        if getattr(self, "use_llm_candidate_filter", False):
            # anchor 即送入"候选合并 + 逐样本判定"的单次 LLM 调用(与
            # wikidata_llm_candidate_merge_experiment 对齐),返回每个 anchor 的描述。
            fft_samples = []
            for sample_index, pos, span_occurrence, prompt_info in prepared:
                chunk_node = span_occurrence.chunk_node
                doc_node = getattr(chunk_node, "doc_node", None)
                title = getattr(doc_node, "doc_name", None) if doc_node is not None else None
                fft_samples.append(
                    {
                        "sample_order": sample_index,
                        "title": title,
                        "matched_text": prompt_info.get("matched_text") or "",
                        "context_text": prompt_info.get("context_text") or "",
                    }
                )
            candidate_bank, sample_sense_descriptions = self._run_combined_merge_for_samples(
                sem_node, fft_samples
            )
            judgments_by_order = self._combined_sample_judgments_by_order(token_text)
            valid_descriptions = {candidate["description"] for candidate in candidate_bank}
            for sample_index, pos, span_occurrence, prompt_info in prepared:
                description = sample_sense_descriptions.get(sample_index)
                top_candidate = self._candidate_by_description(candidate_bank, description)
                accepted = description in valid_descriptions
                self._record_llm_anchor_sample_assignment(
                    sem_node.token_node,
                    sem_node,
                    source="llm_candidate_merge",
                    sample_order=sample_index,
                    anchor_position=pos,
                    span_occurrence=span_occurrence,
                    prompt_info=prompt_info,
                    description=description,
                    top_candidate=top_candidate,
                    raw_judgment=judgments_by_order.get(sample_index),
                    accepted_as_anchor_label=accepted,
                )
                if accepted:
                    anchor_labels[pos] = description
            if not candidate_bank:
                return [], {}
            return candidate_bank, anchor_labels

        candidate_bank = self._load_sem_description_candidate_bank(
            sem_node, self._sem_description_candidate_bank_cache
        )
        if not candidate_bank:
            return [], {}
        if getattr(self, "use_llm_semantic_labeler", False):
            top_candidates = self._predict_descriptions_from_prompt_infos_with_llm(
                token_text,
                [prompt_info for _idx, _pos, _occ, prompt_info in prepared],
                candidate_bank,
            )
            for (idx, pos, span_occurrence, prompt_info), top_candidate in zip(prepared, top_candidates):
                if top_candidate is None:
                    self._record_llm_anchor_sample_assignment(
                        sem_node.token_node,
                        sem_node,
                        source="llm_semantic_labeler",
                        sample_order=idx,
                        anchor_position=pos,
                        span_occurrence=span_occurrence,
                        prompt_info=prompt_info,
                        accepted_as_anchor_label=False,
                    )
                    continue
                anchor_labels[pos] = top_candidate["description"]
                self._record_llm_anchor_sample_assignment(
                    sem_node.token_node,
                    sem_node,
                    source="llm_semantic_labeler",
                    sample_order=idx,
                    anchor_position=pos,
                    span_occurrence=span_occurrence,
                    prompt_info=prompt_info,
                    top_candidate=top_candidate,
                    accepted_as_anchor_label=True,
                )
        else:
            for _idx, pos, span_occurrence, _prompt_info in prepared:
                top_candidate = self._predict_description_for_span_occurrence(
                    token_text, span_occurrence, candidate_bank, self.sem_description_model
                )
                if top_candidate is not None:
                    anchor_labels[pos] = top_candidate["description"]
        return candidate_bank, anchor_labels

    # 用 cross-encoder 对所有 occurrence 全量打分,返回按 pos 索引的
    # (ce_top1, ce_scores, ce_margin)。CE 模型缺失时返回空 dict。
    def _full_cross_encoder_over_records(self, token_text, records, candidate_bank):
        model = self.sem_description_model
        if model is None or not candidate_bank:
            return {}, {}, {}
        pairs = []
        slices = []
        for pos, text_embedding in enumerate(records):
            span_occurrence = text_embedding.to_span_occurrence()
            prompt_info = self._build_occurrence_prompt_info(token_text, span_occurrence)
            if prompt_info is None:
                continue
            for candidate in candidate_bank:
                pairs.append((prompt_info["prompt_text"], candidate["hypothesis"]))
            slices.append((pos, len(candidate_bank)))
        if not pairs:
            return {}, {}, {}
        raw_scores = model.predict(
            pairs,
            batch_size=min(self.sem_description_batch_size, len(pairs)),
            show_progress_bar=False,
        )
        scores = extract_cross_encoder_scores(raw_scores, model)
        ce_top1, ce_scores, ce_margin = {}, {}, {}
        cursor = 0
        for pos, count in slices:
            window = scores[cursor:cursor + count]
            cursor += count
            ranked = sorted(
                zip(candidate_bank, window),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            ce_scores[pos] = {c["description"]: float(s) for c, s in ranked}
            ce_top1[pos] = ranked[0][0]["description"]
            ce_margin[pos] = (
                float(ranked[0][1]) - float(ranked[1][1]) if len(ranked) >= 2 else float("inf")
            )
        return ce_top1, ce_scores, ce_margin

    # 构造 (mutual-)kNN 邻接表(list[set])。
    def _build_anchor_knn_adjacency(self, embeddings, knn_k, knn_type):
        n = len(embeddings)
        normalized = _l2_normalize_rows(embeddings.astype(np.float32))
        sims = normalized @ normalized.T
        np.fill_diagonal(sims, -np.inf)
        k = min(int(knn_k), max(0, n - 1))
        knn_sets = [set() for _ in range(n)]
        if k > 0:
            for i in range(n):
                nbrs = np.argpartition(-sims[i], k - 1)[:k]
                knn_sets[i] = {int(j) for j in nbrs}
        adjacency = [set() for _ in range(n)]
        for i in range(n):
            for j in knn_sets[i]:
                if knn_type == "mutual":
                    if i in knn_sets[j]:
                        adjacency[i].add(j)
                        adjacency[j].add(i)
                else:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        return adjacency

    # 在 (mutual-)kNN 图上从 anchor 逐圈外扩传播。返回 labels/provenance/uncertain。
    def _run_anchor_propagation_core(
        self, records, embeddings, anchor_labels, ce_top1, ce_scores, ce_margin, spec
    ):
        n = len(records)
        normalized = _l2_normalize_rows(embeddings.astype(np.float32))
        adjacency = self._build_anchor_knn_adjacency(
            embeddings, self._anchor_param("prop_knn_k"), spec["knn_type"]
        )

        use_margin = spec["use_margin"]
        use_center = spec["use_center"]
        rare_conservative = spec["rare_conservative"]
        vote_ratio = float(self._anchor_param("prop_vote_ratio"))
        weak_vote_ratio = float(self._anchor_param("prop_weak_vote_ratio"))
        rare_vote_ratio = float(self._anchor_param("prop_rare_vote_ratio"))
        high_margin = float(self._anchor_param("prop_high_margin"))
        ce_oppose_gap = float(self._anchor_param("prop_ce_oppose_gap"))
        rare_anchor_threshold = int(self._anchor_param("prop_rare_anchor_threshold"))
        max_rounds = int(self._anchor_param("prop_max_rounds"))

        labels = dict(anchor_labels)
        provenance = {pos: "llm_anchor" for pos in anchor_labels}
        anchor_set = set(anchor_labels)

        anchor_class_counts = Counter(labels[pos] for pos in anchor_set)
        rare_classes = {
            desc for desc, cnt in anchor_class_counts.items() if cnt <= rare_anchor_threshold
        }
        class_centers = {}
        for desc in anchor_class_counts:
            members = [pos for pos in anchor_set if labels[pos] == desc]
            center = normalized[members].mean(axis=0)
            center = center / max(np.linalg.norm(center), 1e-12)
            class_centers[desc] = center
        center_descs = list(class_centers.keys())
        center_matrix = (
            np.stack([class_centers[d] for d in center_descs]).astype(np.float32)
            if center_descs else None
        )

        def nearest_center_class(pos):
            if center_matrix is None:
                return None
            sims = center_matrix @ normalized[pos]
            return center_descs[int(np.argmax(sims))]

        def ce_opposes(pos, candidate_desc):
            if not use_margin:
                return False
            top1 = ce_top1.get(pos)
            if top1 is None or top1 == candidate_desc:
                return False
            if ce_margin.get(pos, 0.0) < high_margin:
                return False
            scores = ce_scores.get(pos, {})
            return scores.get(candidate_desc, -float("inf")) < scores.get(top1, float("inf")) - ce_oppose_gap

        for _ in range(max_rounds):
            snapshot = dict(labels)
            new_labels = {}
            for pos in range(n):
                if pos in labels:
                    continue
                labeled_neighbors = [j for j in adjacency[pos] if j in snapshot]
                if not labeled_neighbors:
                    continue
                votes = Counter(snapshot[j] for j in labeled_neighbors)
                maj, maj_count = votes.most_common(1)[0]
                ratio = maj_count / len(labeled_neighbors)
                is_rare = maj in rare_classes
                effective_vote_ratio = (
                    rare_vote_ratio if (rare_conservative and is_rare) else vote_ratio
                )
                opposes = ce_opposes(pos, maj)
                center_ok = use_center and (nearest_center_class(pos) == maj)
                ce_t1 = ce_top1.get(pos)

                chosen, prov = None, None
                if ratio >= effective_vote_ratio:
                    if opposes:
                        if center_ok:
                            chosen, prov = maj, "center_override"
                    elif rare_conservative and is_rare and use_center and not center_ok and maj != ce_t1:
                        pass
                    else:
                        prov = "neighbor_ce_agree" if maj == ce_t1 else "neighbor_override"
                        chosen = maj
                else:
                    if use_center and center_ok and ratio >= weak_vote_ratio and not opposes:
                        chosen, prov = maj, "center_support"

                if chosen is not None:
                    new_labels[pos] = (chosen, prov)

            if not new_labels:
                break
            for pos, (lab, prov) in new_labels.items():
                labels[pos] = lab
                provenance[pos] = prov

        uncertain_positions = set()
        for pos in range(n):
            if pos in labels:
                continue
            if use_margin and ce_margin.get(pos, -float("inf")) >= high_margin and ce_top1.get(pos):
                labels[pos] = ce_top1[pos]
                provenance[pos] = "ce_high_margin"
            else:
                uncertain_positions.add(pos)
                provenance[pos] = "uncertain"

        return {
            "labels": labels,
            "provenance": provenance,
            "uncertain_positions": uncertain_positions,
            "anchor_set": anchor_set,
            "rare_classes": rare_classes,
        }

    # re_llm 收尾:对仍 uncertain 的 occurrence 真实再调一次 LLM 逐条判定。
    def _relabel_uncertain_with_llm(self, token_text, pos_te_pairs, candidate_bank):
        prepared = []
        for pos, text_embedding in pos_te_pairs:
            span_occurrence = text_embedding.to_span_occurrence()
            prompt_info = self._build_occurrence_prompt_info(token_text, span_occurrence)
            if prompt_info is None:
                continue
            prepared.append((pos, prompt_info))
        if not prepared:
            return {}
        top_candidates = self._predict_descriptions_from_prompt_infos_with_llm(
            token_text,
            [prompt_info for _pos, prompt_info in prepared],
            candidate_bank,
        )
        relabeled = {}
        for (pos, _prompt_info), top_candidate in zip(prepared, top_candidates):
            if top_candidate is not None:
                relabeled[pos] = top_candidate["description"]
        return relabeled

    # Anchor 传播主入口:替换 _assign_sem_description_on_build 的 consensus/d1d2 路径。
    def _assign_sem_description_anchor_propagation(self, sem_node, cluster_text_embeddings):
        token_node = sem_node.token_node
        token_text = token_node.token_text
        spec = self._anchor_method_spec
        records = list(cluster_text_embeddings)
        n = len(records)

        # 没有任何描述模型/标注器:回退单义节点(与既有 no_model 路径一致)。
        if (
            self.sem_description_model is None
            and not getattr(self, "use_llm_semantic_labeler", False)
            and not getattr(self, "use_llm_candidate_filter", False)
        ):
            sem_node.chunk_node_embed.clear()
            return [sem_node]
        if n == 0:
            sem_node.chunk_node_embed.clear()
            return [sem_node]

        embeddings = np.stack([self._te_embed_to_numpy(te) for te in records]).astype(np.float32)

        anchor_positions = self._sample_anchor_positions(embeddings)
        candidate_bank, anchor_labels = self._label_anchor_positions(
            sem_node, records, anchor_positions
        )
        if not candidate_bank or not anchor_labels:
            # 无法消歧:保留为单义节点。
            sem_node.chunk_node_embed.clear()
            return [sem_node]

        need_ce = spec["use_margin"] or spec["uncertain_mode"] == "ce_fallback"
        if need_ce:
            ce_top1, ce_scores, ce_margin = self._full_cross_encoder_over_records(
                token_text, records, candidate_bank
            )
        else:
            ce_top1, ce_scores, ce_margin = {}, {}, {}

        prop = self._run_anchor_propagation_core(
            records, embeddings, anchor_labels, ce_top1, ce_scores, ce_margin, spec
        )
        labels = prop["labels"]
        provenance = prop["provenance"]
        uncertain_positions = prop["uncertain_positions"]

        # uncertain 收尾。
        if spec["uncertain_mode"] == "ce_fallback":
            for pos in uncertain_positions:
                fallback = ce_top1.get(pos)
                if fallback is not None:
                    labels[pos] = fallback
                    provenance[pos] = "uncertain_ce_fallback"
        elif spec["uncertain_mode"] == "re_llm":
            relabeled = self._relabel_uncertain_with_llm(
                token_text,
                [(pos, records[pos]) for pos in uncertain_positions],
                candidate_bank,
            )
            for pos, description in relabeled.items():
                labels[pos] = description
                provenance[pos] = "uncertain_re_llm"

        # 仍未标记(LLM/CE 兜底失败)的 occurrence -> 取已标记里的全局多数,确保可分组。
        if len(labels) < n:
            if not labels:
                sem_node.chunk_node_embed.clear()
                return [sem_node]
            majority_description = Counter(labels.values()).most_common(1)[0][0]
            for pos in range(n):
                if pos not in labels:
                    labels[pos] = majority_description
                    provenance[pos] = "fallback_majority"

        final_groups = defaultdict(list)
        for pos, text_embedding in enumerate(records):
            final_groups[labels[pos]].append(text_embedding)

        self._log_sem_description_operation(
            "anchor_propagation_assign",
            token_text=token_text,
            sem_node_id=sem_node.sem_node_id,
            method=self.sem_assignment_method,
            knn_type=spec["knn_type"],
            uncertain_mode=spec["uncertain_mode"],
            record_count=n,
            anchor_count=len(prop["anchor_set"]),
            uncertain_count=len(uncertain_positions),
            rare_classes=sorted(prop["rare_classes"]),
            description_counts={d: len(tes) for d, tes in final_groups.items()},
            provenance_counts=dict(Counter(provenance.values())),
        )

        # 单义:直接给原节点写描述。
        if len(final_groups) <= 1:
            sem_node.description = next(iter(final_groups))
            sem_node.chunk_node_embed.clear()
            return [sem_node]

        # 多义:用各 description 子集替换原节点(与 split 路径一致)。
        self.deleted_merged_sem_logs.append(
            {
                "deleted_sem_node_id": sem_node.sem_node_id,
                "kept_sem_node_id": None,
                "token_text": token_text,
                "description": "anchor_propagation_split",
                "deleted_chunk_count": len(sem_node.chunk_node_list),
            }
        )
        try:
            self.sem_nodes.remove(sem_node)
        except ValueError:
            pass
        sem_node.chunk_node_embed.clear()

        new_sem_nodes = []
        for description, tes in final_groups.items():
            if not tes:
                continue
            new_node = self._create_sem_node_from_cluster_subset(token_node, description, tes)
            new_sem_nodes.append(new_node)
            self._log_sem_description_operation(
                "create_anchor_propagation_sem_node",
                token_text=token_text,
                source_sem_node_id=sem_node.sem_node_id,
                sem_node_id=new_node.sem_node_id,
                description=description,
                sample_count=len(tes),
            )
        token_node.has_semantic = bool(new_sem_nodes)
        self._reset_sem_node_ids()
        return new_sem_nodes

    # Create a stable in-memory key for matching span occurrences.
    def _get_span_occurrence_key(self, span_occurrence):
        chunk_node = span_occurrence.chunk_node
        return (
            id(chunk_node),
            span_occurrence.span_start,
            span_occurrence.span_end,
            span_occurrence.span_text,
        )

    # Index retained text embeddings by their span occurrence keys.
    def _build_retained_text_embedding_lookup(self, sem_node):
        lookup = defaultdict(list)
        for text_embedding in getattr(sem_node, "retained_text_embeddings", []):
            lookup[self._get_span_occurrence_key(text_embedding.to_span_occurrence())].append(text_embedding)
        return lookup

    # Resolve usable character bounds for a stored span occurrence.
    def _resolve_span_for_occurrence(self, token_text, span_occurrence):
        match_span = span_occurrence.get_span_tuple()
        if match_span is not None:
            return match_span

        chunk_text = span_occurrence.chunk_node.chunk_text
        if span_occurrence.span_text:
            lowered_text = chunk_text.casefold()
            lowered_span_text = span_occurrence.span_text.casefold()
            match_start = lowered_text.find(lowered_span_text)
            if match_start >= 0:
                return match_start, match_start + len(span_occurrence.span_text)

        return _find_description_match_span(chunk_text, token_text)

    # Re-encode a token span to regenerate a missing retained embedding.
    def _reencode_text_embedding_for_occurrence(self, token_node, span_occurrence):
        match_span = self._resolve_span_for_occurrence(token_node.token_text, span_occurrence)
        if match_span is None:
            return None

        chunk_text = span_occurrence.chunk_node.chunk_text
        token_embeddings, offsets = encode_text(
            chunk_text,
            self.text_encoder,
            self.tokenizer,
            self.device,
        )
        span_text = chunk_text[match_span[0]:match_span[1]]
        phrase_embs, _ = get_token_embeds(
            token_embeddings,
            offsets,
            [(span_text, match_span[0], match_span[1])],
            [],
        )
        if not phrase_embs:
            return None

        _, embed, start_char, end_char = phrase_embs[0]
        return TextEmbedding(
            embed=embed.detach().cpu(),
            chunk_node=span_occurrence.chunk_node,
            span_start=start_char,
            span_end=end_char,
            span_text=span_text,
            surface_text=getattr(span_occurrence, "surface_text", None),
            surface_start=getattr(span_occurrence, "surface_start", None),
            surface_end=getattr(span_occurrence, "surface_end", None),
            phrase_type=getattr(span_occurrence, "phrase_type", None),
            modifier_text=getattr(span_occurrence, "modifier_text", None),
            modifier_texts=list(getattr(span_occurrence, "modifier_texts", []) or []),
            modifier_texts_norm=list(getattr(span_occurrence, "modifier_texts_norm", []) or []),
            modifier_spans=list(getattr(span_occurrence, "modifier_spans", []) or []),
            atomic_modifier_spans=list(getattr(span_occurrence, "atomic_modifier_spans", []) or []),
        )

    # Rebuild the embedding and threshold for a semantic node created by splitting.
    def _finalize_split_sem_node_embed(self, sem_node):
        available_embeds = [
            text_embedding.embed
            for text_embedding in getattr(sem_node, "retained_text_embeddings", [])
            if text_embedding.embed is not None
        ]
        if not available_embeds:
            available_embeds = list(sem_node.chunk_node_embed)
        if available_embeds:
            sem_node.embed = self._store_sem_embed(average_embeds(available_embeds))
        else:
            sample_occurrences = list(getattr(sem_node, "span_occurrences", []))
            if sample_occurrences:
                sample_size = min(len(sample_occurrences), self.sem_retained_embed_limit)
                regenerated_text_embeddings = []
                for span_occurrence in random.sample(sample_occurrences, sample_size):
                    text_embedding = self._reencode_text_embedding_for_occurrence(
                        sem_node.token_node,
                        span_occurrence,
                    )
                    if text_embedding is not None:
                        regenerated_text_embeddings.append(text_embedding)

                if regenerated_text_embeddings:
                    sem_node.embed = self._store_sem_embed(average_embeds(
                        [text_embedding.embed for text_embedding in regenerated_text_embeddings]
                    ))
                    self._initialize_sem_retained_text_embeddings(
                        sem_node,
                        regenerated_text_embeddings,
                    )

        if sem_node.embed is None:
            raise ValueError(
                f"Failed to rebuild embed for split sem node {sem_node.sem_node_id} "
                f"({sem_node.token_node.token_text!r}, description={sem_node.description!r})."
            )

        sem_node.chunk_node_embed.clear()
        sem_node.pending_embed_rebuild = False

    # Create an empty semantic node that will receive split sample assignments.
    def _create_split_sem_node(self, token_node, description):
        new_sem_node = self.create_sem_node(token_node)
        new_sem_node.description = description
        new_sem_node.chunk_node_list = []
        new_sem_node.span_occurrences = []
        new_sem_node.chunk_node_embed = []
        new_sem_node.retained_text_embeddings = []
        new_sem_node.retained_text_embedding_source_count = 0
        new_sem_node.pending_embed_rebuild = True
        new_sem_node.chunk_edge_weight = []
        new_sem_node.embed = None
        new_sem_node.tf_dict_by_chunk_id = None
        new_sem_node.chunk_len_dict_by_id = None
        new_sem_node.BM25 = None
        new_sem_node.df = 0
        new_sem_node.idf = 0
        return new_sem_node

    # Format semantic-description logs as plain text.
    # Format one operation event as a single line for the chronological timeline view.
    def _format_timeline_event(self, event):
        event_type = event.get("event_type", "unknown")
        sample_assignment_event_types = {
            "assign_sample_as_class_seed",
            "assign_sample_by_d1_d2",
            "assign_sample_by_single_safe_class",
            "assign_sample_by_model_judgment",
            "assign_sample_by_medoid_fallback",
        }
        if event_type in sample_assignment_event_types:
            return f"[{event_type}] " + self._format_sample_assignment(event)

        if event_type == "assign_description_by_consensus":
            return (
                f"[{event_type}] sem_node_id={event.get('sem_node_id')} | "
                f"token={event.get('token_text')!r} | description={event.get('description')!r} | "
                f"samples={event.get('sample_count')} | top_count={event.get('top_description_count')} | "
                f"top_ratio={self._safe_float(event.get('top_description_ratio')):.3f}"
            )
        if event_type == "assign_description_by_fallback_top_vote":
            return (
                f"[{event_type}] sem_node_id={event.get('sem_node_id')} | "
                f"token={event.get('token_text')!r} | description={event.get('description')!r} | "
                f"samples={event.get('sample_count')} | top_count={event.get('top_description_count')} | "
                f"top_ratio={self._safe_float(event.get('top_description_ratio')):.3f} | "
                f"valid_classes={event.get('valid_class_count')}"
            )
        if event_type == "split_sem_node_by_sample_labels":
            initial_sizes = event.get("initial_class_sizes") or {}
            final_sizes = event.get("description_counts") or {}
            initial_repr = (
                ", ".join(f"{desc!r}={count}" for desc, count in initial_sizes.items())
                if initial_sizes else "-"
            )
            final_repr = (
                ", ".join(f"{desc!r}={count}" for desc, count in final_sizes.items())
                if final_sizes else "-"
            )
            return (
                f"[{event_type}] source_sem_node_id={event.get('source_sem_node_id')} | "
                f"token={event.get('token_text')!r} | source_chunk_count={event.get('source_chunk_count')} | "
                f"cluster_samples={event.get('cluster_sample_count')} | "
                f"valid_classes={event.get('valid_class_count')} | "
                f"safe_classes={event.get('safe_class_count')} | "
                f"danger_zone={event.get('danger_zone_count')} | "
                f"singleton_classes={event.get('singleton_class_count')} | "
                f"initial_class_sizes={{{initial_repr}}} | final_description_counts={{{final_repr}}}"
            )
        if event_type == "create_split_sem_node":
            return (
                f"[{event_type}] sem_node_id={event.get('sem_node_id')} | "
                f"source_sem_node_id={event.get('source_sem_node_id')} | "
                f"description={event.get('description')!r} | sample_count={event.get('sample_count')}"
            )
        if event_type == "merge_sem_nodes_with_same_description":
            return (
                f"[{event_type}] token={event.get('token_text')!r} | "
                f"description={event.get('description')!r} | "
                f"kept_sem_node_id={event.get('kept_sem_node_id')} | "
                f"merged_sem_node_ids={event.get('merged_sem_node_ids')} | "
                f"merged_chunk_count={event.get('merged_chunk_count')} | "
                f"retained_embed_count={event.get('retained_embed_count')} | "
                f"retained_embed_source_count={event.get('retained_embed_source_count')}"
            )
        # Generic fallback: dump remaining payload fields.
        payload = " | ".join(
            f"{key}={value!r}"
            for key, value in event.items()
            if key != "event_type"
        )
        return f"[{event_type}] {payload}"

    def _format_sem_description_logs_text(self):
        operations = list(self.sem_description_operation_logs)
        event_counts = Counter(item.get("event_type", "unknown") for item in operations)

        # Chronological timeline: replay every operation in the order it was emitted so the
        # full processing flow (per-sem-node consensus, splits, per-cluster descriptions,
        # sample assignments, and merges) is visible step by step.
        timeline_lines = ["=== Processing timeline ===", ""]
        if not operations:
            timeline_lines.append("(no operations recorded)")
        else:
            timeline_lines.append(f"({len(operations)} events in chronological order)")
            timeline_lines.append("")
            for index, event in enumerate(operations, start=1):
                timeline_lines.append(f"#{index:04d} {self._format_timeline_event(event)}")
        timeline_text = "\n".join(timeline_lines)

        consensus_events = [e for e in operations if e.get("event_type") == "assign_description_by_consensus"]
        split_events = [e for e in operations if e.get("event_type") == "split_sem_node_by_sample_labels"]
        create_split_events = [e for e in operations if e.get("event_type") == "create_split_sem_node"]
        sample_assignment_event_types = {
            "assign_sample_as_class_seed",
            "assign_sample_by_d1_d2",
            "assign_sample_by_model_judgment",
            "assign_sample_by_medoid_fallback",
        }
        sample_events = [e for e in operations if e.get("event_type") in sample_assignment_event_types]
        merge_events = [e for e in operations if e.get("event_type") == "merge_sem_nodes_with_same_description"]

        create_split_by_source = defaultdict(list)
        for event in create_split_events:
            create_split_by_source[event.get("source_sem_node_id")].append(event)
        create_split_by_sem_id = {
            event.get("sem_node_id"): event
            for event in create_split_events
            if event.get("sem_node_id") is not None
        }
        sample_events_by_source = defaultdict(list)
        for event in sample_events:
            sample_events_by_source[event.get("source_sem_node_id")].append(event)
        sample_events_by_target = defaultdict(list)
        for event in sample_events:
            sample_events_by_target[event.get("target_sem_node_id")].append(event)
        consensus_by_sem_id = {
            event.get("sem_node_id"): event
            for event in consensus_events
            if event.get("sem_node_id") is not None
        }

        predicted_by_sem_id = {
            item["sem_node_id"]: item for item in self.predicted_sem_description_logs
        }

        lines = []
        timing_lines = self._format_index_session_timing()
        if timing_lines:
            lines.append("=== Index session running info ===")
            lines.append("")
            lines.extend(timing_lines)
            lines.append("")
        lines.append("=== Semantic description log ===")
        lines.append("")
        lines.append(timeline_text)
        lines.append("")
        lines.append("=== Summary ===")
        lines.append("")
        lines.append("Event summary:")
        if not operations:
            lines.append("  (no operations recorded)")
        else:
            for event_type, count in sorted(event_counts.items()):
                lines.append(f"  {event_type}: {count}")

        lines.append("")
        lines.append(f"[1] Consensus description assignments ({len(consensus_events)})")
        if not consensus_events:
            lines.append("  (none)")
        else:
            for event in consensus_events:
                sem_id = event.get("sem_node_id")
                lines.append(
                    "  sem_node_id={} | token={!r} | description={!r} | "
                    "samples={} | top_count={} | top_ratio={:.3f}".format(
                        sem_id,
                        event.get("token_text"),
                        event.get("description"),
                        event.get("sample_count"),
                        event.get("top_description_count"),
                        self._safe_float(event.get("top_description_ratio")),
                    )
                )
                predicted = predicted_by_sem_id.get(sem_id)
                if predicted is not None:
                    sample_predictions = predicted.get("sample_predictions", [])
                    for sample in sample_predictions:
                        lines.append(
                            "    " + self._format_consensus_sample_prediction(sample)
                        )

        lines.append("")
        lines.append(f"[2] Split sem nodes by sample label ({len(split_events)})")
        if not split_events:
            lines.append("  (none)")
        else:
            for event in split_events:
                source_id = event.get("source_sem_node_id")
                lines.append(
                    "  source_sem_node_id={} | token={!r} | source_chunk_count={} | "
                    "cluster_samples={} | initial_samples={}".format(
                        source_id,
                        event.get("token_text"),
                        event.get("source_chunk_count"),
                        event.get("cluster_sample_count"),
                        event.get("initial_prediction_sample_count"),
                    )
                )
                lines.append(
                    "    valid_classes={} | safe_classes={} | safe_seeds={} | "
                    "danger_seeds={} | danger_zone={} | singleton_classes={} | "
                    "class_seed={} | single_safe_class={} | d1_d2={} | "
                    "model_judgment={} | medoid_fallback={}".format(
                        event.get("valid_class_count"),
                        event.get("safe_class_count"),
                        event.get("safe_seed_count"),
                        event.get("danger_seed_count"),
                        event.get("danger_zone_count"),
                        event.get("singleton_class_count"),
                        event.get("class_seed_sample_count"),
                        event.get("single_safe_class_assigned_count"),
                        event.get("d1_d2_assigned_count"),
                        event.get("model_judgment_count"),
                        event.get("medoid_fallback_count"),
                    )
                )
                initial_sizes = event.get("initial_class_sizes") or {}
                if initial_sizes:
                    lines.append(
                        "    initial_class_sizes: "
                        + ", ".join(f"{desc!r}={count}" for desc, count in initial_sizes.items())
                    )
                final_sizes = event.get("description_counts") or {}
                if final_sizes:
                    lines.append(
                        "    final_description_counts: "
                        + ", ".join(f"{desc!r}={count}" for desc, count in final_sizes.items())
                    )
                safe_sizes = event.get("safe_class_sizes") or {}
                if safe_sizes:
                    lines.append(
                        "    safe_class_sizes: "
                        + ", ".join(f"{desc!r}={count}" for desc, count in safe_sizes.items())
                    )
                created_events = create_split_by_source.get(source_id, [])
                if created_events:
                    lines.append("    created sem nodes:")
                    for created in created_events:
                        lines.append(
                            "      sem_node_id={} | description={!r} | sample_count={}".format(
                                created.get("sem_node_id"),
                                created.get("description"),
                                created.get("sample_count"),
                            )
                        )
                assignment_events = sample_events_by_source.get(source_id, [])
                if assignment_events:
                    lines.append("    sample assignments:")
                    for entry in assignment_events:
                        lines.append("      " + self._format_sample_assignment(entry))

        orphan_create_events = [
            event for event in create_split_events
            if event.get("source_sem_node_id") not in {s.get("source_sem_node_id") for s in split_events}
        ]
        if orphan_create_events:
            lines.append("")
            lines.append("  (orphan create_split_sem_node events without a matching split record)")
            for event in orphan_create_events:
                lines.append(
                    "    sem_node_id={} | source_sem_node_id={} | description={!r} | sample_count={}".format(
                        event.get("sem_node_id"),
                        event.get("source_sem_node_id"),
                        event.get("description"),
                        event.get("sample_count"),
                    )
                )

        lines.append("")
        lines.append(f"[3] Same-description merges ({len(merge_events)})")
        if not merge_events:
            lines.append("  (none)")
        else:
            for event in merge_events:
                lines.append(
                    "  token={!r} | description={!r} | kept_sem_node_id={} | "
                    "merged_sem_node_ids={} | merged_chunk_count={} | "
                    "retained_embed_count={} | retained_embed_source_count={}".format(
                        event.get("token_text"),
                        event.get("description"),
                        event.get("kept_sem_node_id"),
                        event.get("merged_sem_node_ids"),
                        event.get("merged_chunk_count"),
                        event.get("retained_embed_count"),
                        event.get("retained_embed_source_count"),
                    )
                )
                merged_details = event.get("merged_sem_node_details") or []
                if not merged_details:
                    continue
                lines.append("    merged sem node details:")
                for detail in merged_details:
                    sem_id = detail.get("sem_node_id")
                    consensus_event = consensus_by_sem_id.get(sem_id)
                    create_split_event = create_split_by_sem_id.get(sem_id)
                    origin = self._describe_sem_node_origin(
                        sem_id,
                        consensus_event,
                        create_split_event,
                    )
                    lines.append(
                        "      sem_node_id={} | origin={} | chunk_count={} | "
                        "span_occurrence_count={} | retained_embed_count={} | "
                        "retained_embed_source_count={} | has_embed={} | pending_embed_rebuild={}".format(
                            sem_id,
                            origin,
                            detail.get("chunk_count"),
                            detail.get("span_occurrence_count"),
                            detail.get("retained_embed_count"),
                            detail.get("retained_embed_source_count"),
                            detail.get("has_embed"),
                            detail.get("pending_embed_rebuild"),
                        )
                    )
                    if consensus_event is not None:
                        lines.append(
                            "        consensus: samples={} | top_count={} | top_ratio={:.3f}".format(
                                consensus_event.get("sample_count"),
                                consensus_event.get("top_description_count"),
                                self._safe_float(consensus_event.get("top_description_ratio")),
                            )
                        )
                        predicted = predicted_by_sem_id.get(sem_id)
                        if predicted is not None:
                            sample_predictions = predicted.get("sample_predictions", [])
                            if sample_predictions:
                                lines.append("        consensus sample predictions:")
                                for sample in sample_predictions:
                                    lines.append(
                                        "          " + self._format_consensus_sample_prediction(sample)
                                    )
                    if create_split_event is not None:
                        lines.append(
                            "        created_by_split: source_sem_node_id={} | sample_count={}".format(
                                create_split_event.get("source_sem_node_id"),
                                create_split_event.get("sample_count"),
                            )
                        )
                    target_assignment_events = sample_events_by_target.get(sem_id, [])
                    if target_assignment_events:
                        assignment_counts = Counter(
                            entry.get("event_type", "unknown")
                            for entry in target_assignment_events
                        )
                        lines.append(
                            "        assignment_summary: class_seed={} | single_safe_class={} | "
                            "d1_d2={} | model_judgment={} | medoid_fallback={}".format(
                                assignment_counts.get("assign_sample_as_class_seed", 0),
                                assignment_counts.get("assign_sample_by_single_safe_class", 0),
                                assignment_counts.get("assign_sample_by_d1_d2", 0),
                                assignment_counts.get("assign_sample_by_model_judgment", 0),
                                assignment_counts.get("assign_sample_by_medoid_fallback", 0),
                            )
                        )
                        lines.append("        assignment_details:")
                        for assignment_event in target_assignment_events:
                            lines.append(
                                "          " + self._format_sample_assignment(assignment_event)
                            )

        lines.append("")
        lines.append(f"[4] Deleted sem nodes ({len(self.deleted_merged_sem_logs)})")
        if not self.deleted_merged_sem_logs:
            lines.append("  (none)")
        else:
            for item in self.deleted_merged_sem_logs:
                lines.append(
                    "  deleted_sem_node_id={} | kept_sem_node_id={} | token={!r} | "
                    "description={!r} | deleted_chunk_count={}".format(
                        item["deleted_sem_node_id"],
                        item["kept_sem_node_id"],
                        item["token_text"],
                        item["description"],
                        item["deleted_chunk_count"],
                    )
                )

        lines.append("")
        lines.append(f"[5] Wikidata lookups with no usable results ({len(self.wikidata_no_result_logs)})")
        if not self.wikidata_no_result_logs:
            lines.append("  (none)")
        else:
            for item in self.wikidata_no_result_logs:
                lines.append(
                    "  term={!r} | stage={} | reason={}".format(
                        item["term"],
                        item["stage"],
                        item["reason"],
                    )
                )

        return "\n".join(lines)

    # Format a per-sample assignment event into a single human-readable line.
    def _format_sample_assignment(self, event):
        event_type = event.get("event_type", "")
        span_repr = "chunk_node_id={} | span=({}, {}) | span_text={!r}".format(
            event.get("chunk_node_id"),
            event.get("span_start"),
            event.get("span_end"),
            event.get("span_text"),
        )
        description = event.get("description")
        target = event.get("target_sem_node_id")
        if event_type == "assign_sample_as_class_seed":
            return (
                f"[class_seed] target_sem_node_id={target} | description={description!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_d1_d2":
            return (
                f"[d1_d2]     target_sem_node_id={target} | description={description!r} | "
                f"d1={self._safe_float(event.get('d1')):.4f} | d2={self._safe_float(event.get('d2')):.4f} | "
                f"ratio={self._safe_float(event.get('ratio')):.4f} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_single_safe_class":
            return (
                f"[single]    target_sem_node_id={target} | description={description!r} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_model_judgment":
            return (
                f"[model]     target_sem_node_id={target} | description={description!r} | "
                f"predicted_label={event.get('predicted_label')!r} | "
                f"predicted_definition={event.get('predicted_definition')!r} | "
                f"score={self._safe_float(event.get('score')):.4f} | "
                f"d1={self._safe_float(event.get('d1')):.4f} | d2={self._safe_float(event.get('d2')):.4f} | "
                f"ratio={self._safe_float(event.get('ratio')):.4f} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        if event_type == "assign_sample_by_medoid_fallback":
            return (
                f"[fallback]  target_sem_node_id={target} | description={description!r} | "
                f"nearest_medoid_distance={self._safe_float(event.get('nearest_medoid_distance')):.4f} | "
                f"d1={self._safe_float(event.get('d1')):.4f} | d2={self._safe_float(event.get('d2')):.4f} | "
                f"ratio={self._safe_float(event.get('ratio')):.4f} | "
                f"singleton_class_description={event.get('singleton_class_description')!r} | "
                f"{span_repr}"
            )
        return f"[{event_type}] target_sem_node_id={target} | description={description!r} | {span_repr}"

    # Format a sampled prediction used in consensus description assignment.
    def _format_consensus_sample_prediction(self, sample):
        return (
            "sample #{} | chunk_node_id={} | predicted_description={!r} | "
            "predicted_label={!r} | score={:.4f} | matched_text={!r}".format(
                sample["sample_index"],
                sample["chunk_node_id"],
                sample["predicted_description"],
                sample["predicted_label"],
                self._safe_float(sample["prediction_score"]),
                sample["matched_text"],
            )
        )

    # Classify where a sem node came from so merge logs can show the full history.
    def _describe_sem_node_origin(self, sem_node_id, consensus_event, create_split_event):
        if consensus_event is not None:
            return "consensus_assignment"
        if create_split_event is not None:
            return "split_sem_node"
        return "preexisting_sem_node"

    # Coerce a possibly-None numeric field to float for safe formatting.
    @staticmethod
    def _safe_float(value):
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    # Format semantic-description logs as a small HTML details block.
    def _format_sem_description_logs_html(self, open_details=False):
        open_attr = " open" if open_details else ""
        return (
            f"<details{open_attr}>"
            "<summary>Semantic description logs</summary>"
            f"<pre>{escape(self._format_sem_description_logs_text())}</pre>"
            "</details>"
        )

    # Render the LiteSemRAG configuration (self.* attributes) as a JSON-formatted block.
    # Scalar attributes are dumped verbatim; small JSON-serializable containers (config
    # dicts/lists such as anchor_prop_params) are expanded so the dump shows their real
    # values instead of "<dict>". Runtime handles, large graph containers, per-token
    # caches, and log buffers are skipped.
    def _format_self_config_text(self):
        skip_field_names = set(RUNTIME_FIELD_NAMES) | {
            "doc_nodes", "chunk_nodes", "token_nodes", "phrase_token_nodes", "sem_nodes",
            "token_node_query", "phrase_index", "modifier_postings", "query_database",
            "predicted_sem_description_logs", "deleted_merged_sem_logs",
            "sem_description_operation_logs", "wikidata_no_result_logs",
            "_wikidata_no_result_keys", "_sem_description_candidate_bank_cache",
            "_combined_merge_sample_sense_descriptions",
            "llm_candidate_filter_result_cache",
        }
        scalar_types = (str, int, float, bool, type(None))
        # Cap on the serialized length of an expanded container so large runtime
        # caches that slipped past skip_field_names still render as "<dict>".
        max_container_chars = 4000
        config = {}
        for name, value in sorted(vars(self).items()):
            if name in skip_field_names:
                continue
            if isinstance(value, scalar_types):
                config[name] = value
                continue
            if isinstance(value, (dict, list, tuple)):
                try:
                    serialized = json.dumps(value, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    serialized = None
                if serialized is not None and len(serialized) <= max_container_chars:
                    config[name] = json.loads(serialized)
                    continue
            config[name] = f"<{type(value).__name__}>"
        return json.dumps(config, indent=2, ensure_ascii=False, default=str)

    # Save semantic-description logs and return the output path.
    def save_sem_description_logs(self, output_path=None, as_html=False, open_details=False):
        output_path = self.sem_description_log_path if output_path is None else output_path
        config_text = self._format_self_config_text()
        body_text = self._format_sem_description_logs_text()
        if as_html:
            open_attr = " open" if open_details else ""
            log_text = (
                f"<details{open_attr}>"
                "<summary>LiteSemRAG configuration</summary>"
                f"<pre>{escape(config_text)}</pre>"
                "</details>\n"
                f"<details{open_attr}>"
                "<summary>Semantic description logs</summary>"
                f"<pre>{escape(body_text)}</pre>"
                "</details>"
            )
        else:
            log_text = (
                "===== LiteSemRAG configuration =====\n"
                f"{config_text}\n"
                "===== Semantic description logs =====\n"
                f"{body_text}"
            )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(log_text)
        return output_path

    # Backward-compatible wrapper for callers that still use the old name.
    def print_sem_description_logs(self, output_path=None):
        return self.save_sem_description_logs(output_path=output_path)

    # Backward-compatible wrapper that now honors as_html and open_details.
    def show_sem_description_logs(self, as_html=True, open_details=False, output_path=None):
        return self.save_sem_description_logs(
            output_path=output_path,
            as_html=as_html,
            open_details=open_details,
        )

    # Print Wikidata lookup failures collected during description prediction.
    def print_wikidata_no_result_logs(self):
        print("Wikidata lookups with no usable results:")
        if not self.wikidata_no_result_logs:
            print("  (none)")
            return

        for item in self.wikidata_no_result_logs:
            print(
                "  "
                f"term={item['term']!r}, "
                f"stage={item['stage']}, "
                f"reason={item['reason']}"
            )

    # Compute BM25 chunk scores for every semantic node.
    def get_sem_BM25(self):
        for sem_node in self.sem_nodes:
            sem_node.get_BM25(self.chunk_avg_len)

    # Accumulate every extracted token/phrase embedding on its token node.
    # Semantic nodes are not built here: all embeddings buffer until finalize.
    def process_embeds(self, new_chunk_node, phrase_embs, token_embs):
        for embed_record in phrase_embs + token_embs:
            text, embed, span_start, span_end, *metadata = embed_record
            occurrence_metadata = metadata[0] if metadata else None
            text_embedding = self._make_text_embedding(
                embed,
                new_chunk_node,
                span_start=span_start,
                span_end=span_end,
                occurrence_metadata=occurrence_metadata,
            )
            token_node = self._get_or_create_token_node(text, is_placeholder=False)
            if token_node is None:
                continue
            if (
                occurrence_metadata
                and occurrence_metadata.get("phrase_type") == PHRASE_TYPE_ATOMIC
            ):
                token_node.force_single_semantic = True
            self._register_compositional_phrase_relation(
                token_node,
                occurrence_metadata,
                new_chunk_node,
            )
            token_node.embeds_buffer.append(text_embedding)
            self._append_token_occurrence(token_node, text_embedding)

    # Index one document through the shared staged indexing pipeline.
    def index_document(self, doc_name, multiprocessing=True):
        # multiprocessing is retained for API compatibility; all document
        # indexing now uses the shared staged pipeline.
        _ = multiprocessing
        return self.index_document_parallel(doc_name)

    # Backward-compatible wrapper for the shared document indexing pipeline.
    def index_document_single_processing(self, doc_name):
        return self.index_document_parallel(doc_name)

    # Backward-compatible wrapper for the shared document indexing pipeline.
    def index_document_threaded(self, doc_name):
        return self.index_document_parallel(doc_name)

    # Backward-compatible wrapper for the shared document indexing pipeline.
    def index_document_multi_processing(self, doc_name):
        return self.index_document_parallel(doc_name)

    # Plot a token node embedding distribution for clustered embeddings.
    def plot_embed_distribution(self, token_node, clusters):
        embeds = [k.embed.cpu() for k in token_node.embeds_buffer]
        plot_embeddings(embeds, token_node.token_text, clusters)

    # Print the approximate memory size of the graph object.
    def print_memory_size(self):
        print_size_mb(self)

    # Build an inverted index from words to phrase token nodes.
    def build_phrase_query(self):
        for i, token_node in enumerate(self.phrase_token_nodes):
            words = token_node.token_text.split()
            for w in words:
                self.phrase_index[w].add(i)

    # Find phrase token nodes that contain all words from the input text.
    def phrase_word_intersection_query(self, text: str):
        words = text.split()
        if not words:
            return []

        result = None
        for w in words:
            s = self.phrase_index.get(w)
            if not s:
                return []
            result = s if result is None else (result & s)
            if not result:
                return []
        return [
            idx for idx in result
            if self.phrase_token_nodes[idx].token_text != text
        ]

    # Backward-compatible name for word-intersection phrase lookup.
    def phrase_fuzzy_query(self, text: str):
        return self.phrase_word_intersection_query(text)

    # Return semantic nodes with the highest cosine similarity to a query embedding.
    def max_cosine_sem_nodes(
            self,
            query_tensor,
            token_node_list,
            k=1,
            index_as_input=True
    ):

        all_embeds = []
        sem_node_map = []

        for item in token_node_list:
            if index_as_input:
                token_node = self.phrase_token_nodes[item]
            else:
                token_node = item

            for sem_node in token_node.sem_node_list:
                all_embeds.append(sem_node.embed)
                sem_node_map.append(sem_node)

        if len(all_embeds) == 0:
            return [], torch.tensor([])

        matrix = torch.stack(all_embeds).to(self.device)

        query_norm = F.normalize(query_tensor.unsqueeze(0), dim=1)
        matrix_norm = F.normalize(matrix, dim=1)

        similarities = torch.mm(query_norm, matrix_norm.t()).squeeze(0)

        k = min(k, similarities.size(0))

        topk_similarities, topk_indices = torch.topk(similarities, k)

        topk_sem_nodes = [sem_node_map[i] for i in topk_indices.tolist()]

        return topk_sem_nodes

    # Rebuild reverse edges from chunks to connected semantic nodes.
    def build_chunk2sem_edge(self):
        seen_sem_ids = {}
        for chunk_node in self.chunk_nodes:
            chunk_node.sem_node_list = []
            seen_sem_ids[id(chunk_node)] = set()
        for sem_node in self.sem_nodes:
            for chunk_node in sem_node.chunk_node_list:
                chunk_seen = seen_sem_ids.setdefault(id(chunk_node), set())
                if id(sem_node) not in chunk_seen:
                    chunk_seen.add(id(sem_node))
                    chunk_node.sem_node_list.append(sem_node)

    # Rebuild graph metadata after document or chunk deletion.
    def rebuild_metadata_after_deletion(self):
        valid_chunk_ids = {id(chunk_node) for chunk_node in self.chunk_nodes}

        for doc_node in self.doc_nodes:
            doc_node.chunk_node_list = [
                chunk_node for chunk_node in doc_node.chunk_node_list
                if id(chunk_node) in valid_chunk_ids
            ]

        valid_sem_nodes = []
        for sem_node in self.sem_nodes:
            keep_indices = [
                i for i, chunk_node in enumerate(sem_node.chunk_node_list)
                if id(chunk_node) in valid_chunk_ids
            ]
            sem_node.keep_occurrence_indices(keep_indices)
            sem_node.retained_text_embeddings = [
                text_embedding for text_embedding in getattr(sem_node, "retained_text_embeddings", [])
                if id(text_embedding.chunk_node) in valid_chunk_ids
            ]
            sem_node.retained_text_embedding_source_count = max(
                sem_node.retained_text_embedding_source_count,
                len(sem_node.retained_text_embeddings),
            )
            if sem_node.chunk_node_list:
                valid_sem_nodes.append(sem_node)

        self.sem_nodes = valid_sem_nodes
        self._reset_sem_node_ids()

        valid_sem_ids = {id(sem_node) for sem_node in self.sem_nodes}
        valid_token_nodes = []
        phrase_token_nodes = []
        token_node_query = {}

        for token_node in self.token_nodes:
            token_node.sem_node_list = [
                sem_node for sem_node in token_node.sem_node_list
                if id(sem_node) in valid_sem_ids
            ]
            token_node.has_semantic = len(token_node.sem_node_list) > 0
            token_node.span_occurrences = [
                item for item in token_node.span_occurrences
                if id(item.chunk_node) in valid_chunk_ids
            ]
            for record_key, record in list(getattr(token_node, "headed_phrase_records", {}).items()):
                occurrences = [
                    occurrence
                    for occurrence in record.get("occurrences", [])
                    if id(occurrence.get("chunk_node")) in valid_chunk_ids
                ]
                if not occurrences:
                    del token_node.headed_phrase_records[record_key]
                    continue
                record["occurrences"] = occurrences
                record["count"] = len(occurrences)
            if token_node.has_semantic or token_node.embeds_buffer:
                valid_token_nodes.append(token_node)
                token_node_query[token_node.token_text] = token_node
                if token_node.node_type == "phrase":
                    phrase_token_nodes.append(token_node)

        self.token_nodes = valid_token_nodes
        for index, token_node in enumerate(self.token_nodes):
            token_node.token_node_id = index
        self.next_token_node_id = len(self.token_nodes)
        self.token_node_query = token_node_query
        self.phrase_token_nodes = phrase_token_nodes
        self._cleanup_empty_placeholder_token_nodes()

        self.reset_chunk_node_id()
        self.reset_doc_node_id()

        self._recompute_chunk_avg_len()

        for token_node in self.token_nodes:
            self.assign_idf(token_node)

        if self.chunk_avg_len is not None:
            self.get_sem_BM25()
        else:
            for sem_node in self.sem_nodes:
                sem_node.BM25 = {}

        self.build_chunk2sem_edge()
        self.phrase_index = defaultdict(set)
        self.build_phrase_query()
        self.build_modifier_postings()

        if self.sem_nodes:
            self.build_query_database()
        else:
            self.query_database = None

    # Return the index of the semantic node most similar to an embedding.
    def get_max_sim_sem(self, token_node, embeds):
        sem_embeds = []
        sem_indices = []
        for index, sem_node in enumerate(token_node.sem_node_list):
            if sem_node.embed is None:
                continue
            sem_embeds.append(sem_node.embed.to(self.device))
            sem_indices.append(index)
        if not sem_embeds:
            return None
        x = torch.stack(sem_embeds, 0).to(self.device)
        q = embeds.unsqueeze(0).expand_as(x)
        return sem_indices[int(F.cosine_similarity(x, q, dim=1).argmax().item())]

    # Index prepared chunk records with staged CPU preprocessing, GPU encoding, and CPU consumption.
    def _index_chunk_records_pipeline(
        self,
        chunk_records,
        batch_size=8,
        queue_size=4,
        *,
        completion_message=None,
        reset_sem_build_stats=True,
    ):
        """Index records shaped as {"doc_name": str, "text": str}."""
        if self.start_time is None:
            self.start_time = time.perf_counter()
        if reset_sem_build_stats:
            self._reset_sem_build_stats()
        self._start_phrase_audit_session()

        total_chunks = len(chunk_records)
        doc_node_map = {}

        preprocess_queue = queue.Queue(maxsize=queue_size)
        result_queue = queue.Queue(maxsize=queue_size)

        progress_lock = threading.Lock()
        preprocess_done = 0
        gpu_done = 0
        cpu_done = 0

        worker_errors = []
        stop_event = threading.Event()

        # Capture the first worker exception and its stage name.
        def record_error(stage_name, exc):
            if worker_errors:
                return
            worker_errors.append(
                (
                    stage_name,
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                )
            )
            stop_event.set()

        # Return a thread-safe snapshot of indexing progress counters.
        def get_progress_snapshot():
            with progress_lock:
                return preprocess_done, gpu_done, cpu_done

        # Increment the preprocessing progress counter.
        def inc_preprocess_done(n=1):
            nonlocal preprocess_done
            with progress_lock:
                preprocess_done += n

        # Increment the GPU encoding progress counter.
        def inc_gpu_done(n=1):
            nonlocal gpu_done
            with progress_lock:
                gpu_done += n

        # Increment the CPU consumption progress counter.
        def inc_cpu_done(n=1):
            nonlocal cpu_done
            with progress_lock:
                cpu_done += n

        last_progress_update_time = [0.0]

        # Print throttled progress for the staged indexing pipeline.
        def print_progress(force=False):
            p_done, g_done, c_done = get_progress_snapshot()
            now = time.perf_counter()
            if not force and c_done < total_chunks and (now - last_progress_update_time[0]) < 0.2:
                return
            last_progress_update_time[0] = now
            print(
                f"\rCPU preprocessed: {p_done}/{total_chunks} | "
                f"GPU encoded: {g_done}/{total_chunks} | "
                f"CPU processed: {c_done}/{total_chunks}",
                end="",
                flush=True
            )

        # Put an item into a queue while respecting a recorded worker error.
        def safe_queue_put(target_queue, item):
            while not stop_event.is_set():
                try:
                    target_queue.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        # Create graph nodes and extract spans before GPU encoding.
        def cpu_preprocess_worker():
            try:
                for idx, chunk in enumerate(chunk_records):
                    if stop_event.is_set():
                        break

                    doc_name = chunk["doc_name"]
                    chunk_text = chunk["text"]
                    new_doc_node = doc_node_map.get(doc_name)
                    if new_doc_node is None:
                        new_doc_node = self.create_doc_node(doc_name)
                        doc_node_map[doc_name] = new_doc_node
                    # Parse the chunk with spaCy once and reuse the doc across
                    # chunk-node sizing, span extraction, and phrase routing.
                    spacy_doc = self.nlp(chunk_text)
                    # schema 7: record spaCy sentence-end offsets so span→sentence
                    # mapping is precomputed at index time instead of per query.
                    sentence_boundaries = [sent.end_char for sent in spacy_doc.sents]
                    new_chunk_node = self.create_chunk_node(
                        chunk_text,
                        new_doc_node,
                        num_tokens=len(spacy_doc),
                        sentence_boundaries=sentence_boundaries,
                    )

                    if getattr(self, "phrase_audit_enabled", False):
                        phrases, tokens, phrase_audit_records = extract_important_spans(
                            chunk_text,
                            self.nlp,
                            min_tokens=2,
                            discard_no_word=self.discard_no_word,
                            return_phrase_audit=True,
                            doc=spacy_doc,
                        )
                        self._append_phrase_audit_records(
                            phrase_audit_records,
                            doc_name,
                            new_chunk_node,
                            idx,
                        )
                    else:
                        phrases, tokens = extract_important_spans(
                            chunk_text,
                            self.nlp,
                            min_tokens=2,
                            discard_no_word=self.discard_no_word,
                            doc=spacy_doc,
                        )
                    phrases = self._prepare_index_phrase_spans(
                        chunk_text, phrases, doc=spacy_doc
                    )

                    if not safe_queue_put(
                        preprocess_queue,
                        (
                            idx,
                            new_chunk_node,
                            phrases,
                            tokens,
                            chunk_text
                        )
                    ):
                        break

                    inc_preprocess_done(1)
                    print_progress()

            except Exception as exc:
                record_error("cpu_preprocess_worker", exc)

            finally:
                safe_queue_put(preprocess_queue, None)

        # Batch preprocessed chunks and encode them on the GPU.
        def gpu_worker():
            # Encode and emit one accumulated GPU batch.
            def flush_batch(batch_buffer):
                if not batch_buffer:
                    return True

                batch_texts = [item[4] for item in batch_buffer]
                batch_items = [
                    (item[0], item[1], item[2], item[3])
                    for item in batch_buffer
                ]

                token_embeddings_batch, offsets_batch = encode_chunk_batch(
                    batch_texts,
                    self.text_encoder,
                    self.tokenizer,
                    self.device
                )

                if not safe_queue_put(
                    result_queue,
                    (
                        batch_items,
                        token_embeddings_batch,
                        offsets_batch
                    )
                ):
                    return False

                inc_gpu_done(len(batch_buffer))
                print_progress()
                return True

            batch_buffer = []

            try:
                while True:
                    if stop_event.is_set() and preprocess_queue.empty():
                        break

                    try:
                        item = preprocess_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if item is None:
                        break

                    batch_buffer.append(item)

                    if len(batch_buffer) >= batch_size:
                        if not flush_batch(batch_buffer):
                            break
                        batch_buffer = []

                if batch_buffer:
                    flush_batch(batch_buffer)

            except Exception as exc:
                record_error("gpu_worker", exc)

            finally:
                safe_queue_put(result_queue, None)

        # Thread-safety invariant (graph mutation is intentionally lock-free):
        # this is a strict single-producer -> single-encoder -> single-consumer
        # pipeline, and mutable graph state is partitioned by exclusive owner:
        #   - cpu_preprocess_worker is the SOLE writer of doc_nodes / chunk_nodes
        #     / doc_node_map and the SOLE caller of self.nlp;
        #   - the main consumer loop below is the SOLE writer of token nodes
        #     (via process_embeds);
        #   - gpu_worker touches no graph state.
        # The only shared mutable counters (preprocess_done / gpu_done /
        # cpu_done) are guarded by progress_lock. Do NOT spawn a second worker
        # for any stage without first adding locks around node creation and id
        # allocation, or this partitioning (and its lock-free safety) breaks.
        preprocess_thread = threading.Thread(target=cpu_preprocess_worker)
        gpu_thread = threading.Thread(target=gpu_worker)

        print_progress(force=True)
        preprocess_thread.start()
        gpu_thread.start()

        try:
            while True:
                if stop_event.is_set() and result_queue.empty() and not gpu_thread.is_alive():
                    break

                try:
                    item = result_queue.get(timeout=0.2)
                except queue.Empty:
                    print_progress()
                    continue

                if item is None:
                    break

                batch_items, token_embeddings_batch, offsets_batch = item

                for i_in_batch, meta in enumerate(batch_items):
                    _, node, phrases, tokens = meta

                    phrase_embs, token_embs = get_token_embeds(
                        token_embeddings_batch[i_in_batch],
                        offsets_batch[i_in_batch],
                        phrases,
                        tokens
                    )

                    self.process_embeds(node, phrase_embs, token_embs)
                    inc_cpu_done(1)
                    print_progress()

        finally:
            stop_event.set()
            preprocess_thread.join(timeout=1.0)
            gpu_thread.join(timeout=1.0)

        print_progress(force=True)
        print()

        if worker_errors:
            stage_name, error_text = worker_errors[0]
            raise RuntimeError(
                f"Error in {stage_name}:\n{error_text}"
            )

        # Indexing only accumulates embeddings on token nodes now; semantic
        # nodes (and modifier postings, IDF, BM25, ...) are all built later in
        # finalize() from each token's fully accumulated embedding set.

        if completion_message is not None:
            self.log_time(completion_message())

    # Index one document with staged CPU preprocessing, GPU encoding, and CPU consumption.
    def index_document_parallel(self, doc_name, batch_size=8, queue_size=4):
        index_start_time = time.perf_counter()
        chunk_records = [
            {
                "doc_name": doc_name,
                "text": chunk_text,
            }
            for chunk_text in split_doc(doc_name, self.nlp, max_tokens=self.chunk_size)
        ]
        self._start_index_session_timer(
            indexed_document_count=1,
            start_time=index_start_time,
        )
        self._index_chunk_records_pipeline(
            chunk_records,
            batch_size=batch_size,
            queue_size=queue_size,
            completion_message=lambda: (
                f"File {doc_name} completed. "
                f"Index pipeline time: {time.perf_counter() - index_start_time:.4f}s"
            ),
        )

    # Map chunk IDs to their stored chunk texts.
    def chunk_id2text(self, ids):
        for chunk_id in ids:
            if chunk_id < 0 or chunk_id >= len(self.chunk_nodes):
                raise IndexError(f"Chunk id {chunk_id} is outside the chunk node list.")
            if self.chunk_nodes[chunk_id].chunk_node_id != chunk_id:
                raise ValueError(
                    "chunk_nodes index invariant is broken: "
                    f"position {chunk_id} contains chunk id {self.chunk_nodes[chunk_id].chunk_node_id}."
                )
        return [self.chunk_nodes[id].chunk_text for id in ids]

    # Extract a representative sentence containing a token from a chunk.
    def _extract_sentence_for_token(self, chunk_text, token_text):
        if not chunk_text:
            return ""

        token_text_lower = token_text.lower()
        doc = self.nlp(chunk_text)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            if token_text_lower in sent_text.lower():
                return sent_text

        if len(doc) > 0:
            first_sent = next(doc.sents, None)
            if first_sent is not None:
                return first_sent.text.strip()

        return chunk_text.strip()

    # Collect representative sentence examples for one semantic node.
    def _collect_sem_sentence_examples(self, sem_node, token_text, max_sentences_per_sem=3):
        examples = []
        seen_sentences = set()

        for chunk_node in sem_node.chunk_node_list:
            sentence_text = self._extract_sentence_for_token(chunk_node.chunk_text, token_text)
            if not sentence_text:
                continue
            sentence_key = sentence_text.lower()
            if sentence_key in seen_sentences:
                continue

            seen_sentences.add(sentence_key)
            examples.append({
                "doc_name": chunk_node.doc_node.doc_name,
                "chunk_node_id": chunk_node.chunk_node_id,
                "sentence_text": sentence_text,
            })

            if len(examples) >= max_sentences_per_sem:
                break

        return examples

    # Return token nodes that have at least a requested number of semantic nodes.
    def inspect_multi_sem_token_nodes(self, min_sem_count=2, max_sentences_per_sem=3):
        result = {}

        for token_node in self.token_nodes:
            sem_count = len(token_node.sem_node_list)
            if sem_count < min_sem_count:
                continue

            result[token_node.token_text] = {
                "token_node_id": token_node.token_node_id,
                "sem_count": sem_count,
                "s_mean": getattr(token_node, "s_mean", None),
                "sem_nodes": [],
            }

            for sem_node in token_node.sem_node_list:
                result[token_node.token_text]["sem_nodes"].append({
                    "sem_node_id": sem_node.sem_node_id,
                    "description": sem_node.description,
                    "chunk_count": len(sem_node.chunk_node_list),
                    "sentence_examples": self._collect_sem_sentence_examples(
                        sem_node,
                        token_node.token_text,
                        max_sentences_per_sem=max_sentences_per_sem,
                    ),
                })

        return result

    # Aggregate the persisted s_mean over all token nodes with >= min_sem_count sem nodes.
    def compute_multi_sem_smean_stats(self, min_sem_count=2):
        # 统计口径:全图中拥有 >= min_sem_count 个 sem 节点的 token,不受显示截断影响。
        # s_mean 在 finalize 的 semantic_type_cls 里持久化;走多义切分的 token 必然有值,
        # 但 entity/原子短语(force_single)若因后续合并恰好有多个 sem,则 s_mean 为 None,
        # 这类样本计入 token 总数但不计入平均,用 missing 计数单独反映。
        total = 0
        smean_values = []
        for token_node in self.token_nodes:
            if len(token_node.sem_node_list) < min_sem_count:
                continue
            total += 1
            s_mean = getattr(token_node, "s_mean", None)
            if s_mean is not None:
                smean_values.append(float(s_mean))
        mean_smean = (sum(smean_values) / len(smean_values)) if smean_values else None
        return {
            "min_sem_count": min_sem_count,
            "token_count": total,
            "smean_available_count": len(smean_values),
            "smean_missing_count": total - len(smean_values),
            "mean_smean": mean_smean,
        }

    # Return token nodes with semantic nodes that have descriptions.
    def inspect_described_sem_token_nodes(self, max_sentences_per_sem=3):
        result = {}

        for token_node in self.token_nodes:
            described_sem_nodes = [
                sem_node for sem_node in token_node.sem_node_list
                if getattr(sem_node, "description", None)
            ]
            if not described_sem_nodes:
                continue

            result[token_node.token_text] = {
                "token_node_id": token_node.token_node_id,
                "sem_count": len(described_sem_nodes),
                "s_mean": getattr(token_node, "s_mean", None),
                "sem_nodes": [],
            }

            for sem_node in described_sem_nodes:
                result[token_node.token_text]["sem_nodes"].append({
                    "sem_node_id": sem_node.sem_node_id,
                    "description": sem_node.description,
                    "chunk_count": len(sem_node.chunk_node_list),
                    "sentence_examples": self._collect_sem_sentence_examples(
                        sem_node,
                        token_node.token_text,
                        max_sentences_per_sem=max_sentences_per_sem,
                    ),
                })

        return result

    # Filter, sort, and truncate semantic-node inspection results.
    def _select_multi_sem_token_nodes(
            self,
            inspect_data,
            token_contains=None,
            sort_by="sem_count",
            max_token_nodes=None,
            max_sems_per_token=None,
    ):
        items = list(inspect_data.items())

        if token_contains:
            keyword = token_contains.lower()
            items = [
                (token_text, token_info)
                for token_text, token_info in items
                if keyword in token_text.lower()
            ]

        if sort_by == "token_text":
            items.sort(key=lambda x: x[0].lower())
        elif sort_by == "total_chunk_count":
            items.sort(
                key=lambda x: sum(sem_info["chunk_count"] for sem_info in x[1]["sem_nodes"]),
                reverse=True,
            )
        else:
            items.sort(key=lambda x: x[1]["sem_count"], reverse=True)

        if max_token_nodes is not None:
            items = items[:max_token_nodes]

        selected_data = {}
        for token_text, token_info in items:
            sem_nodes = token_info["sem_nodes"]
            if max_sems_per_token is not None:
                sem_nodes = sorted(
                    sem_nodes,
                    key=lambda x: x["chunk_count"],
                    reverse=True,
                )[:max_sems_per_token]

            selected_data[token_text] = {
                "token_node_id": token_info["token_node_id"],
                "sem_count": token_info["sem_count"],
                "s_mean": token_info.get("s_mean"),
                "displayed_sem_count": len(sem_nodes),
                "total_chunk_count": sum(sem_info["chunk_count"] for sem_info in token_info["sem_nodes"]),
                "sem_nodes": sem_nodes,
            }

        return selected_data

    # Limit displayed sentence examples across inspection results.
    def _limit_sem_sentence_examples(self, inspect_data, max_examples_per_token=None):
        if max_examples_per_token is None:
            return inspect_data

        limited_data = {}
        for token_text, token_info in inspect_data.items():
            remaining = max_examples_per_token
            limited_sem_nodes = []

            for sem_info in token_info["sem_nodes"]:
                if remaining <= 0:
                    limited_examples = []
                else:
                    limited_examples = sem_info["sentence_examples"][:remaining]
                remaining -= len(limited_examples)

                limited_sem = dict(sem_info)
                limited_sem["sentence_examples"] = limited_examples
                limited_sem_nodes.append(limited_sem)

            limited_token_info = dict(token_info)
            limited_token_info["sem_nodes"] = limited_sem_nodes
            limited_token_info["displayed_example_count"] = (
                max_examples_per_token - max(remaining, 0)
            )
            limited_data[token_text] = limited_token_info

        return limited_data

    # Return token-grouped LLM assignments made for Anchor samples.
    def inspect_llm_anchor_sample_assignments(self):
        result = {}
        for token_node in self.token_nodes:
            records = list(getattr(token_node, "llm_anchor_sample_assignments", []) or [])
            if not records:
                continue
            description_counts = Counter(
                record.get("assigned_description") or "(unassigned)"
                for record in records
            )
            source_counts = Counter(
                record.get("source") or "unknown"
                for record in records
            )
            result[token_node.token_text] = {
                "token_node_id": token_node.token_node_id,
                "sem_count": len(token_node.sem_node_list),
                "s_mean": getattr(token_node, "s_mean", None),
                "record_count": len(records),
                "accepted_count": sum(
                    1 for record in records if record.get("accepted_as_anchor_label")
                ),
                "description_counts": dict(description_counts),
                "source_counts": dict(source_counts),
                "records": records,
            }
        return result

    # Filter, sort, and truncate LLM Anchor assignment inspection data.
    def _select_llm_anchor_sample_assignments(
            self,
            inspect_data,
            token_contains=None,
            sort_by="record_count",
            max_token_nodes=None,
            max_records_per_token=None,
    ):
        items = list(inspect_data.items())
        if token_contains:
            keyword = token_contains.lower()
            items = [
                (token_text, token_info)
                for token_text, token_info in items
                if keyword in token_text.lower()
            ]

        if sort_by == "token_text":
            items.sort(key=lambda x: x[0].lower())
        elif sort_by == "accepted_count":
            items.sort(key=lambda x: x[1]["accepted_count"], reverse=True)
        elif sort_by == "sem_count":
            items.sort(key=lambda x: x[1]["sem_count"], reverse=True)
        else:
            items.sort(key=lambda x: x[1]["record_count"], reverse=True)

        if max_token_nodes is not None:
            items = items[:max_token_nodes]

        selected = {}
        for token_text, token_info in items:
            records = list(token_info["records"])
            records.sort(
                key=lambda record: (
                    record.get("source_sem_node_id") if record.get("source_sem_node_id") is not None else -1,
                    record.get("sample_order") if record.get("sample_order") is not None else -1,
                    record.get("anchor_position") if record.get("anchor_position") is not None else -1,
                )
            )
            if max_records_per_token is not None:
                records = records[:max_records_per_token]
            selected[token_text] = {
                **token_info,
                "displayed_record_count": len(records),
                "records": records,
            }
        return selected

    # Format LLM Anchor assignment inspection results as plain text.
    def _format_llm_anchor_sample_assignments_text(self, inspect_data):
        lines = []
        for token_text, token_info in inspect_data.items():
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            lines.append(
                f"[Token] {token_text} | token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']} | s_mean={s_mean_text} | "
                f"records={token_info['record_count']} | accepted={token_info['accepted_count']} | "
                f"displayed={token_info.get('displayed_record_count', token_info['record_count'])}"
            )
            lines.append(f"  descriptions={token_info.get('description_counts', {})}")
            lines.append(f"  sources={token_info.get('source_counts', {})}")
            for idx, record in enumerate(token_info["records"], start=1):
                lines.append(
                    "  ({}) source={} | sample_order={} | anchor_position={} | "
                    "accepted={} | description={!r} | doc={!r} | chunk_id={} | span=({}, {})".format(
                        idx,
                        record.get("source"),
                        record.get("sample_order"),
                        record.get("anchor_position"),
                        record.get("accepted_as_anchor_label"),
                        record.get("assigned_description"),
                        record.get("doc_name"),
                        record.get("chunk_node_id"),
                        record.get("span_start"),
                        record.get("span_end"),
                    )
                )
                if record.get("judgment"):
                    lines.append(
                        "      judgment={} | confidence={} | reason={!r}".format(
                            record.get("judgment"),
                            record.get("confidence"),
                            record.get("reason"),
                        )
                    )
                if record.get("matched_text"):
                    lines.append(f"      matched_text={record.get('matched_text')!r}")
                if record.get("context_text"):
                    lines.append(f"      context={record.get('context_text')}")
            lines.append("")
        return "\n".join(lines).rstrip()

    # Format LLM Anchor assignment inspection results as clickable HTML details.
    def _build_llm_anchor_sample_assignments_html(self, inspect_data, open_details=False):
        total_tokens = len(inspect_data)
        total_records = sum(info["record_count"] for info in inspect_data.values())
        total_displayed = sum(
            info.get("displayed_record_count", info["record_count"])
            for info in inspect_data.values()
        )
        html_parts = [
            "<div style='font-family:Arial, sans-serif; line-height:1.5;'>",
            "<div style='margin-bottom:12px; padding:10px 12px; background:#f3f6f9; "
            "border:1px solid #d8e0e8; border-radius:8px;'>"
            f"<strong>LLM Anchor sample assignment token nodes:</strong> {total_tokens} | "
            f"records: {total_records} | displayed records: {total_displayed}"
            "</div>",
        ]
        open_attr = " open" if open_details else ""

        for token_text, token_info in inspect_data.items():
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            desc_counts = ", ".join(
                f"{desc}: {count}"
                for desc, count in token_info.get("description_counts", {}).items()
            ) or "-"
            source_counts = ", ".join(
                f"{source}: {count}"
                for source, count in token_info.get("source_counts", {}).items()
            ) or "-"
            token_meta = (
                f"token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']}, "
                f"s_mean={s_mean_text}, "
                f"records={token_info['record_count']}, "
                f"accepted={token_info['accepted_count']}, "
                f"displayed={token_info.get('displayed_record_count', token_info['record_count'])}"
            )
            html_parts.append(
                "<details style='margin:10px 0; border:1px solid #ddd; "
                f"border-radius:8px; padding:8px 12px; background:#fafafa;'{open_attr}>"
                f"<summary style='cursor:pointer; font-weight:700;'>{escape(token_text)}</summary>"
                f"<div style='margin:6px 0 0 2px; font-size:12px; color:#666;'>{escape(token_meta)}</div>"
                f"<div style='margin:4px 0 0 2px; font-size:12px; color:#666;'>descriptions: {escape(desc_counts)}</div>"
                f"<div style='margin:4px 0 8px 2px; font-size:12px; color:#666;'>sources: {escape(source_counts)}</div>"
            )

            for record in token_info["records"]:
                description = record.get("assigned_description") or "(unassigned)"
                summary = (
                    f"sample_order={record.get('sample_order')} | "
                    f"anchor_position={record.get('anchor_position')} | "
                    f"description={description} | "
                    f"accepted={record.get('accepted_as_anchor_label')}"
                )
                meta = (
                    f"source={record.get('source')} | method={record.get('sem_assignment_method')} | "
                    f"doc={record.get('doc_name')} | chunk_id={record.get('chunk_node_id')} | "
                    f"span=({record.get('span_start')}, {record.get('span_end')})"
                )
                judgment = (
                    f"judgment={record.get('judgment')} | sense_id={record.get('sense_id')} | "
                    f"confidence={record.get('confidence')} | reason={record.get('reason')}"
                    if record.get("judgment") or record.get("reason")
                    else ""
                )
                candidate = (
                    f"entity_id={record.get('predicted_entity_id')} | "
                    f"label={record.get('predicted_label')} | "
                    f"definition={record.get('predicted_definition')}"
                    if record.get("predicted_entity_id") or record.get("predicted_label")
                    else ""
                )
                html_parts.append(
                    "<details style='margin:8px 0 8px 16px; padding:8px 10px; "
                    "border-left:4px solid #9467bd; background:#fff;'>"
                    f"<summary style='cursor:pointer; font-weight:600;'>{escape(summary)}</summary>"
                    f"<div style='font-size:12px; color:#666; margin-top:4px;'>{escape(meta)}</div>"
                )
                if judgment:
                    html_parts.append(
                        f"<div style='font-size:12px; color:#666; margin-top:4px;'>{escape(judgment)}</div>"
                    )
                if candidate:
                    html_parts.append(
                        f"<div style='font-size:12px; color:#666; margin-top:4px;'>{escape(candidate)}</div>"
                    )
                if record.get("matched_text"):
                    html_parts.append(
                        "<div style='margin-top:6px;'><strong>Matched text:</strong> "
                        f"{escape(str(record.get('matched_text')))}</div>"
                    )
                if record.get("context_text"):
                    html_parts.append(
                        "<div style='margin-top:6px; white-space:pre-wrap;'>"
                        f"{escape(str(record.get('context_text')))}</div>"
                    )
                html_parts.append("</details>")
            html_parts.append("</details>")

        html_parts.append("</div>")
        return "".join(html_parts)

    # Display LLM-labeled Anchor samples grouped by token node.
    def show_llm_anchor_sample_assignments(
            self,
            as_html=True,
            token_contains=None,
            sort_by="record_count",
            max_token_nodes=50,
            max_records_per_token=20,
            open_details=False,
    ):
        inspect_data = self.inspect_llm_anchor_sample_assignments()
        inspect_data = self._select_llm_anchor_sample_assignments(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_records_per_token=max_records_per_token,
        )

        if not inspect_data:
            empty_text = "No LLM Anchor sample assignment records were found."
            if as_html:
                try:
                    from IPython.display import HTML
                    return HTML(f"<div>{escape(empty_text)}</div>")
                except ImportError:
                    return empty_text
            return empty_text

        if as_html:
            try:
                from IPython.display import HTML
                return HTML(
                    self._build_llm_anchor_sample_assignments_html(
                        inspect_data,
                        open_details=open_details,
                    )
                )
            except ImportError:
                pass

        return self._format_llm_anchor_sample_assignments_text(inspect_data)

    # 列顺序:把 token 级元信息放前面,原文(matched_text/context_text/span_text)放后面,
    # 既能当表格扫一眼,又完整保留上下文文本,避免在 notebook 里渲染海量嵌套 <details>。
    _LLM_ANCHOR_CSV_COLUMNS = [
        "token_text", "token_node_id", "sem_count", "s_mean",
        "source_sem_node_id", "sem_assignment_method", "source",
        "sample_order", "anchor_position", "accepted_as_anchor_label",
        "assigned_description",
        "doc_name", "chunk_node_id", "span_start", "span_end",
        "predicted_entity_id", "predicted_label", "predicted_definition",
        "prediction_score", "prediction_method", "llm_cache_hit", "llm_reason",
        "judgment", "sense_id", "confidence", "reason",
        "matched_text", "context_text", "span_text",
    ]

    # 把所有 LLM Anchor 样本分配记录展平成一行一条、含原文的表格写入本地文件(默认 CSV)。
    # 返回写出的文件路径。token_contains/sort_by 仅用于可选过滤/排序,默认导出全部记录。
    def save_llm_anchor_sample_assignments_csv(
            self,
            output_path=None,
            token_contains=None,
            sort_by="record_count",
            max_token_nodes=None,
            max_records_per_token=None,
    ):
        import csv

        inspect_data = self.inspect_llm_anchor_sample_assignments()
        inspect_data = self._select_llm_anchor_sample_assignments(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_records_per_token=max_records_per_token,
        )

        if output_path is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(project_root, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(logs_dir, f"llm_anchor_assignments_{timestamp}.csv")
        else:
            parent = os.path.dirname(os.path.abspath(output_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        columns = self._LLM_ANCHOR_CSV_COLUMNS
        row_count = 0
        with open(output_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for token_info in inspect_data.values():
                token_meta = {
                    "sem_count": token_info.get("sem_count"),
                    "s_mean": token_info.get("s_mean"),
                }
                for record in token_info["records"]:
                    row = {column: record.get(column) for column in columns}
                    row.update(token_meta)
                    writer.writerow(row)
                    row_count += 1

        print(
            f"Saved {row_count} LLM Anchor sample assignment records "
            f"({len(inspect_data)} token nodes) to {output_path}"
        )
        return output_path

    # Format semantic-node inspection results as plain text.
    # Render the s_mean aggregate as a one-line plain-text summary.
    def _format_multi_sem_smean_summary_text(self, smean_stats):
        mean_smean = smean_stats.get("mean_smean")
        mean_text = f"{mean_smean:.4f}" if mean_smean is not None else "N/A"
        return (
            f"[s_mean summary] token nodes with >= {smean_stats['min_sem_count']} sem nodes: "
            f"{smean_stats['token_count']} | mean s_mean = {mean_text} "
            f"(over {smean_stats['smean_available_count']} with s_mean, "
            f"{smean_stats['smean_missing_count']} missing)"
        )

    # Render the s_mean aggregate as an HTML summary box.
    def _build_multi_sem_smean_summary_html(self, smean_stats):
        mean_smean = smean_stats.get("mean_smean")
        mean_text = f"{mean_smean:.4f}" if mean_smean is not None else "N/A"
        return (
            "<div style='font-family:Arial, sans-serif; margin-bottom:12px; padding:10px 12px; "
            "background:#eef7ee; border:1px solid #cfe3cf; border-radius:8px;'>"
            f"<strong>s_mean summary</strong> — token nodes with &ge; {smean_stats['min_sem_count']} sem nodes: "
            f"<strong>{smean_stats['token_count']}</strong> | "
            f"mean s_mean = <strong>{escape(mean_text)}</strong> "
            f"<span style='color:#666;'>(over {smean_stats['smean_available_count']} with s_mean, "
            f"{smean_stats['smean_missing_count']} missing)</span>"
            "</div>"
        )

    def _format_multi_sem_token_nodes_text(self, inspect_data):
        lines = []

        for token_text, token_info in inspect_data.items():
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            lines.append(
                f"[Token] {token_text} | token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']} | "
                f"s_mean={s_mean_text} | "
                f"displayed_sem_count={token_info['displayed_sem_count']} | "
                f"total_chunk_count={token_info['total_chunk_count']} | "
                f"displayed_example_count={token_info.get('displayed_example_count', 'all')}"
            )

            for sem_info in token_info["sem_nodes"]:
                description = sem_info.get("description") or "(none)"
                lines.append(
                    f"  [Sem] sem_node_id={sem_info['sem_node_id']} | "
                    f"description={description!r} | "
                    f"chunk_count={sem_info['chunk_count']}"
                )

                for idx, example in enumerate(sem_info["sentence_examples"], start=1):
                    lines.append(
                        f"    ({idx}) doc={example['doc_name']} | chunk_id={example['chunk_node_id']}"
                    )
                    lines.append(f"        {example['sentence_text']}")

            lines.append("")

        return "\n".join(lines).rstrip()

    # Format semantic-node inspection results as HTML.
    def _build_multi_sem_token_nodes_html(self, inspect_data, open_details=False):
        total_tokens = len(inspect_data)
        html_parts = [
            "<div style='font-family:Arial, sans-serif; line-height:1.5;'>"
        ]
        html_parts.append(
            "<div style='margin-bottom:12px; padding:10px 12px; background:#f3f6f9; "
            "border:1px solid #d8e0e8; border-radius:8px;'>"
            f"<strong>Matched token nodes:</strong> {total_tokens}"
            "</div>"
        )

        for token_text, token_info in inspect_data.items():
            token_header = (
                f"{escape(token_text)}"
            )
            s_mean = token_info.get("s_mean")
            s_mean_text = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            token_meta = (
                f"<span style='color:#666;'>"
                f"token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']}, "
                f"s_mean={s_mean_text}, "
                f"displayed_sem_count={token_info['displayed_sem_count']}, "
                f"total_chunk_count={token_info['total_chunk_count']}, "
                f"displayed_example_count={token_info.get('displayed_example_count', 'all')}"
                f"</span>"
            )
            html_parts.append(
                "<details style='margin:10px 0; border:1px solid #ddd; "
                f"border-radius:8px; padding:8px 12px; background:#fafafa;' {'open' if open_details else ''}>"
                f"<summary style='cursor:pointer; font-weight:700;'>{token_header}</summary>"
                f"<div style='margin:6px 0 0 2px; font-size:12px;'>{token_meta}</div>"
            )

            for sem_info in token_info["sem_nodes"]:
                description = sem_info.get("description") or "(none)"
                sem_header = (
                    f"sem_node_id={sem_info['sem_node_id']} | "
                    f"description={description} | "
                    f"chunk_count={sem_info['chunk_count']}"
                )
                html_parts.append(
                    "<div style='margin:10px 0 6px 16px; padding:8px 10px; "
                    "border-left:4px solid #4c78a8; background:#fff;'>"
                    f"<div style='font-weight:600; margin-bottom:6px;'>{escape(sem_header)}</div>"
                )

                for example in sem_info["sentence_examples"]:
                    meta = (
                        f"doc={example['doc_name']} | "
                        f"chunk_id={example['chunk_node_id']}"
                    )
                    html_parts.append(
                        "<div style='margin:8px 0 10px 8px;'>"
                        f"<div style='font-size:12px; color:#666; margin-bottom:2px;'>{escape(meta)}</div>"
                        f"<div style='white-space:pre-wrap;'>{escape(example['sentence_text'])}</div>"
                        "</div>"
                    )

                html_parts.append("</div>")

            html_parts.append("</details>")

        html_parts.append("</div>")
        return "".join(html_parts)

    # Display token nodes that have multiple semantic nodes.
    def show_multi_sem_token_nodes(
            self,
            min_sem_count=2,
            max_sentences_per_sem=3,
            as_html=True,
            token_contains=None,
            sort_by="sem_count",
            max_token_nodes=20,
            max_sems_per_token=5,
            max_examples_per_token=10,
            open_details=False,
    ):
        inspect_data = self.inspect_multi_sem_token_nodes(
            min_sem_count=min_sem_count,
            max_sentences_per_sem=max_sentences_per_sem,
        )
        inspect_data = self._select_multi_sem_token_nodes(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_sems_per_token=max_sems_per_token,
        )
        inspect_data = self._limit_sem_sentence_examples(
            inspect_data,
            max_examples_per_token=max_examples_per_token,
        )

        # 在全图(不受 token_contains / max_token_nodes 截断影响)上统计 >= min_sem_count
        # 的 token 的 s_mean 平均值,作为汇总信息展示。
        smean_stats = self.compute_multi_sem_smean_stats(min_sem_count=min_sem_count)

        if not inspect_data:
            empty_text = "No token nodes matched the multi-semantic condition."
            if as_html:
                try:
                    from IPython.display import HTML
                    return HTML(f"<div>{escape(empty_text)}</div>")
                except ImportError:
                    return empty_text
            return empty_text

        if as_html:
            try:
                from IPython.display import HTML
                return HTML(
                    self._build_multi_sem_smean_summary_html(smean_stats)
                    + self._build_multi_sem_token_nodes_html(inspect_data, open_details=open_details)
                )
            except ImportError:
                pass

        summary_text = self._format_multi_sem_smean_summary_text(smean_stats)
        body_text = self._format_multi_sem_token_nodes_text(inspect_data)
        return f"{summary_text}\n\n{body_text}" if summary_text else body_text

    # Display token nodes whose semantic nodes have descriptions.
    def show_described_sem_token_nodes(
            self,
            max_sentences_per_sem=3,
            as_html=True,
            token_contains=None,
            sort_by="sem_count",
            max_token_nodes=20,
            max_sems_per_token=10,
            max_examples_per_token=10,
            open_details=False,
    ):
        inspect_data = self.inspect_described_sem_token_nodes(
            max_sentences_per_sem=max_sentences_per_sem,
        )
        inspect_data = self._select_multi_sem_token_nodes(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_sems_per_token=max_sems_per_token,
        )
        inspect_data = self._limit_sem_sentence_examples(
            inspect_data,
            max_examples_per_token=max_examples_per_token,
        )

        if not inspect_data:
            empty_text = "No sem nodes with descriptions were found."
            if as_html:
                try:
                    from IPython.display import HTML
                    return HTML(f"<div>{escape(empty_text)}</div>")
                except ImportError:
                    return empty_text
            return empty_text

        if as_html:
            try:
                from IPython.display import HTML
                return HTML(self._build_multi_sem_token_nodes_html(inspect_data, open_details=open_details))
            except ImportError:
                pass

        return self._format_multi_sem_token_nodes_text(inspect_data)

    # Delete all chunks that belong to selected document names.
    def delete_by_document(self, doc_name_list):
        for doc_name in doc_name_list:
            doc_nodes_to_delete = [doc_node for doc_node in self.doc_nodes if doc_node.doc_name == doc_name]
            if doc_nodes_to_delete:
                for doc_node_to_delete in doc_nodes_to_delete:
                    for chunk_node in list(doc_node_to_delete.chunk_node_list):
                        self.delete_chunk_node(chunk_node)
                    self.doc_nodes.remove(doc_node_to_delete)
                print(f"Delete document '{doc_name}' complete. Removed {len(doc_nodes_to_delete)} document node(s).")
            else:
                print(f"Can not find document '{doc_name}' in database.")
        self.rebuild_metadata_after_deletion()
        self.save_doc_to_json()

    # Remove one chunk node from graph indexes and connected nodes.
    def delete_chunk_node(self, chunk_node_to_delete):
        for node in chunk_node_to_delete.sem_node_list:
            index_to_delete = []
            for index, chunk_node in enumerate(node.chunk_node_list):
                if chunk_node is chunk_node_to_delete:
                    index_to_delete.append(index)
            index_to_delete = set(index_to_delete)
            node.keep_occurrence_indices([
                i for i in range(len(node.chunk_node_list))
                if i not in index_to_delete
            ])
            node.retained_text_embeddings = [
                text_embedding
                for text_embedding in getattr(node, "retained_text_embeddings", [])
                if text_embedding.chunk_node is not chunk_node_to_delete
            ]
            node.retained_text_embedding_source_count = max(
                node.retained_text_embedding_source_count,
                len(node.retained_text_embeddings),
            )
        self.chunk_nodes.remove(chunk_node_to_delete)

    # Renumber chunk nodes and refresh chunk IDs.
    def reset_chunk_node_id(self):
        for index, chunk_node in enumerate(self.chunk_nodes):
            chunk_node.chunk_node_id = index
        self.next_chunk_node_id = len(self.chunk_nodes)

    # Renumber document nodes and refresh document IDs.
    def reset_doc_node_id(self):
        for index, doc_node in enumerate(self.doc_nodes):
            doc_node.doc_node_id = index
        self.next_doc_node_id = len(self.doc_nodes)

    # Load and customize the spaCy tokenizer used for span extraction.
    def _load_nlp(self):

        nlp = spacy.load("en_core_web_lg")

        infixes = nlp.Defaults.infixes

        infixes = [x for x in infixes if '-' not in x]

        infix_re = compile_infix_regex(infixes)

        nlp.tokenizer = Tokenizer(
            nlp.vocab,
            prefix_search=nlp.tokenizer.prefix_search,
            suffix_search=nlp.tokenizer.suffix_search,
            infix_finditer=infix_re.finditer,
            token_match=nlp.tokenizer.token_match,
        )
        self.nlp = nlp
        self.phrase_analyzer = PhraseAnalyzer(self.nlp)

    # Load the tokenizer and transformer text encoder used for embeddings.
    def _load_text_encoder(self):
        print("Loading text encoder models in device:", "GPU" if torch.cuda.is_available() else "CPU")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.text_encoder_addr,
            local_files_only=True,
            fix_mistral_regex=True,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "The text encoder tokenizer must be a fast tokenizer because "
                "offset_mapping is required during indexing."
            )
        self.text_encoder = AutoModel.from_pretrained(
            self.text_encoder_addr,
            local_files_only=True
        )
        self.text_encoder.to(self.device)

    # Capture runtime-only fields before temporarily clearing them.
    def _snapshot_runtime_fields(self):
        missing = object()
        return {
            name: getattr(self, name, missing)
            for name in RUNTIME_FIELD_NAMES
            if hasattr(self, name)
        }, missing

    # Clear runtime-only fields that are not pickle-safe or should not be persisted.
    def _clear_runtime_fields(self):
        for name in RUNTIME_FIELD_NAMES:
            if hasattr(self, name):
                setattr(self, name, None)

    # Restore runtime-only fields after a temporary serialization cleanup.
    def _restore_runtime_fields_from_snapshot(self, snapshot, missing):
        for name in RUNTIME_FIELD_NAMES:
            if name in snapshot:
                setattr(self, name, snapshot[name])
            elif hasattr(self, name):
                delattr(self, name)

    # Prepare the graph for pickling by removing runtime-only components.
    def __getstate__(self):
        state = self.__dict__.copy()
        for name in RUNTIME_FIELD_NAMES:
            state[name] = None
        return state

    # Restore a pickled graph state and reinitialize runtime placeholders.
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.text_encoder = None
        self.tokenizer = None
        self.executor = None
        self.nlp = None
        self.reranker = None
        self.sem_description_model = None
        self.llm_candidate_filter = None
        self.llm_semantic_label_client = None
        self.phrase_analyzer = None
        self.encoder_lock = threading.Lock()
        self._ensure_backward_compatible_attrs()

    # Reload runtime components needed after deserialization.
    def _restore_runtime_components(self, load_nlp=True, load_reranker=False):
        self._ensure_backward_compatible_attrs()
        if self.text_encoder is None or self.tokenizer is None:
            self._load_text_encoder()
        # Loading must not write to disk: index_documents.json is refreshed by
        # save_doc_to_json() at index/finalize time, not on deserialization.
        # Writing here would let loading one graph overwrite another graph's
        # doc-name list under the same working directory.
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=4)
        if getattr(self, "encoder_lock", None) is None:
            self.encoder_lock = threading.Lock()
        if load_nlp and self.nlp is None:
            self._load_nlp()
        if self.nlp is not None and self.phrase_analyzer is None:
            self.phrase_analyzer = PhraseAnalyzer(self.nlp)
        if load_reranker and self.reranker is None:
            self.load_reranker()
        if self.sem_description_model is None:
            self._load_sem_description_model()

    # Save the full graph object to a pickle file.
    def save_data(self, path):
        with open(path, "wb") as f:
            deep_pickle_dump(self, f)

    # Save indexed document names to the configured JSON file.
    def save_doc_to_json(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([x.doc_name for x in self.doc_nodes], f, ensure_ascii=False, indent=2)

    # Load a pickled graph and restore runtime components.
    @classmethod
    def load_data(cls, path):
        with open(path, "rb") as f:
            obj= pickle.load(f)
        obj._restore_runtime_components(load_nlp=True, load_reranker=False)
        obj.build_modifier_postings()
        return obj

    # Snapshot object references before temporary ID conversion for split serialization.
    def _snapshot_node_references(self):
        return {
            "chunks": [
                (chunk_node, chunk_node.doc_node, list(chunk_node.sem_node_list))
                for chunk_node in self.chunk_nodes
            ],
            "docs": [
                (doc_node, list(doc_node.chunk_node_list))
                for doc_node in self.doc_nodes
            ],
            "sems": [
                (
                    sem_node,
                    list(sem_node.chunk_node_list),
                    sem_node.token_node,
                    getattr(sem_node, "retained_text_embeddings", []),
                    [
                        (text_embedding, text_embedding.chunk_node)
                        for text_embedding in getattr(sem_node, "retained_text_embeddings", [])
                    ],
                )
                for sem_node in self.sem_nodes
            ],
            "tokens": [
                (
                    token_node,
                    list(token_node.sem_node_list),
                    getattr(token_node, "compositional_head", None),
                    list(getattr(token_node, "compositional_modifiers", []) or []),
                    {
                        key: (
                            record.get("phrase_token_node"),
                            [
                                (occurrence, occurrence.get("chunk_node"))
                                for occurrence in record.get("occurrences", [])
                            ],
                        )
                        for key, record in getattr(token_node, "headed_phrase_records", {}).items()
                    },
                )
                for token_node in self.token_nodes
            ],
        }

    # Restore references captured by _snapshot_node_references.
    def _restore_node_references_from_snapshot(self, snapshot):
        for chunk_node, doc_node, sem_node_list in snapshot["chunks"]:
            chunk_node.doc_node = doc_node
            chunk_node.sem_node_list = sem_node_list
        for doc_node, chunk_node_list in snapshot["docs"]:
            doc_node.chunk_node_list = chunk_node_list
        for sem_node, chunk_node_list, token_node, retained_list, retained_refs in snapshot["sems"]:
            sem_node.chunk_node_list = chunk_node_list
            sem_node.token_node = token_node
            sem_node.retained_text_embeddings = retained_list
            for text_embedding, chunk_node in retained_refs:
                text_embedding.chunk_node = chunk_node
        for token_node, sem_node_list, compositional_head, compositional_modifiers, headed_refs in snapshot["tokens"]:
            token_node.sem_node_list = sem_node_list
            token_node.compositional_head = compositional_head
            token_node.compositional_modifiers = compositional_modifiers
            for key, (phrase_token_node, occurrence_refs) in headed_refs.items():
                record = token_node.headed_phrase_records.get(key)
                if record is None:
                    continue
                record["phrase_token_node"] = phrase_token_node
                for occurrence, chunk_node in occurrence_refs:
                    occurrence["chunk_node"] = chunk_node

    # Save graph structure and tensors into separate files.
    def save_data_split(self, pkl_path: str):
        pt_path = pkl_path.replace(".pkl", "_tensors.pt")
        reference_snapshot = self._snapshot_node_references()
        qbak = None
        pbak = None
        retained_bak = None
        tensor_fields_cleared = False
        runtime_snapshot, missing_runtime = self._snapshot_runtime_fields()
        runtime_fields_cleared = False
        missing_tensors_path = object()
        tensors_path_bak = getattr(self, "tensors_path", missing_tensors_path)
        missing_manifest = object()
        manifest_bak = getattr(self, "tensor_split_manifest", missing_manifest)
        try:
            self.node_instance2id()

            tensor_pack = {
                "query_database": self.query_database.detach().cpu() if self.query_database is not None else None,

                "sem_embeds": [
                    (p.embed.detach().cpu() if p.embed is not None else None)
                    for p in self.sem_nodes
                ],
                "sem_retained_text_embeddings": [
                    [
                        {
                            "embed": (
                                text_embedding.embed.detach().cpu()
                                if text_embedding.embed is not None else None
                            ),
                            "chunk_node": text_embedding.chunk_node,
                            "span_start": text_embedding.span_start,
                            "span_end": text_embedding.span_end,
                            "span_text": text_embedding.span_text,
                            "surface_text": getattr(text_embedding, "surface_text", None),
                            "surface_start": getattr(text_embedding, "surface_start", None),
                            "surface_end": getattr(text_embedding, "surface_end", None),
                            "phrase_type": getattr(text_embedding, "phrase_type", None),
                            "modifier_text": getattr(text_embedding, "modifier_text", None),
                            "modifier_texts": list(getattr(text_embedding, "modifier_texts", []) or []),
                            "modifier_texts_norm": list(getattr(text_embedding, "modifier_texts_norm", []) or []),
                            "modifier_spans": list(getattr(text_embedding, "modifier_spans", []) or []),
                            "atomic_modifier_spans": list(getattr(text_embedding, "atomic_modifier_spans", []) or []),
                        }
                        for text_embedding in getattr(p, "retained_text_embeddings", [])
                    ]
                    for p in self.sem_nodes
                ],
            }
            torch.save(tensor_pack, pt_path)

            qbak = self.query_database
            pbak = [p.embed for p in self.sem_nodes]
            retained_bak = [p.retained_text_embeddings for p in self.sem_nodes]

            self.query_database = None
            for p in self.sem_nodes:
                p.embed = None
                p.retained_text_embeddings = [
                    TextEmbedding(
                        embed=None,
                        chunk_node=text_embedding.chunk_node,
                        span_start=text_embedding.span_start,
                        span_end=text_embedding.span_end,
                        span_text=text_embedding.span_text,
                        surface_text=getattr(text_embedding, "surface_text", None),
                        surface_start=getattr(text_embedding, "surface_start", None),
                        surface_end=getattr(text_embedding, "surface_end", None),
                        phrase_type=getattr(text_embedding, "phrase_type", None),
                        modifier_text=getattr(text_embedding, "modifier_text", None),
                        modifier_texts=list(getattr(text_embedding, "modifier_texts", []) or []),
                        modifier_texts_norm=list(getattr(text_embedding, "modifier_texts_norm", []) or []),
                        modifier_spans=list(getattr(text_embedding, "modifier_spans", []) or []),
                        atomic_modifier_spans=list(getattr(text_embedding, "atomic_modifier_spans", []) or []),
                    )
                    for text_embedding in getattr(p, "retained_text_embeddings", [])
                ]
            tensor_fields_cleared = True

            self._clear_runtime_fields()
            runtime_fields_cleared = True

            self.tensors_path = pt_path
            self.tensor_split_manifest = {
                "format_version": 1,
                "tensors_path": pt_path,
            }
            with open(pkl_path, "wb") as f:
                deep_pickle_dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            if tensor_fields_cleared:
                self.query_database = qbak
                for p, e, retained in zip(self.sem_nodes, pbak, retained_bak):
                    p.embed = e
                    p.retained_text_embeddings = retained

            if runtime_fields_cleared:
                self._restore_runtime_fields_from_snapshot(runtime_snapshot, missing_runtime)

            self._restore_node_references_from_snapshot(reference_snapshot)

            if tensors_path_bak is missing_tensors_path:
                if hasattr(self, "tensors_path"):
                    del self.tensors_path
            else:
                self.tensors_path = tensors_path_bak
            if manifest_bak is missing_manifest:
                if hasattr(self, "tensor_split_manifest"):
                    del self.tensor_split_manifest
            else:
                self.tensor_split_manifest = manifest_bak

    # Load split graph structure and tensor files, then restore runtime components.
    @classmethod
    def load_data_split(cls, pkl_path: str):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        split_manifest = getattr(obj, "tensor_split_manifest", None)
        if split_manifest:
            pt_path = split_manifest.get("tensors_path", pkl_path.replace(".pkl", "_tensors.pt"))
        else:
            pt_path = getattr(obj, "tensors_path", pkl_path.replace(".pkl", "_tensors.pt"))
        tensor_pack = torch.load(pt_path)

        query_database = tensor_pack.get("query_database", None)
        obj.query_database = query_database.to(obj.device) if query_database is not None else None

        sem_embeds = tensor_pack.get("sem_embeds", tensor_pack.get("proto_embeds", []))
        sem_retained_text_embeddings = tensor_pack.get(
            "sem_retained_text_embeddings",
            tensor_pack.get("proto_retained_text_embeddings", []),
        )

        for i, p in enumerate(obj.sem_nodes):
            embed = sem_embeds[i] if i < len(sem_embeds) else None
            p.embed = embed.to("cpu") if embed is not None else None
            retained_pack = sem_retained_text_embeddings[i] if i < len(sem_retained_text_embeddings) else []
            p.retained_text_embeddings = [
                TextEmbedding(
                    embed=(
                        item["embed"].to("cpu")
                        if item.get("embed") is not None else None
                    ),
                    chunk_node=item["chunk_node"],
                    span_start=item.get("span_start"),
                    span_end=item.get("span_end"),
                    span_text=item.get("span_text"),
                    surface_text=item.get("surface_text"),
                    surface_start=item.get("surface_start"),
                    surface_end=item.get("surface_end"),
                    phrase_type=item.get("phrase_type"),
                    modifier_text=item.get("modifier_text"),
                    modifier_texts=list(item.get("modifier_texts", []) or []),
                    modifier_texts_norm=list(item.get("modifier_texts_norm", []) or []),
                    modifier_spans=list(item.get("modifier_spans", []) or []),
                    atomic_modifier_spans=list(item.get("atomic_modifier_spans", []) or []),
                )
                for item in retained_pack
            ]
            p.retained_text_embedding_source_count = max(
                getattr(p, "retained_text_embedding_source_count", 0),
                len(p.retained_text_embeddings),
            )

        obj._restore_runtime_components(load_nlp=True, load_reranker=False)
        obj.node_id2instance()
        obj.build_modifier_postings()
        if obj.query_database is None and obj.sem_nodes and all(p.embed is not None for p in obj.sem_nodes):
            obj.build_query_database()
        obj.load_reranker()
        return obj

    # Convert object references into integer IDs before serialization.
    def node_instance2id(self):
        for chunk_node in self.chunk_nodes:
            chunk_node.doc_node = chunk_node.doc_node.doc_node_id
            chunk_node.sem_node_list = [node.sem_node_id for node in chunk_node.sem_node_list]
        for doc_node in self.doc_nodes:
            doc_node.chunk_node_list = [chunk_node.chunk_node_id for chunk_node in doc_node.chunk_node_list]
        for sem_node in self.sem_nodes:
            sem_node.chunk_node_list = [chunk_node.chunk_node_id for chunk_node in sem_node.chunk_node_list]
            for text_embedding in getattr(sem_node, "retained_text_embeddings", []):
                text_embedding.chunk_node = text_embedding.chunk_node.chunk_node_id
            sem_node.token_node = sem_node.token_node.token_node_id
        for token_node in self.token_nodes:
            token_node.sem_node_list = [sem_node.sem_node_id for sem_node in token_node.sem_node_list]
            if getattr(token_node, "compositional_head", None) is not None:
                token_node.compositional_head = token_node.compositional_head.token_node_id
            token_node.compositional_modifiers = [
                modifier_node.token_node_id
                for modifier_node in getattr(token_node, "compositional_modifiers", []) or []
                if modifier_node is not None
            ]
            for record in getattr(token_node, "headed_phrase_records", {}).values():
                phrase_token_node = record.get("phrase_token_node")
                if phrase_token_node is not None:
                    record["phrase_token_node"] = phrase_token_node.token_node_id
                for occurrence in record.get("occurrences", []):
                    chunk_node = occurrence.get("chunk_node")
                    if chunk_node is not None:
                        occurrence["chunk_node"] = chunk_node.chunk_node_id

    # Convert serialized integer IDs back into object references.
    def node_id2instance(self):
        doc_by_id = {doc_node.doc_node_id: doc_node for doc_node in self.doc_nodes}
        chunk_by_id = {chunk_node.chunk_node_id: chunk_node for chunk_node in self.chunk_nodes}
        sem_by_id = {sem_node.sem_node_id: sem_node for sem_node in self.sem_nodes}
        token_by_id = {token_node.token_node_id: token_node for token_node in self.token_nodes}

        for chunk_node in self.chunk_nodes:
            chunk_node.doc_node = doc_by_id[chunk_node.doc_node]
            chunk_node.sem_node_list = [sem_by_id[idx] for idx in chunk_node.sem_node_list]
        for doc_node in self.doc_nodes:
            doc_node.chunk_node_list = [chunk_by_id[idx] for idx in doc_node.chunk_node_list]
        for sem_node in self.sem_nodes:
            sem_node.chunk_node_list = [chunk_by_id[idx] for idx in sem_node.chunk_node_list]
            sem_node.token_node = token_by_id[sem_node.token_node]
            if not hasattr(sem_node, "description"):
                sem_node.description = self._get_initial_sem_description(sem_node.token_node)
            self._ensure_sem_node_backward_compatible_attrs(sem_node)
            if not sem_node.span_occurrences and sem_node.chunk_node_list:
                sem_node.span_occurrences = [
                    SpanOccurrence(chunk_node=chunk_node)
                    for chunk_node in sem_node.chunk_node_list
                ]
            for text_embedding in sem_node.retained_text_embeddings:
                text_embedding.chunk_node = chunk_by_id[text_embedding.chunk_node]
        for token_node in self.token_nodes:
            token_node.sem_node_list = [sem_by_id[idx] for idx in token_node.sem_node_list]
            self._ensure_token_node_backward_compatible_attrs(token_node)
            head_id = getattr(token_node, "compositional_head", None)
            if isinstance(head_id, int):
                token_node.compositional_head = token_by_id[head_id]
            token_node.compositional_modifiers = [
                token_by_id[idx] if isinstance(idx, int) else idx
                for idx in getattr(token_node, "compositional_modifiers", []) or []
            ]
            for record in getattr(token_node, "headed_phrase_records", {}).values():
                phrase_token_id = record.get("phrase_token_node")
                if isinstance(phrase_token_id, int):
                    record["phrase_token_node"] = token_by_id[phrase_token_id]
                for occurrence in record.get("occurrences", []):
                    chunk_id = occurrence.get("chunk_node")
                    if isinstance(chunk_id, int):
                        occurrence["chunk_node"] = chunk_by_id[chunk_id]
        self._reset_sem_node_ids()

    # Index a list of document records with a staged CPU/GPU pipeline.
    def index_json(self, chunk_list, batch_size=8, queue_size=4, sample_count=None):
        """Index JSON document records shaped as {"title": str, "text": str}.

        Timing summaries always report the average per indexed document.
        sample_count is accepted for backward compatibility and ignored.
        """
        index_start_time = time.perf_counter()
        chunk_records = []
        for idx, chunk in enumerate(chunk_list):
            missing_keys = {"title", "text"} - set(chunk)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(
                    f"JSON document record at index {idx} is missing required key(s): {missing}"
                )
            chunk_records.append(
                {
                    "doc_name": chunk["title"],
                    "text": chunk["text"],
                }
            )

        indexed_document_count = len({record["doc_name"] for record in chunk_records})
        self._start_index_session_timer(
            indexed_document_count=indexed_document_count,
            start_time=index_start_time,
        )
        self._index_chunk_records_pipeline(
            chunk_records,
            batch_size=batch_size,
            queue_size=queue_size,
            completion_message=lambda: (
                f"Index {indexed_document_count} documents. "
                f"Index pipeline time: {time.perf_counter() - index_start_time:.4f}s"
            ),
        )

class ListBatchExtractor:
    # Initialize batched extraction state for round-robin or sequential modes.
    def __init__(self, list_of_lists, k, mode="round", exclude_list=None):
        if mode not in ("round", "sequential"):
            raise ValueError("mode must be 'round' or 'sequential'")

        self.list_of_lists = list_of_lists
        self.k = k
        self.mode = mode
        self.exclude_set = set(exclude_list) if exclude_list else set()

        self.positions = [0] * len(list_of_lists)

        self.seq_outer_idx = 0
        self.seq_inner_idx = 0

        self.finished = False

    # Extract values until result reaches target_length, preserving previous state.
    def extract(self, target_length, result=None):
        if self.finished:
            return result if result else []

        if result is None:
            result = []

        if self.mode == "round":
            self._extract_round(target_length, result)
        else:
            self._extract_sequential(target_length, result)

        return result

    # Extract values in round-robin order across source lists.
    def _extract_round(self, target_length, result):
        while len(result) < target_length:
            added = 0

            for idx, sublist in enumerate(self.list_of_lists):
                count = 0
                while (
                    self.positions[idx] < len(sublist)
                    and count < self.k
                    and len(result) < target_length
                ):
                    value = sublist[self.positions[idx]]
                    self.positions[idx] += 1

                    if value in self.exclude_set:
                        continue

                    result.append(value)
                    count += 1
                    added += 1

            if added == 0:
                self.finished = True
                break

    # Extract values from source lists sequentially.
    def _extract_sequential(self, target_length, result):
        while len(result) < target_length and self.seq_outer_idx < len(self.list_of_lists):

            sublist = self.list_of_lists[self.seq_outer_idx]

            while (
                self.seq_inner_idx < len(sublist)
                and len(result) < target_length
            ):
                value = sublist[self.seq_inner_idx]
                self.seq_inner_idx += 1

                if value in self.exclude_set:
                    continue

                result.append(value)

            if self.seq_inner_idx >= len(sublist):
                self.seq_outer_idx += 1
                self.seq_inner_idx = 0

        if self.seq_outer_idx >= len(self.list_of_lists):
            self.finished = True

    # Return a snapshot of the extractor state.
    def get_state(self):
        return {
            "positions": self.positions.copy(),
            "seq_outer_idx": self.seq_outer_idx,
            "seq_inner_idx": self.seq_inner_idx,
            "finished": self.finished,
            "mode": self.mode
        }

    # Restore extractor state from a previous snapshot.
    def load_state(self, state):
        if state["mode"] != self.mode:
            raise ValueError("State mode does not match extractor mode")

        self.positions = state["positions"].copy()
        self.seq_outer_idx = state["seq_outer_idx"]
        self.seq_inner_idx = state["seq_inner_idx"]
        self.finished = state["finished"]

    # Reset extractor cursors and completion status.
    def reset(self):
        self.positions = [0] * len(self.list_of_lists)
        self.seq_outer_idx = 0
        self.seq_inner_idx = 0
        self.finished = False
