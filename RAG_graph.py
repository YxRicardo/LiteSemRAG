import json
import math
import pickle
import random
import re
import time
import traceback
from html import escape
from collections import Counter, defaultdict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import queue
import threading

import spacy
import torch
import torch.nn.functional as F
#from localizeJina import LocalJinaReranker
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from transformers import AutoModel, AutoTokenizer

from text_processing import (
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
    split_doc,
)
from utils import (
    average_embeds,
    build_wikidata_candidate_bank,
    extract_cross_encoder_scores,
    get_anomaly_threshold,
    get_s_mean,
    hdbscan_cluster,
    inspect_prototypes,
    load_wikidata_definition_candidates,
    plot_embeddings,
    print_size_mb,
    proto_embed_sim,
)

def get_COG_edge_weight(node_a, node_b):
    chunk_node_list_a = node_a.chunk_node_list
    chunk_node_list_b = node_b.chunk_node_list
    a_ids = {n.chunk_node_id for n in chunk_node_list_a}
    b_ids = {n.chunk_node_id for n in chunk_node_list_b}
    weight = len(a_ids & b_ids)/(math.sqrt(len(a_ids)*len(b_ids)))

    return weight

def unique_counts_by_id(a_list, b_list):
    a_ids = {n.chunk_node_id for n in a_list}
    b_ids = {n.chunk_node_id for n in b_list}
    return len(a_ids), len(b_ids), len(a_ids & b_ids)

class CoOccurrenceGraph:
    def __init__(self, proto_node_list):
        self.node_list = [CoOccurrenceNone(proto_node) for proto_node in proto_node_list]
        self.connected_node_list = []
        self.isolate_node_list = []
        self.weighted_chunk_node_list = []
        self.ranked_proto_list = []
        self.ranked_chunk_BM25 = []

    def build_edges(self):
        for node_a, node_b in combinations(self.node_list, 2):
            weight = get_COG_edge_weight(node_a.proto_node , node_b.proto_node)
            if weight > 0:
                node_a.neighbor_node_list.append((node_b, weight))
                node_b.neighbor_node_list.append((node_a, weight))

        for node in self.node_list:
            if len(node.neighbor_node_list) > 0:
                self.connected_node_list.append(node)
                for _, weight in node.neighbor_node_list:
                    node.node_weight += weight
                #node.node_weight = node.node_weight * node.node_level_weight * node.node_query_weight
                node.node_weight = node.node_weight * node.node_level_weight
            else:
                self.isolate_node_list.append(node)

    def assign_chunk_weight(self, avg_chunk_len, debug_mode=False):
        """
        对 self.connected_node_list 中每个 node:
          1) 取 node.proto_node.chunk_node (ChunkNode 列表)
          2) 按 chunk_node_id 去重
          3) 把权重累计到 self.weighted_chunk_node_list:
             - 若已存在该 chunk_node_id: weight += node.node_weight
             - 否则新增 (chunk_node_id, node.node_weight)
        注意：self.weighted_chunk_node_list 存的是 (chunk_node_id, weight) 的 tuple
        """
        # 先把现有的 weighted list 转成 dict 方便 O(1) 查找与累加
        if len(self.connected_node_list) > 0 :
            weight_map = {}
            token_record_map = defaultdict(list)
            token_weight_map = {}
            chunk_len_map = {}
            for node in self.connected_node_list:
                tf_chunk_dict = dict(Counter([chunk_node.chunk_node_id for chunk_node in node.proto_node.chunk_node_list]))
                for chunk_node in node.proto_node.chunk_node_list:
                    chunk_node_id = chunk_node.chunk_node_id
                    if node.proto_node.token_node.token_text not in token_record_map[chunk_node_id]:
                        #bm25_score = bm25_tf_saturation(tf_chunk_dict[chunk_node_id],chunk_node.num_tokens,avg_chunk_len)
                        bm25_score = node.node_weight * node.proto_node.BM25[chunk_node_id]
                        weight_map[chunk_node_id] = weight_map.get(chunk_node_id, 0) + bm25_score
                        token_weight_map.setdefault(chunk_node_id, []).append(f"Token:{node.proto_node.token_node.token_text},Score:{(bm25_score ):.4f}")
                        token_record_map[chunk_node_id].append(node.proto_node.token_node.token_text)
                        chunk_len_map[chunk_node_id] = chunk_node.num_tokens
            # for chunk_node_id, num_tokens in chunk_len_map.items():
            #     weight_map[chunk_node_id] = weight_map[chunk_node_id] / num_tokens
            self.weighted_chunk_node_list = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)
        if debug_mode:
            print(token_weight_map)
            print(weight_map)
            print(self.weighted_chunk_node_list)
    def rank_chunk_by_BM25(self):
        self.rank_proto_node_by_level()
        results = []

        for group in self.ranked_proto_list:
            chunk_scores = defaultdict(float)

            for inst in group:
                for chunk_id, score in inst.BM25.items():
                    chunk_scores[chunk_id] += score

            ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

            results.append([chunk_id for chunk_id, _ in ranked])
        self.ranked_chunk_BM25 = results

    def rank_proto_node_by_level(self):
        self.ranked_proto_list = [[] for i in range(4)]
        for node in self.node_list:
            self.ranked_proto_list[node.node_level].append(node.proto_node)

    def rank_proto_node(self):
        ranked_con_list = sorted(
            self.node_list,
            key=lambda node: (node.node_level, -node.proto_node.token_node.idf)
        )
        self.ranked_proto_list = [co_node.proto_node for co_node in ranked_con_list]
        # if len(self.isolate_node_list) > 0:
        #     self.isolate_node_list.sort(
        #         key=lambda x: (x.node_level, -x.proto_node.token_node.idf)
        #     )
        #     self.ranked_proto_list += [x.proto_node for x in self.isolate_node_list]
        # if len(self.connected_node_list) > 0:
        #     sort_list = sorted(
        #         self.connected_node_list,
        #         key=lambda x: (x.node_level, -x.proto_node.token_node.idf)
        #     )
        #     self.ranked_proto_list += [x.proto_node for x in sort_list]

    def print_node_weight(self):
        node_weights = []
        for node in self.node_list:
            node_weights.append(f"token:{node.proto_node.token_node.token_text}, weight: {node.node_weight:.4f}")
        print(node_weights)

CoOccurrenceNone_query_weight = [1.2,1,0.8,0.5]

@dataclass
class CoOccurrenceNone:
    proto_node: object
    node_level: int
    node_query_weight: float
    neighbor_node_list: list = field(default_factory=list)
    node_weight: float = 0
    node_level_weight: float = field(init=False)

    def __init__(self, proto_node_info):
        self.proto_node = proto_node_info[0]
        self.node_level = proto_node_info[1]
        self.node_query_weight = proto_node_info[2]
        self.neighbor_node_list = []
        self.node_weight = 0
        self.node_level_weight = CoOccurrenceNone_query_weight[self.node_level]


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
    proto_node_list: list = field(default_factory=list)
    length_norm: float = 0
    num_tokens: int | None = None


@dataclass
class TokenNode:
    token_text: str
    token_node_id: int
    node_type: str
    is_multi_semantic: bool = field(init=False)
    descriptions: list = field(init=False)
    wikidata_info_loaded: bool = field(init=False)
    has_prototype: bool = False
    proto_node_list: list = field(default_factory=list)
    embeds_buffer: list = field(default_factory=list)
    span_occurrences: list = field(default_factory=list)
    idf: float = 0
    df: int = 0
    anomaly_section: list = field(default_factory=list)

    def __post_init__(self):
        self.is_multi_semantic = None
        self.descriptions = None
        self.wikidata_info_loaded = None


@dataclass
class TextEmbedding:
    embed: object
    chunk_node: ChunkNode
    span_start: int | None = None
    span_end: int | None = None
    span_text: str | None = None

    def to_span_occurrence(self):
        span_text = self.span_text
        if (
            span_text is None
            and self.span_start is not None
            and self.span_end is not None
        ):
            span_text = self.chunk_node.chunk_text[self.span_start:self.span_end]
        return SpanOccurrence(
            chunk_node=self.chunk_node,
            span_start=self.span_start,
            span_end=self.span_end,
            span_text=span_text,
        )


@dataclass
class SpanOccurrence:
    chunk_node: ChunkNode
    span_start: int | None = None
    span_end: int | None = None
    span_text: str | None = None

    def get_span_tuple(self):
        if self.span_start is None or self.span_end is None:
            return None
        return self.span_start, self.span_end


@dataclass
class AnomalyTextEmbedding:
    text_embedding: TextEmbedding
    max_val: float
    max_idx: int


@dataclass
class Prototype:
    proto_node_id: int
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
    anomaly_threshold: float | None = None
    tf_dict_by_chunk_id: dict | None = None
    idf: float = 0
    df: int = 0
    chunk_len_dict_by_id: dict | None = None
    BM25: dict | None = None

    def get_tf(self):
        self.tf_dict_by_chunk_id = dict(Counter([chunk_node.chunk_node_id for chunk_node in self.chunk_node_list]))
        self.chunk_len_dict_by_id = {chunk_node.chunk_node_id: chunk_node.num_tokens for chunk_node in self.chunk_node_list}

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



class ProtoGraphRAG:
    def __init__(self, text_embed_dim, df_ratio, buffer_size=100, anomaly_threshold_percentile=0.9,
                 anomaly_section_size=50,query_token_percentile=0.8,
                 retrieve_top_k=5, chunk_size=300, remove_duplicate_token=True, device="cuda",
                 discard_no_word=False, plot_embeds=False,
                 proto_description_prompt_context_mode="sentence_neighbors"):
        self.text_embed_dim = text_embed_dim
        self.df_ratio = df_ratio
        self.doc_nodes = []
        self.chunk_nodes = []
        self.token_nodes = []
        self.phrase_token_nodes = []
        self.proto_nodes = []
        self.next_doc_node_id = 0
        self.next_chunk_node_id = 0
        self.next_token_node_id = 0
        self.next_proto_node_id = 0
        self.buffer_size = buffer_size
        self.token_node_query = {}
        self.tau_conc = 0.90
        self.tau_disp = 0.78
        self.build_proto_waitlist = []
        self.device = device
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.anomaly_section_size = anomaly_section_size
        self.anomaly_waitlist = []
        self.query_database = None
        self.retrieve_top_k = retrieve_top_k
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.start_time = None
        self.remove_duplicate_token = remove_duplicate_token
        self.phrase_index = defaultdict(set)
        self.query_token_percentile = query_token_percentile
        self.nlp = None
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_addr = "/home/xiaoyue/ProtoGraphRAG/deberta-v3-large"
        self._load_text_encoder()
        self._load_nlp()
        self.json_path = "index_documents.json"
        self.reranker = None
        self.load_reranker()
        self.proto_description_model_name = "cross-encoder/nli-deberta-v3-large"
        self.proto_description_model = None
        self._load_proto_description_model()
        self.proto_description_candidate_limit = 5
        self.proto_description_batch_size = 32
        self.proto_description_use_detailed_description = False
        self.proto_description_require_detailed_description = True
        self.proto_description_exact_match_text = True
        self.proto_description_exact_match_first = False
        self.proto_retained_embed_limit = 10
        self.proto_description_log_path = "proto_description_logs.txt"
        self.chunk_avg_len = None
        self.discard_no_word = discard_no_word
        self.plot_embeds = plot_embeds
        self.proto_description_prompt_context_mode = proto_description_prompt_context_mode
        self.predicted_proto_description_logs = []
        self.deleted_merged_proto_logs = []
        self.proto_description_operation_logs = []
        self.wikidata_no_result_logs = []
        self._wikidata_no_result_keys = set()
        self.hdbscan_attempt_count = 0
        self.hdbscan_success_count = 0

    def load_reranker(self):
        #self.reranker = LocalJinaReranker()
        self.reranker = None

    def _load_proto_description_model(self):
        from sentence_transformers import CrossEncoder

        self.proto_description_model = CrossEncoder(self.proto_description_model_name)

    def shutdown(self):
        self.executor.shutdown()

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
            pid = self.next_proto_node_id
            self.next_proto_node_id += 1
        return pid

    def _reset_proto_description_logs(self):
        self.predicted_proto_description_logs = []
        self.deleted_merged_proto_logs = []
        self.proto_description_operation_logs = []
        self.wikidata_no_result_logs = []
        self._wikidata_no_result_keys = set()

    def _log_proto_description_operation(self, event_type, **payload):
        self.proto_description_operation_logs.append(
            {
                "event_type": event_type,
                **payload,
            }
        )

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

    def _reset_hdbscan_stats(self):
        self.hdbscan_attempt_count = 0
        self.hdbscan_success_count = 0

    def _record_hdbscan_attempt(self, n_clusters):
        self.hdbscan_attempt_count += 1
        if n_clusters >= 1:
            self.hdbscan_success_count += 1

    def create_doc_node(self, doc_name):
        new_doc_node = DocumentNode(doc_name, self._new_node_id("doc"))
        self.doc_nodes.append(new_doc_node)
        return new_doc_node

    def create_chunk_node(self, chunk_text, doc_node):
        new_chunk_node = ChunkNode(chunk_text, self._new_node_id("chunk"), doc_node)
        new_chunk_node.num_tokens = get_num_tokens(chunk_text, self.nlp)
        self.chunk_nodes.append(new_chunk_node)
        doc_node.chunk_node_list.append(new_chunk_node)
        return new_chunk_node

    def create_token_node(self, token_text):
        node_type = 'phrase' if len(token_text.split()) >= 2 else 'token'
        new_token_node = TokenNode(token_text, self._new_node_id("token"), node_type)
        self.token_nodes.append(new_token_node)
        if new_token_node.node_type == "phrase":
            self.phrase_token_nodes.append(new_token_node)
        self.token_node_query[token_text] = new_token_node
        return new_token_node

    def create_proto_node(self, token_node):
        new_proto_node = Prototype(self._new_node_id("proto"), token_node)
        new_proto_node.description = self._get_initial_proto_description(token_node)
        self.proto_nodes.append(new_proto_node)
        token_node.has_prototype = True
        return new_proto_node

    def _make_text_embedding(self, embed, chunk_node, span_start=None, span_end=None):
        span_text = None
        if span_start is not None and span_end is not None:
            span_text = chunk_node.chunk_text[span_start:span_end]
        return TextEmbedding(
            embed=embed,
            chunk_node=chunk_node,
            span_start=span_start,
            span_end=span_end,
            span_text=span_text,
        )

    def _append_token_occurrence(self, token_node, text_embedding):
        token_node.span_occurrences.append(text_embedding.to_span_occurrence())

    def _clone_text_embedding(self, text_embedding):
        return TextEmbedding(
            embed=text_embedding.embed.clone() if torch.is_tensor(text_embedding.embed) else text_embedding.embed,
            chunk_node=text_embedding.chunk_node,
            span_start=text_embedding.span_start,
            span_end=text_embedding.span_end,
            span_text=text_embedding.span_text,
        )

    def _sample_text_embeddings(self, text_embeddings, max_samples=None):
        text_embeddings = list(text_embeddings)
        if max_samples is not None and max_samples > 0 and len(text_embeddings) > max_samples:
            text_embeddings = random.sample(text_embeddings, max_samples)
        return [self._clone_text_embedding(text_embedding) for text_embedding in text_embeddings]

    def _initialize_proto_retained_text_embeddings(self, proto_node, text_embeddings):
        text_embeddings = list(text_embeddings)
        proto_node.retained_text_embeddings = self._sample_text_embeddings(
            text_embeddings,
            max_samples=self.proto_retained_embed_limit,
        )
        proto_node.retained_text_embedding_source_count = len(text_embeddings)

    def _retain_text_embedding_for_proto(self, proto_node, text_embedding):
        if text_embedding is None or text_embedding.embed is None:
            return
        if not hasattr(proto_node, "retained_text_embeddings") or proto_node.retained_text_embeddings is None:
            proto_node.retained_text_embeddings = []
        if not hasattr(proto_node, "retained_text_embedding_source_count"):
            proto_node.retained_text_embedding_source_count = 0

        proto_node.retained_text_embedding_source_count += 1
        cloned_text_embedding = self._clone_text_embedding(text_embedding)
        if len(proto_node.retained_text_embeddings) < self.proto_retained_embed_limit:
            proto_node.retained_text_embeddings.append(cloned_text_embedding)
            return

        replace_index = random.randint(0, proto_node.retained_text_embedding_source_count - 1)
        if replace_index < self.proto_retained_embed_limit:
            proto_node.retained_text_embeddings[replace_index] = cloned_text_embedding

    def _append_proto_occurrence(self, proto_node, text_embedding, edge_weight=None):
        proto_node.chunk_node_list.append(text_embedding.chunk_node)
        proto_node.span_occurrences.append(text_embedding.to_span_occurrence())
        self._retain_text_embedding_for_proto(proto_node, text_embedding)
        if edge_weight is not None:
            proto_node.chunk_edge_weight.append(edge_weight)

    def _append_span_occurrence_to_proto(
        self,
        proto_node,
        span_occurrence,
        edge_weight=None,
        retained_text_embedding=None,
    ):
        proto_node.chunk_node_list.append(span_occurrence.chunk_node)
        proto_node.span_occurrences.append(
            SpanOccurrence(
                chunk_node=span_occurrence.chunk_node,
                span_start=span_occurrence.span_start,
                span_end=span_occurrence.span_end,
                span_text=span_occurrence.span_text,
            )
        )
        if retained_text_embedding is not None:
            self._retain_text_embedding_for_proto(proto_node, retained_text_embedding)
            if getattr(proto_node, "pending_embed_rebuild", False):
                proto_node.chunk_node_embed.append(retained_text_embedding.embed)
        if edge_weight is not None:
            proto_node.chunk_edge_weight.append(edge_weight)

    def _get_initial_proto_description(self, token_node):
        return None

    def create_basic_proto_node(self, token_node):
        new_proto_node = self.create_proto_node(token_node)
        new_proto_node.chunk_node_embed = ([i.embed for i in token_node.embeds_buffer])
        new_proto_node.embed = average_embeds(new_proto_node.chunk_node_embed)
        new_proto_node.chunk_node_list = [k.chunk_node for k in token_node.embeds_buffer]
        new_proto_node.span_occurrences = [k.to_span_occurrence() for k in token_node.embeds_buffer]
        self._initialize_proto_retained_text_embeddings(new_proto_node, token_node.embeds_buffer)
        new_proto_node.chunk_edge_weight = proto_embed_sim(new_proto_node).cpu().tolist()
        new_proto_node.anomaly_threshold = get_anomaly_threshold(new_proto_node.chunk_edge_weight,
                                                                 self.anomaly_threshold_percentile)
        new_proto_node.chunk_node_embed.clear()

        token_node.proto_node_list.append(new_proto_node)

    def build_proto(self, token_node):
        if token_node.node_type == "token" and not self.semantic_type_cls(token_node):
            self.create_basic_proto_node(token_node)
        else:
            n_clusters, clusters, cluster_centers = hdbscan_cluster([(k.embed.cpu(),k.chunk_node) for k in token_node.embeds_buffer],
                                                                    min_cluster_size=int(len(token_node.embeds_buffer)/20),
                                                                    percentile=self.anomaly_threshold_percentile, merge_chunks=False)
            self._record_hdbscan_attempt(n_clusters)
            if n_clusters >= 1:
                if self.plot_embeds:
                    self.plot_embed_distribution(token_node, clusters)
                for clusters_label in range(n_clusters):
                    new_proto_node = self.create_proto_node(token_node)
                    new_proto_node.embed = torch.from_numpy(cluster_centers[clusters_label])
                    cluster_text_embeddings = []
                    for idx in clusters[clusters_label]:
                        text_embedding = token_node.embeds_buffer[idx]
                        new_proto_node.chunk_node_list.append(text_embedding.chunk_node)
                        new_proto_node.span_occurrences.append(text_embedding.to_span_occurrence())
                        new_proto_node.chunk_node_embed.append(text_embedding.embed)
                        cluster_text_embeddings.append(text_embedding)
                    self._initialize_proto_retained_text_embeddings(new_proto_node, cluster_text_embeddings)
                    new_proto_node.chunk_edge_weight = proto_embed_sim(new_proto_node).cpu().tolist()
                    new_proto_node.anomaly_threshold = get_anomaly_threshold(new_proto_node.chunk_edge_weight,
                                                                             self.anomaly_threshold_percentile)
                    new_proto_node.chunk_node_embed.clear()
                    token_node.proto_node_list.append(new_proto_node)
                anomaly_idx = clusters.get(-1)
                if anomaly_idx is not None:
                    for idx in anomaly_idx:
                        text_embedding = token_node.embeds_buffer[idx]
                        max_val, max_idx = inspect_prototypes(text_embedding.embed, token_node.proto_node_list)
                        token_node.anomaly_section.append(
                            AnomalyTextEmbedding(
                                text_embedding=text_embedding,
                                max_val=max_val,
                                max_idx=max_idx,
                            )
                        )
            else:
                self.create_basic_proto_node(token_node)
        token_node.embeds_buffer.clear()

    def solve_proto(self):
        for token_node in self.build_proto_waitlist:
            self.build_proto(token_node)
        self.build_proto_waitlist = []

    def solve_anomaly(self):
        for token_node in self.anomaly_waitlist:
            n_clusters, clusters, cluster_centers = hdbscan_cluster(
                [
                    (
                        item.text_embedding.embed.cpu(),
                        item.text_embedding.chunk_node,
                    )
                    for item in token_node.anomaly_section
                ],
                min_cluster_size=10, percentile=self.anomaly_threshold_percentile)
            self._record_hdbscan_attempt(n_clusters)
            if n_clusters >= 1:
                for clusters_label in range(n_clusters):
                    new_proto_node = self.create_proto_node(token_node)
                    new_proto_node.embed = torch.from_numpy(cluster_centers[clusters_label]).to(self.device)
                    cluster_text_embeddings = []
                    for idx in clusters[clusters_label]:
                        text_embedding = token_node.anomaly_section[idx].text_embedding
                        new_proto_node.chunk_node_list.append(text_embedding.chunk_node)
                        new_proto_node.span_occurrences.append(text_embedding.to_span_occurrence())
                        new_proto_node.chunk_node_embed.append(text_embedding.embed)
                        cluster_text_embeddings.append(text_embedding)
                    self._initialize_proto_retained_text_embeddings(new_proto_node, cluster_text_embeddings)
                    new_proto_node.chunk_edge_weight = proto_embed_sim(new_proto_node).cpu().tolist()
                    new_proto_node.anomaly_threshold = get_anomaly_threshold(new_proto_node.chunk_edge_weight,
                                                                             self.anomaly_threshold_percentile)
                    new_proto_node.chunk_node_embed.clear()
                anomaly_idx = clusters.get(-1)
                if anomaly_idx is None:
                    token_node.anomaly_section = []
                else:
                    token_node.anomaly_section = [token_node.anomaly_section[i] for i in anomaly_idx]
            else:
                for item in token_node.anomaly_section:
                    self._append_proto_occurrence(
                        token_node.proto_node_list[item.max_idx],
                        item.text_embedding,
                        edge_weight=item.max_val,
                    )
        self.anomaly_waitlist = []

    # def assign_edge_weight(self, proto_node):
    #     sim = proto_embed_sim(proto_node)
    #     proto_node.chunk_node_embed.clear()
    #     proto_node.chunk_edge_weight = proto_node_combine_sim(sim, proto_node)

    def log_time(self, msg):
        print(f"[{time.perf_counter() - self.start_time:.4f}s] {msg}")

    def _make_progress_updater(self, label, total, min_interval=0.2):
        last_update_time = [0.0]

        def update(processed, force=False):
            now = time.perf_counter()
            if not force and processed < total and (now - last_update_time[0]) < min_interval:
                return
            last_update_time[0] = now
            print(
                f"\r{label}: {processed}/{total}",
                end="",
                flush=True,
            )

        return update

    def assign_idf(self, token_node):
        N = len(self.chunk_nodes)
        token_chunk_ids = set()
        for proto_node in token_node.proto_node_list:
            proto_chunk_ids = {chunk_node.chunk_node_id for chunk_node in proto_node.chunk_node_list}
            proto_node.df = len(proto_chunk_ids)
            proto_node.idf = math.log((N + 1) / (proto_node.df + 1)) + 1.0
            token_chunk_ids.update(proto_chunk_ids)
        token_node.df = len(token_chunk_ids)
        token_node.idf = math.log((N + 1) / (token_node.df + 1)) + 1.0

    def semantic_type_cls(self, token_node):
        if token_node.df > len(self.chunk_nodes) * self.df_ratio:
            return False
        s_mean = get_s_mean([i.embed for i in token_node.embeds_buffer])
        if s_mean > self.tau_conc:
            return False
        else:
            return True

    def query_token_node(self, text):
        return self.token_node_query.get(text, None)

    def _ensure_backward_compatible_attrs(self):
        if not hasattr(self, "remove_duplicate_token"):
            self.remove_duplicate_token = True
        if not hasattr(self, "discard_no_word"):
            self.discard_no_word = False
        if not hasattr(self, "plot_embeds"):
            self.plot_embeds = False
        if not hasattr(self, "proto_description_prompt_context_mode"):
            self.proto_description_prompt_context_mode = "sentence_neighbors"
        if not hasattr(self, "proto_description_candidate_limit"):
            self.proto_description_candidate_limit = 5
        if not hasattr(self, "proto_description_batch_size"):
            self.proto_description_batch_size = 32
        if not hasattr(self, "proto_description_use_detailed_description"):
            self.proto_description_use_detailed_description = False
        if not hasattr(self, "proto_description_require_detailed_description"):
            self.proto_description_require_detailed_description = True
        if not hasattr(self, "proto_description_exact_match_text"):
            self.proto_description_exact_match_text = True
        if not hasattr(self, "proto_description_exact_match_first"):
            self.proto_description_exact_match_first = False
        if not hasattr(self, "predicted_proto_description_logs"):
            self.predicted_proto_description_logs = []
        if not hasattr(self, "deleted_merged_proto_logs"):
            self.deleted_merged_proto_logs = []
        if not hasattr(self, "proto_description_operation_logs"):
            self.proto_description_operation_logs = []
        if not hasattr(self, "proto_description_log_path"):
            self.proto_description_log_path = "proto_description_logs.txt"
        if not hasattr(self, "wikidata_no_result_logs"):
            self.wikidata_no_result_logs = []
        if not hasattr(self, "_wikidata_no_result_keys"):
            self._wikidata_no_result_keys = set()
        if not hasattr(self, "proto_description_model_name"):
            self.proto_description_model_name = "cross-encoder/nli-deberta-v3-large"
        if not hasattr(self, "proto_description_model"):
            self.proto_description_model = None
        for token_node in getattr(self, "token_nodes", []):
            if not hasattr(token_node, "is_multi_semantic"):
                token_node.is_multi_semantic = None
            if not hasattr(token_node, "descriptions"):
                token_node.descriptions = None
            if not hasattr(token_node, "wikidata_info_loaded"):
                token_node.wikidata_info_loaded = None
            if not hasattr(token_node, "span_occurrences"):
                token_node.span_occurrences = []
            for text_embedding in getattr(token_node, "embeds_buffer", []):
                if not hasattr(text_embedding, "span_start"):
                    text_embedding.span_start = None
                if not hasattr(text_embedding, "span_end"):
                    text_embedding.span_end = None
                if not hasattr(text_embedding, "span_text"):
                    text_embedding.span_text = None
            upgraded_anomaly_section = []
            for item in getattr(token_node, "anomaly_section", []):
                if isinstance(item, AnomalyTextEmbedding):
                    if not hasattr(item.text_embedding, "span_start"):
                        item.text_embedding.span_start = None
                    if not hasattr(item.text_embedding, "span_end"):
                        item.text_embedding.span_end = None
                    if not hasattr(item.text_embedding, "span_text"):
                        item.text_embedding.span_text = None
                    upgraded_anomaly_section.append(item)
                    continue
                if isinstance(item, tuple) and len(item) == 4:
                    embed, chunk_node, max_val, max_idx = item
                    upgraded_anomaly_section.append(
                        AnomalyTextEmbedding(
                            text_embedding=TextEmbedding(embed=embed, chunk_node=chunk_node),
                            max_val=max_val,
                            max_idx=max_idx,
                        )
                    )
                    continue
                upgraded_anomaly_section.append(item)
            token_node.anomaly_section = upgraded_anomaly_section
        for proto_node in getattr(self, "proto_nodes", []):
            if not hasattr(proto_node, "span_occurrences"):
                proto_node.span_occurrences = []
            if not hasattr(proto_node, "retained_text_embeddings") or proto_node.retained_text_embeddings is None:
                proto_node.retained_text_embeddings = []
            if not hasattr(proto_node, "retained_text_embedding_source_count"):
                proto_node.retained_text_embedding_source_count = len(proto_node.retained_text_embeddings)
            if not hasattr(proto_node, "pending_embed_rebuild"):
                proto_node.pending_embed_rebuild = False
            for text_embedding in proto_node.retained_text_embeddings:
                if not hasattr(text_embedding, "span_start"):
                    text_embedding.span_start = None
                if not hasattr(text_embedding, "span_end"):
                    text_embedding.span_end = None
                if not hasattr(text_embedding, "span_text"):
                    text_embedding.span_text = None

    def debug_extract_important_spans(
        self,
        chunk,
        min_tokens=2,
        remove_duplicate=None,
        discard_no_word=None,
        debug_mode=True,
        clean_input=False,
    ):
        self._ensure_backward_compatible_attrs()

        if self.nlp is None:
            self._load_nlp()

        if remove_duplicate is None:
            remove_duplicate = getattr(self, "remove_duplicate_token", True)

        if discard_no_word is None:
            discard_no_word = getattr(self, "discard_no_word", False)

        if clean_input:
            chunk = clean_text(chunk)

        return extract_important_spans(
            chunk,
            self.nlp,
            min_tokens=min_tokens,
            remove_duplicate=remove_duplicate,
            discard_no_word=discard_no_word,
            debug_mode=debug_mode,
        )

    def build_query_database(self):
        embeds_list = [prototype.embed for prototype in self.proto_nodes]
        database = torch.stack(embeds_list).to(self.device)
        self.query_database = F.normalize(database, dim=1)

    def _prepare_query_tokens(self, query_text, print_important_tokens=False):
        query_text = clean_text(query_text)
        token_embeddings, offsets = encode_text(query_text, self.text_encoder, self.tokenizer, self.device)
        important_phrases, num_ents = extract_important_phrases(query_text, self.nlp)
        important_tokens = extract_important_tokens(query_text, self.nlp)
        tokens_for_processing = (
            [("ent", phrase, start_char, end_char) for phrase, start_char, end_char in important_phrases[:num_ents]] +
            [("phrase", phrase, start_char, end_char) for phrase, start_char, end_char in important_phrases[num_ents:]] +
            [("token", token, start_char, end_char) for token, start_char, end_char in important_tokens]
        )

        if print_important_tokens:
            print(f"important ents: {[text for text, _, _ in important_phrases[:num_ents]]}")
            print(f"important phrases: {[text for text, _, _ in important_phrases[num_ents:]]}")
            print(f"important tokens: {[text for text, _, _ in important_tokens]}")

        return query_text, token_embeddings, offsets, tokens_for_processing

    def _resolve_query_matches(self, query_text, search_mode="broad", print_important_tokens=False):
        query_text, token_embeddings, offsets, tokens_for_processing = self._prepare_query_tokens(
            query_text,
            print_important_tokens=print_important_tokens,
        )
        query_tokens = []
        tokens_in_phrase = []
        resolved_matches = []

        for token_type, token, start_char, end_char in tokens_for_processing:
            if token_type == "token" and token in tokens_in_phrase:
                continue

            token_embed = get_embed_by_offest(token_embeddings, offsets, (token, start_char, end_char))
            token_node = self.query_token_node(token)
            exact_proto = None
            fuzzy_protos = []
            semantic_proto = None
            semantic_weight = None

            if token_node is not None:
                exact_proto = token_node.proto_node_list[self.get_max_sim_proto(token_node, token_embed)]
                if token_type in {"phrase", "ent"} and search_mode != "broad":
                    tokens_in_phrase.extend(token.split(" "))

            fuzzy_query_list = self.phrase_fuzzy_query(token)
            if fuzzy_query_list:
                fuzzy_protos = self.max_cosine_prototype(token_embed, fuzzy_query_list, k=2)

            if exact_proto is None and not fuzzy_protos:
                proto_node_indices, weights = self.query_by_sim([token_embed])
                semantic_proto = self.proto_nodes[proto_node_indices[0]]
                semantic_weight = weights[0]

            if exact_proto is not None and token_type in {"phrase", "ent"} and search_mode == "broad":
                tokens_in_phrase.extend(token.split(" "))

            query_tokens.append(token)
            resolved_matches.append({
                "token_type": token_type,
                "token": token,
                "token_embed": token_embed,
                "exact_proto": exact_proto,
                "fuzzy_protos": fuzzy_protos,
                "semantic_proto": semantic_proto,
                "semantic_weight": semantic_weight,
            })

        return query_text, query_tokens, resolved_matches

    def broad_search_query(self, query_text, top_k=10,candidate=30):
        query_text, _, resolved_matches = self._resolve_query_matches(query_text)
        tokens_in_phrase = []
        retrieved_proto_nodes = []
        for match in resolved_matches:
            token_type = match["token_type"]
            token = match["token"]
            if token_type == "token" and token in tokens_in_phrase:
                continue
            if match["exact_proto"] is not None:
                retrieved_proto_nodes.append(match["exact_proto"])
                if token_type in {"phrase", "ent"}:
                    tokens_in_phrase.extend(token.split(" "))
            if match["fuzzy_protos"]:
                retrieved_proto_nodes.extend(match["fuzzy_protos"])
            elif match["semantic_proto"] is not None:
                retrieved_proto_nodes.append(match["semantic_proto"])
        retrieved_chunk_ids = []
        for proto_node in retrieved_proto_nodes:
            for chunk_node in proto_node.chunk_node_list:
                retrieved_chunk_ids.append(chunk_node.chunk_node_id)
        retrieved_chunk_ids = list(set(retrieved_chunk_ids))
        retrieved_chunks = self.chunk_id2text(retrieved_chunk_ids)[:candidate]
        rerank_chunks, rerank_chunk_index= self.reranker.rerank(query_text, retrieved_chunks, top_k=top_k)
        rerank_chunk_ids = [retrieved_chunk_ids[i] for i in rerank_chunk_index]

        return rerank_chunks, rerank_chunk_ids

    def multi_level_query(self, query_text, top_k_chunk=10, top_k_each_isolated_chunk=2, isolate_chunk_ratio=0.2, isolate_retrieve_mode='sequential',print_important_tokens=True, search_mode='broad'):
        query_text, query_tokens, resolved_matches = self._resolve_query_matches(
            query_text,
            search_mode=search_mode,
            print_important_tokens=print_important_tokens,
        )
        query_tokens = []
        low_level_tokens = []
        high_level_tokens = []
        low_level_protos = []
        high_level_protos = []
        tokens_in_phrase = []
        for match in resolved_matches:
            token_type = match["token_type"]
            token = match["token"]
            if token_type == "token" and token in tokens_in_phrase:
                continue
            query_tokens.append(token)
            exact_match = match["exact_proto"] is not None
            fuzzy_match = len(match["fuzzy_protos"]) > 0
            if exact_match:
                max_proto_node = match["exact_proto"]
                low_level_tokens.append(token + f"(exact matched)")
                if token_type == 'ent':
                    low_level_protos.append((max_proto_node, 0, 1))
                else:
                    low_level_protos.append((max_proto_node, 1, 1))
                if (token_type == 'phrase' or token_type == 'ent') and search_mode != 'broad':
                    tokens_in_phrase.extend(token.split(" "))
            if fuzzy_match:
                for proto_node in match["fuzzy_protos"][:1]:
                    weight = count_words(token) / count_words(proto_node.token_node.token_text)
                    if exact_match:
                        if token_type == 'ent':
                            high_level_protos.append((proto_node,1, weight))
                        else:
                            high_level_protos.append((proto_node, 2, weight))
                        high_level_tokens.append(
                            proto_node.token_node.token_text + f"(partial matched)")
                    else:
                        if token_type == 'ent':
                            low_level_protos.append((proto_node,1, weight))
                        else:
                            low_level_protos.append((proto_node,2, weight))
                        low_level_tokens.append(
                            proto_node.token_node.token_text + f"(partial matched)")
            if not (exact_match or fuzzy_match):
                low_level_protos.append((match["semantic_proto"], 3, match["semantic_weight"]))
                low_level_tokens.append(match["semantic_proto"].token_node.token_text + f"(sim matched)")

                high_level_tokens.append(['N/A'])

        if print_important_tokens:
            print(f"query tokens: {[text for text in query_tokens]}")
            print(f"low level tokens: {[text for text in low_level_tokens]}")
            print(f"high level tokens: {[text for text in high_level_tokens]}")

        cog = CoOccurrenceGraph(low_level_protos + high_level_protos)
        cog.build_edges()
        cog.assign_chunk_weight(self.chunk_avg_len,print_important_tokens)
        cog.rank_proto_node()
        cog.rank_chunk_by_BM25()


        num_isolate_chunk = math.floor(top_k_chunk * isolate_chunk_ratio)
        num_connected_chunk = top_k_chunk - num_isolate_chunk

        retrieved_connected_chunk = [chunk_id for chunk_id, _ in cog.weighted_chunk_node_list[:num_connected_chunk]]
        connect_chunk_full = True if len(retrieved_connected_chunk) == num_connected_chunk else False
        #ranked_chunks_by_protos = [self.get_top_k_chunk(proto_node, retrieve_all=True)[1] for proto_node in cog.ranked_proto_list]

        isolate_chunk_extractor = ListBatchExtractor(cog.ranked_chunk_BM25, mode=isolate_retrieve_mode, k=top_k_each_isolated_chunk,exclude_list=retrieved_connected_chunk)
        retrieved_isolated_chunk = isolate_chunk_extractor.extract(num_isolate_chunk, [])
        isolate_chunk_full = not isolate_chunk_extractor.finished

        if connect_chunk_full != isolate_chunk_full:
            if not connect_chunk_full:
                retrieved_isolated_chunk = isolate_chunk_extractor.extract(top_k_chunk-len(retrieved_connected_chunk), retrieved_isolated_chunk)
            else:
                retrieved_connected_chunk = [chunk_id for chunk_id, _ in
                                             cog.weighted_chunk_node_list[:(top_k_chunk - num_isolate_chunk)]]


        return self.chunk_id2text(retrieved_connected_chunk + retrieved_isolated_chunk), retrieved_connected_chunk + retrieved_isolated_chunk, cog


    def get_top_k_chunk(self, proto_node, top_k=None, retrieve_all=False):
        top_k = self.retrieve_top_k if top_k is None else top_k
        sorted_indices = sorted(
            range(len(proto_node.chunk_edge_weight)),
            key=lambda i: proto_node.chunk_edge_weight[i],
            reverse=True
        )
        result = []
        result_chunk_id = []
        seen = set()
        for idx in sorted_indices:
            node = proto_node.chunk_node_list[idx]
            if node not in seen:
                result.append(node.chunk_text)
                result_chunk_id.append(node.chunk_node_id)
                seen.add(node)
            if len(result) == top_k and not retrieve_all:
                break
        return result, result_chunk_id

    def query_by_sim(self, query_embeds):
        query_tensor = torch.stack(query_embeds).to(self.device)  # (N, D)
        query_tensor = F.normalize(query_tensor, dim=1)
        sims = torch.matmul(self.query_database, query_tensor.T)
        best_scores, best_indices = torch.max(sims, dim=0)

        return best_indices.tolist(), best_scores.tolist()

    def finalize(self):
        self._reset_proto_description_logs()
        self.log_time("Finalize started.")
        self.chunk_avg_len = sum([chunk_node.num_tokens for chunk_node in self.chunk_nodes])/len(self.chunk_nodes)
        self.log_time("Computed average chunk length.")
        self.finalize_token_nodes()
        self.log_time("Finished token node finalization.")
        self.merge_duplicate_description_proto_nodes()
        self.log_time("Finished merging proto nodes by description.")
        for token_node in self.token_nodes:
            self.assign_idf(token_node)
        self.log_time("Finished assigning token and proto IDF.")
        self.get_proto_BM25()
        self.log_time("Finished computing proto BM25.")
        self.build_query_database()
        self.log_time("Finished building query database.")
        self.build_phrase_query()
        self.log_time("Finished building phrase query index.")
        self.build_chunk2proto_edge()
        self.log_time("Finished building chunk-to-proto edges.")
        self.save_doc_to_json()
        self.log_time("Finalizing completed.")

    def finalize_token_nodes(self):
        total_token_nodes = len(self.token_nodes)
        if total_token_nodes == 0:
            return

        progress_update = self._make_progress_updater("finalize_token_nodes", total_token_nodes)
        progress_update(0, force=True)
        for index, token_node in enumerate(self.token_nodes, start=1):
            if not token_node.has_prototype:
                self.create_basic_proto_node(token_node)
            elif len(token_node.anomaly_section) > 0:
                for item in token_node.anomaly_section:
                    self._append_proto_occurrence(
                        token_node.proto_node_list[item.max_idx],
                        item.text_embedding,
                        edge_weight=item.max_val,
                    )
                token_node.anomaly_section.clear()
            token_node.embeds_buffer.clear()
            progress_update(index)
        progress_update(total_token_nodes, force=True)
        print()

    def _sample_proto_chunk_nodes(self, proto_node, max_samples=10):
        unique_chunk_nodes = []
        seen_chunk_ids = set()
        for chunk_node in proto_node.chunk_node_list:
            if chunk_node.chunk_node_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_node.chunk_node_id)
            unique_chunk_nodes.append(chunk_node)
        if max_samples is None or max_samples <= 0:
            return unique_chunk_nodes
        if len(unique_chunk_nodes) <= max_samples:
            return unique_chunk_nodes
        return random.sample(unique_chunk_nodes, max_samples)

    def _sample_proto_span_occurrences(self, proto_node, max_samples=10):
        span_occurrences = list(getattr(proto_node, "span_occurrences", []))
        if span_occurrences:
            if max_samples is None or max_samples <= 0:
                return span_occurrences
            if len(span_occurrences) <= max_samples:
                return span_occurrences
            return random.sample(span_occurrences, max_samples)

        fallback_chunk_nodes = self._sample_proto_chunk_nodes(proto_node, max_samples=max_samples)
        return [SpanOccurrence(chunk_node=chunk_node) for chunk_node in fallback_chunk_nodes]

    def _find_left_boundary_for_description(self, text, index):
        return max(
            text.rfind(".", 0, index),
            text.rfind("!", 0, index),
            text.rfind("?", 0, index),
        )

    def _find_right_boundary_for_description(self, text, index):
        right_candidates = [
            text.find(".", index),
            text.find("!", index),
            text.find("?", index),
        ]
        right_candidates = [idx for idx in right_candidates if idx != -1]
        return len(text) if not right_candidates else min(right_candidates) + 1

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

    def _find_description_match_span(self, chunk_text, token_text):
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

    def _extract_sentence_context_for_description(self, chunk_text, token_text, match_span=None):
        if match_span is None:
            match_span = self._find_description_match_span(chunk_text, token_text)
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

    def _extract_neighbor_sentence_context_for_description(self, chunk_text, token_text, match_span=None):
        if match_span is None:
            match_span = self._find_description_match_span(chunk_text, token_text)
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

    def _extract_full_context_for_description(self, chunk_text, token_text, match_span=None):
        if match_span is None:
            match_span = self._find_description_match_span(chunk_text, token_text)
        if match_span is None:
            return None
        return self._build_description_context_from_bounds(
            chunk_text,
            match_span,
            0,
            len(chunk_text),
        )

    def _extract_prompt_context_for_description(self, chunk_text, token_text, match_span=None):
        mode = self.proto_description_prompt_context_mode
        if mode == "sentence":
            return self._extract_sentence_context_for_description(chunk_text, token_text, match_span=match_span)
        if mode == "sentence_neighbors":
            return self._extract_neighbor_sentence_context_for_description(chunk_text, token_text, match_span=match_span)
        if mode == "full_text":
            return self._extract_full_context_for_description(chunk_text, token_text, match_span=match_span)
        raise ValueError(
            f"Unsupported proto_description_prompt_context_mode={mode!r}. "
            "Use 'sentence', 'sentence_neighbors', or 'full_text'."
        )

    def _build_proto_description_prompt(self, chunk_text, token_text, match_span=None):
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

    def _build_fallback_proto_description_prompt(self, chunk_text, token_text):
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

    def _load_proto_description_candidate_bank(self, token_text, candidate_bank_cache):
        cached_candidate_bank = candidate_bank_cache.get(token_text)
        if cached_candidate_bank is not None:
            return cached_candidate_bank

        lookup_attempts = [
            (
                "default",
                self.proto_description_require_detailed_description,
                self.proto_description_exact_match_text,
            ),
            (
                "relax_require_detailed_description",
                False,
                self.proto_description_exact_match_text,
            ),
            (
                "relax_exact_match_text",
                False,
                False,
            ),
        ]

        seen_configs = set()
        for attempt_name, require_detailed_description, exact_match_text in lookup_attempts:
            config_key = (require_detailed_description, exact_match_text)
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)

            stage = f"proto_description:{attempt_name}"
            try:
                candidates_df, definition_column = load_wikidata_definition_candidates(
                    token_text,
                    use_detailed_description=self.proto_description_use_detailed_description,
                    exact_match_text=exact_match_text,
                    exact_match_first=self.proto_description_exact_match_first,
                    limit=self.proto_description_candidate_limit,
                    require_detailed_description=require_detailed_description,
                )
            except ValueError:
                self._log_wikidata_no_result(
                    token_text,
                    stage,
                    "no_candidate_definitions",
                )
                continue

            candidate_bank = build_wikidata_candidate_bank(candidates_df, definition_column=definition_column)
            if candidate_bank:
                candidate_bank_cache[token_text] = candidate_bank
                if attempt_name != "default":
                    self._log_proto_description_operation(
                        "fallback_wikidata_candidate_lookup",
                        token_text=token_text,
                        fallback=attempt_name,
                        require_detailed_description=require_detailed_description,
                        exact_match_text=exact_match_text,
                        candidate_count=len(candidate_bank),
                    )
                return candidate_bank

            self._log_wikidata_no_result(
                token_text,
                stage,
                "empty_candidate_bank",
            )

        candidate_bank_cache[token_text] = []
        return []

    def _predict_proto_description_from_samples(
        self,
        proto_node,
        model,
        candidate_bank_cache,
        max_samples=10,
    ):
        token_text = proto_node.token_node.token_text
        candidate_bank = self._load_proto_description_candidate_bank(
            token_text,
            candidate_bank_cache,
        )

        if not candidate_bank:
            self._log_wikidata_no_result(
                token_text,
                "proto_description",
                "empty_candidate_bank",
            )
            return None

        description_vote_map = {}
        sample_prediction_records = []
        sample_span_occurrences = self._sample_proto_span_occurrences(proto_node, max_samples=max_samples)
        for sample_index, span_occurrence in enumerate(sample_span_occurrences, start=1):
            chunk_node = span_occurrence.chunk_node
            prompt_info = self._build_proto_description_prompt(
                chunk_node.chunk_text,
                token_text,
                match_span=span_occurrence.get_span_tuple(),
            )
            if prompt_info is None:
                prompt_info = self._build_fallback_proto_description_prompt(
                    chunk_node.chunk_text,
                    token_text,
                )
            if prompt_info is None:
                continue

            pairs = [(prompt_info["prompt_text"], candidate["hypothesis"]) for candidate in candidate_bank]
            raw_scores = model.predict(
                pairs,
                batch_size=min(self.proto_description_batch_size, len(pairs)),
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
            if ranked_candidates:
                top_candidate = ranked_candidates[0]
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
                state["count"] += 1
                state["score_sum"] += top_candidate["score"]
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
                        "prediction_score": top_candidate["score"],
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

    def _get_span_occurrence_key(self, span_occurrence):
        chunk_node = span_occurrence.chunk_node
        return (
            id(chunk_node),
            span_occurrence.span_start,
            span_occurrence.span_end,
            span_occurrence.span_text,
        )

    def _build_retained_text_embedding_lookup(self, proto_node):
        lookup = defaultdict(list)
        for text_embedding in getattr(proto_node, "retained_text_embeddings", []):
            lookup[self._get_span_occurrence_key(text_embedding.to_span_occurrence())].append(text_embedding)
        return lookup

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

        return self._find_description_match_span(chunk_text, token_text)

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
        )

    def _finalize_split_proto_node_embed(self, proto_node):
        available_embeds = list(proto_node.chunk_node_embed)
        if available_embeds:
            proto_node.embed = average_embeds(available_embeds)
        else:
            sample_occurrences = list(getattr(proto_node, "span_occurrences", []))
            if sample_occurrences:
                sample_size = min(len(sample_occurrences), self.proto_retained_embed_limit)
                regenerated_text_embeddings = []
                for span_occurrence in random.sample(sample_occurrences, sample_size):
                    text_embedding = self._reencode_text_embedding_for_occurrence(
                        proto_node.token_node,
                        span_occurrence,
                    )
                    if text_embedding is not None:
                        regenerated_text_embeddings.append(text_embedding)

                if regenerated_text_embeddings:
                    proto_node.embed = average_embeds(
                        [text_embedding.embed for text_embedding in regenerated_text_embeddings]
                    )
                    self._initialize_proto_retained_text_embeddings(
                        proto_node,
                        regenerated_text_embeddings,
                    )

        if proto_node.embed is None:
            raise ValueError(
                f"Failed to rebuild embed for split proto node {proto_node.proto_node_id} "
                f"({proto_node.token_node.token_text!r}, description={proto_node.description!r})."
            )

        if proto_node.chunk_edge_weight:
            proto_node.anomaly_threshold = get_anomaly_threshold(
                proto_node.chunk_edge_weight,
                self.anomaly_threshold_percentile,
            )
        else:
            proto_node.anomaly_threshold = None
        proto_node.chunk_node_embed.clear()
        proto_node.pending_embed_rebuild = False

    def _create_split_proto_node(self, token_node, description):
        new_proto_node = self.create_proto_node(token_node)
        new_proto_node.description = description
        new_proto_node.chunk_node_list = []
        new_proto_node.span_occurrences = []
        new_proto_node.chunk_node_embed = []
        new_proto_node.retained_text_embeddings = []
        new_proto_node.retained_text_embedding_source_count = 0
        new_proto_node.pending_embed_rebuild = True
        new_proto_node.chunk_edge_weight = []
        new_proto_node.embed = None
        new_proto_node.anomaly_threshold = None
        new_proto_node.tf_dict_by_chunk_id = None
        new_proto_node.chunk_len_dict_by_id = None
        new_proto_node.BM25 = None
        new_proto_node.df = 0
        new_proto_node.idf = 0
        return new_proto_node

    def _merge_proto_node_group(self, proto_group):
        primary_proto = proto_group[0]
        merged_chunk_nodes = []
        merged_chunk_edge_weights = []
        merged_span_occurrences = []
        merged_embeds = []
        merged_retained_text_embeddings = []
        merged_retained_source_count = 0

        for proto_node in proto_group:
            merged_chunk_nodes.extend(proto_node.chunk_node_list)
            merged_chunk_edge_weights.extend(proto_node.chunk_edge_weight)
            merged_span_occurrences.extend(getattr(proto_node, "span_occurrences", []))
            merged_retained_text_embeddings.extend(getattr(proto_node, "retained_text_embeddings", []))
            merged_retained_source_count += getattr(
                proto_node,
                "retained_text_embedding_source_count",
                len(getattr(proto_node, "retained_text_embeddings", [])),
            )
            if proto_node.embed is not None:
                merged_embeds.append(proto_node.embed)

        primary_proto.chunk_node_list = merged_chunk_nodes
        primary_proto.chunk_edge_weight = merged_chunk_edge_weights
        primary_proto.span_occurrences = merged_span_occurrences
        primary_proto.chunk_node_embed = []
        primary_proto.retained_text_embeddings = self._sample_text_embeddings(
            merged_retained_text_embeddings,
            max_samples=self.proto_retained_embed_limit,
        )
        primary_proto.retained_text_embedding_source_count = max(
            merged_retained_source_count,
            len(merged_retained_text_embeddings),
        )
        primary_proto.pending_embed_rebuild = False
        retained_embeds = [
            text_embedding.embed
            for text_embedding in merged_retained_text_embeddings
            if text_embedding.embed is not None
        ]
        if retained_embeds:
            primary_proto.embed = average_embeds(retained_embeds)
        elif merged_embeds:
            primary_proto.embed = torch.stack(merged_embeds).mean(dim=0)
        if primary_proto.chunk_edge_weight:
            primary_proto.anomaly_threshold = get_anomaly_threshold(
                primary_proto.chunk_edge_weight,
                self.anomaly_threshold_percentile,
            )
        else:
            primary_proto.anomaly_threshold = None
        primary_proto.tf_dict_by_chunk_id = None
        primary_proto.chunk_len_dict_by_id = None
        primary_proto.BM25 = None
        primary_proto.df = 0
        primary_proto.idf = 0
        return primary_proto

    def merge_duplicate_description_proto_nodes(self):
        consensus_ratio_threshold = 0.8
        redundant_proto_ids = set()
        proto_nodes_changed = False
        target_token_nodes = [
            token_node for token_node in self.token_nodes
            if len(token_node.proto_node_list) > 1
        ]
        candidate_bank_cache = {}

        total_token_nodes = len(target_token_nodes)
        progress_update = self._make_progress_updater(
            "merge_duplicate_description_proto_nodes",
            total_token_nodes,
        ) if total_token_nodes > 0 else None
        if progress_update is not None:
            progress_update(0, force=True)
        for index, token_node in enumerate(target_token_nodes, start=1):
            ordered_proto_list = []
            seen_proto_ids = set()
            for proto_node in token_node.proto_node_list:
                proto_id = id(proto_node)
                if proto_id in seen_proto_ids:
                    continue
                seen_proto_ids.add(proto_id)
                ordered_proto_list.append(proto_node)

            new_proto_list = []
            description_proto_map = defaultdict(list)
            for proto_node in ordered_proto_list:
                if proto_node.description:
                    new_proto_list.append(proto_node)
                    description_proto_map[proto_node.description].append(proto_node)
                    continue

                if self.proto_description_model is None:
                    new_proto_list.append(proto_node)
                    continue

                prediction_result = self._predict_proto_description_from_samples(
                    proto_node,
                    model=self.proto_description_model,
                    candidate_bank_cache=candidate_bank_cache,
                    max_samples=10,
                )
                if prediction_result is None:
                    new_proto_list.append(proto_node)
                    continue

                if (
                    prediction_result["sample_count"] > 0
                    and prediction_result["top_description_ratio"] >= consensus_ratio_threshold
                ):
                    predicted_description = prediction_result["description"]
                    proto_node.description = predicted_description
                    new_proto_list.append(proto_node)
                    description_proto_map[predicted_description].append(proto_node)
                    self.predicted_proto_description_logs.append(
                        {
                            "proto_node_id": proto_node.proto_node_id,
                            "token_text": proto_node.token_node.token_text,
                            "description": predicted_description,
                            "predicted_entity_id": prediction_result.get("predicted_entity_id"),
                            "predicted_label": prediction_result.get("predicted_label"),
                            "predicted_definition": prediction_result.get("predicted_definition"),
                            "prediction_score_mean": prediction_result.get("prediction_score_mean"),
                            "chunk_count": len(proto_node.chunk_node_list),
                            "sample_count": prediction_result.get("sample_count"),
                            "top_description_count": prediction_result.get("top_description_count"),
                            "top_description_ratio": prediction_result.get("top_description_ratio"),
                            "sample_predictions": prediction_result["sample_predictions"],
                        }
                    )
                    self._log_proto_description_operation(
                        "assign_description_by_consensus",
                        token_text=token_node.token_text,
                        proto_node_id=proto_node.proto_node_id,
                        description=predicted_description,
                        sample_count=prediction_result.get("sample_count"),
                        top_description_count=prediction_result.get("top_description_count"),
                        top_description_ratio=prediction_result.get("top_description_ratio"),
                    )
                    continue

                exhaustive_prediction_result = self._predict_proto_description_from_samples(
                    proto_node,
                    model=self.proto_description_model,
                    candidate_bank_cache=candidate_bank_cache,
                    max_samples=None,
                )
                if exhaustive_prediction_result is None:
                    new_proto_list.append(proto_node)
                    continue

                proto_nodes_changed = True
                redundant_proto_ids.add(id(proto_node))
                self._log_proto_description_operation(
                    "split_proto_node_by_sample_labels",
                    token_text=token_node.token_text,
                    source_proto_node_id=proto_node.proto_node_id,
                    source_chunk_count=len(proto_node.chunk_node_list),
                    description_counts=exhaustive_prediction_result.get("description_counts", {}),
                )
                retained_text_embedding_lookup = self._build_retained_text_embedding_lookup(proto_node)
                self.deleted_merged_proto_logs.append(
                    {
                        "deleted_proto_node_id": proto_node.proto_node_id,
                        "kept_proto_node_id": None,
                        "token_text": token_node.token_text,
                        "description": "split_by_sample_label",
                        "deleted_chunk_count": len(proto_node.chunk_node_list),
                    }
                )

                for sample_prediction in exhaustive_prediction_result["sample_prediction_records"]:
                    predicted_description = sample_prediction["predicted_description"]
                    if not predicted_description:
                        continue
                    span_occurrence = sample_prediction["span_occurrence"]
                    retained_candidates = retained_text_embedding_lookup.get(
                        self._get_span_occurrence_key(span_occurrence),
                    )
                    retained_text_embedding = None if not retained_candidates else retained_candidates.pop()
                    target_proto_group = description_proto_map.get(predicted_description)
                    created_new_proto = False
                    if target_proto_group:
                        target_proto = target_proto_group[0]
                    else:
                        target_proto = self._create_split_proto_node(token_node, predicted_description)
                        description_proto_map[predicted_description].append(target_proto)
                        new_proto_list.append(target_proto)
                        created_new_proto = True
                        self._log_proto_description_operation(
                            "create_split_proto_node",
                            token_text=token_node.token_text,
                            proto_node_id=target_proto.proto_node_id,
                            description=predicted_description,
                        )
                    self._append_span_occurrence_to_proto(
                        target_proto,
                        span_occurrence,
                        edge_weight=0.0,
                        retained_text_embedding=retained_text_embedding,
                    )
                    self._log_proto_description_operation(
                        "assign_sample_to_proto_node",
                        token_text=token_node.token_text,
                        source_proto_node_id=proto_node.proto_node_id,
                        target_proto_node_id=target_proto.proto_node_id,
                        description=predicted_description,
                        chunk_node_id=span_occurrence.chunk_node.chunk_node_id,
                        span_start=span_occurrence.span_start,
                        span_end=span_occurrence.span_end,
                        span_text=span_occurrence.span_text,
                        used_retained_embedding=retained_text_embedding is not None,
                        created_new_proto=created_new_proto,
                    )

            for proto_node in new_proto_list:
                if getattr(proto_node, "pending_embed_rebuild", False):
                    self._finalize_split_proto_node_embed(proto_node)
                    self._log_proto_description_operation(
                        "finalize_split_proto_node_embed",
                        token_text=token_node.token_text,
                        proto_node_id=proto_node.proto_node_id,
                        description=proto_node.description,
                        retained_embed_count=len(getattr(proto_node, "retained_text_embeddings", [])),
                    )

            merged_proto_map = {}
            for description, proto_group in defaultdict(list, {
                description: [proto for proto in new_proto_list if proto.description == description]
                for description in {proto.description for proto in new_proto_list if proto.description}
            }).items():
                if len(proto_group) < 2:
                    continue
                merged_proto = self._merge_proto_node_group(proto_group)
                merged_proto_map[description] = merged_proto
                self._log_proto_description_operation(
                    "merge_proto_nodes_with_same_description",
                    token_text=token_node.token_text,
                    description=description,
                    kept_proto_node_id=merged_proto.proto_node_id,
                    merged_proto_node_ids=[proto.proto_node_id for proto in proto_group],
                    retained_embed_count=len(getattr(merged_proto, "retained_text_embeddings", [])),
                )
                for redundant_proto in proto_group[1:]:
                    proto_nodes_changed = True
                    redundant_proto_ids.add(id(redundant_proto))
                    self.deleted_merged_proto_logs.append(
                        {
                            "deleted_proto_node_id": redundant_proto.proto_node_id,
                            "kept_proto_node_id": merged_proto.proto_node_id,
                            "token_text": token_node.token_text,
                            "description": description,
                            "deleted_chunk_count": len(redundant_proto.chunk_node_list),
                        }
                    )

            if merged_proto_map:
                deduped_proto_list = []
                added_proto_ids = set()
                for proto_node in new_proto_list:
                    if id(proto_node) in redundant_proto_ids:
                        continue
                    target_proto = merged_proto_map.get(proto_node.description, proto_node)
                    target_proto_id = id(target_proto)
                    if target_proto_id in added_proto_ids:
                        continue
                    added_proto_ids.add(target_proto_id)
                    deduped_proto_list.append(target_proto)
                new_proto_list = deduped_proto_list

            token_node.proto_node_list = new_proto_list
            token_node.has_prototype = len(token_node.proto_node_list) > 0
            progress_update(index)

        if not redundant_proto_ids and not proto_nodes_changed:
            if total_token_nodes > 0:
                progress_update(total_token_nodes, force=True)
                print()
            return

        self.proto_nodes = [
            proto_node for proto_node in self.proto_nodes
            if id(proto_node) not in redundant_proto_ids
        ]
        for index, proto_node in enumerate(self.proto_nodes):
            proto_node.proto_node_id = index
        self.next_proto_node_id = len(self.proto_nodes)
        if total_token_nodes > 0:
            progress_update(total_token_nodes, force=True)
            print()

    def _format_proto_description_logs_text(self):
        lines = []

        lines.append("Proto description operations:")
        if not self.proto_description_operation_logs:
            lines.append("  (none)")
        else:
            for item in self.proto_description_operation_logs:
                event_type = item.get("event_type", "unknown")
                payload = ", ".join(
                    f"{key}={value!r}"
                    for key, value in item.items()
                    if key != "event_type"
                )
                lines.append(f"  [{event_type}] {payload}")

        lines.append("")
        lines.append("Proto nodes assigned predicted descriptions:")
        if not self.predicted_proto_description_logs:
            lines.append("  (none)")
        else:
            for item in self.predicted_proto_description_logs:
                lines.append(
                    "[Proto] proto_node_id={} | token={!r} | description={!r} | chunk_count={}".format(
                        item["proto_node_id"],
                        item["token_text"],
                        item["description"],
                        item["chunk_count"],
                    )
                )
                sample_predictions = item.get("sample_predictions", [])
                if not sample_predictions:
                    lines.append("  sample_predictions: (none)")
                    lines.append("")
                    continue
                for sample in sample_predictions:
                    lines.append(
                        "  (sample {}) chunk_node_id={} | predicted_description={!r} | predicted_label={!r} | score={:.4f}".format(
                            sample["sample_index"],
                            sample["chunk_node_id"],
                            sample["predicted_description"],
                            sample["predicted_label"],
                            sample["prediction_score"],
                        )
                    )
                    lines.append("    matched_text={!r}".format(sample["matched_text"]))
                    lines.append("    context_text={!r}".format(sample["context_text"]))
                lines.append("")

        lines.append("Proto nodes deleted by description merge:")
        if not self.deleted_merged_proto_logs:
            lines.append("  (none)")
        else:
            for item in self.deleted_merged_proto_logs:
                lines.append(
                    "  deleted_proto_node_id={} | kept_proto_node_id={} | token={!r} | description={!r} | deleted_chunk_count={}".format(
                        item["deleted_proto_node_id"],
                        item["kept_proto_node_id"],
                        item["token_text"],
                        item["description"],
                        item["deleted_chunk_count"],
                    )
                )

        lines.append("")
        lines.append("Wikidata lookups with no usable results:")
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

    def print_proto_description_logs(self, output_path=None):
        return self.show_proto_description_logs(output_path=output_path)

    def show_proto_description_logs(self, as_html=True, open_details=False, output_path=None):
        output_path = self.proto_description_log_path if output_path is None else output_path
        log_text = self._format_proto_description_logs_text()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(log_text)
        return output_path


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

    def get_proto_BM25(self):
        for proto_node in self.proto_nodes:
            proto_node.get_BM25(self.chunk_avg_len)

    def process_embeds(self, new_chunk_node, phrase_embs, token_embs):
        for text, embed, span_start, span_end in phrase_embs + token_embs:
            text_embedding = self._make_text_embedding(
                embed,
                new_chunk_node,
                span_start=span_start,
                span_end=span_end,
            )
            token_node = self.query_token_node(text)
            if token_node is None:
                token_node = self.create_token_node(text)
                token_node.embeds_buffer.append(text_embedding)
            else:
                if token_node.has_prototype:
                    max_val, max_idx = inspect_prototypes(embed, token_node.proto_node_list)
                    if max_val >= token_node.proto_node_list[max_idx].anomaly_threshold:
                        self._append_proto_occurrence(
                            token_node.proto_node_list[max_idx],
                            text_embedding,
                            edge_weight=max_val,
                        )
                    else:
                        token_node.anomaly_section.append(
                            AnomalyTextEmbedding(
                                text_embedding=text_embedding,
                                max_val=max_val,
                                max_idx=max_idx,
                            )
                        )
                else:
                    token_node.embeds_buffer.append(text_embedding)
            self._append_token_occurrence(token_node, text_embedding)
            if len(token_node.embeds_buffer) == self.buffer_size:
                self.build_proto_waitlist.append(token_node)

    def index_document(self, doc_name, multiprocessing=True):
        self.start_time = time.perf_counter()
        document_start_time = time.perf_counter()
        if multiprocessing:
            self.index_document_multi_processing(doc_name)
        else:
            self.index_document_single_processing(doc_name)
        self.log_time(f"File {doc_name} completed. Index time: {time.perf_counter() - document_start_time:.4f}s")


    def index_document_single_processing(self, doc_name):
        new_doc_node = self.create_doc_node(doc_name)
        chunk_list = split_doc(doc_name, self.nlp, max_tokens=self.chunk_size)
        token_embeddings_list, offsets_list = encode_chunk_batch(chunk_list, self.text_encoder, self.tokenizer, self.device)
        for chunk, token_embeddings, offsets in zip(chunk_list, token_embeddings_list, offsets_list):
            new_chunk_node = self.create_chunk_node(chunk, new_doc_node)
            phrases, tokens = extract_important_spans(chunk, self.nlp, min_tokens=2,
                                                      remove_duplicate=self.remove_duplicate_token)
            phrase_embs, token_embs = get_token_embeds(token_embeddings, offsets, phrases, tokens)
            self.process_embeds(new_chunk_node, phrase_embs, token_embs)
        self.solve_proto()
        self.solve_anomaly()

    def index_document_multi_processing(self, doc_name):
        new_doc_node = self.create_doc_node(doc_name)
        chunk_list = split_doc(doc_name, self.nlp, max_tokens=self.chunk_size)
        chunk_meta = []
        futures = []

        for chunk in chunk_list:
            futures.append(self.executor.submit(encode_chunk, chunk, self.text_encoder, self.tokenizer, self.device))
            new_chunk_node = self.create_chunk_node(chunk, new_doc_node)
            phrases, tokens = extract_important_spans(chunk, self.nlp, min_tokens=2, remove_duplicate=self.remove_duplicate_token)
            chunk_meta.append((new_chunk_node, phrases, tokens))

        for i, future in enumerate(futures):
            token_embeddings, offsets = future.result()
            node, phrases, tokens = chunk_meta[i]
            phrase_embs, token_embs = get_token_embeds(
                token_embeddings,
                offsets,
                phrases,
                tokens
            )

            self.process_embeds(node, phrase_embs, token_embs)
        self.solve_proto()
        self.solve_anomaly()

    def plot_embed_distribution(self, token_node, clusters):
        embeds = [k.embed.cpu() for k in token_node.embeds_buffer]
        plot_embeddings(embeds, token_node.token_text, clusters)

    def print_memory_size(self):
        print_size_mb(self)

    def build_phrase_query(self):
        for i, token_nodes in enumerate(self.phrase_token_nodes):
            words = token_nodes.token_text.split()
            for w in words:
                self.phrase_index[w].add(i)

    def phrase_fuzzy_query(self, text: str):
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

    def max_cosine_prototype(
            self,
            query_tensor,
            token_node_list,
            k=1,
            index_as_input=True
    ):
        """
        query_tensor: shape (d,)
        token_node_list: list of token_node indices or objects
        k: number of top results to return

        return:
            topk_prototypes: list of prototype_node
            topk_similarities: tensor of shape (k,)
        """

        all_embeds = []
        prototype_map = []

        for item in token_node_list:
            if index_as_input:
                token_node = self.phrase_token_nodes[item]
            else:
                token_node = item

            for prototype_node in token_node.proto_node_list:
                all_embeds.append(prototype_node.embed)
                prototype_map.append(prototype_node)

        if len(all_embeds) == 0:
            return [], torch.tensor([])

        matrix = torch.stack(all_embeds).to(self.device)  # (N, d)

        query_norm = F.normalize(query_tensor.unsqueeze(0), dim=1)  # (1, d)
        matrix_norm = F.normalize(matrix, dim=1)  # (N, d)

        similarities = torch.mm(query_norm, matrix_norm.t()).squeeze(0)  # (N,)

        k = min(k, similarities.size(0))

        topk_similarities, topk_indices = torch.topk(similarities, k)

        topk_prototypes = [prototype_map[i] for i in topk_indices.tolist()]

        return topk_prototypes

    def build_chunk2proto_edge(self):
        for chunk_node in self.chunk_nodes:
            chunk_node.proto_node_list = []
        for proto_node in self.proto_nodes:
            for chunk_node in proto_node.chunk_node_list:
                if not id(proto_node) in (id(x) for x in chunk_node.proto_node_list):
                    chunk_node.proto_node_list.append(proto_node)

    def rebuild_metadata_after_deletion(self):
        valid_chunk_ids = {id(chunk_node) for chunk_node in self.chunk_nodes}

        for doc_node in self.doc_nodes:
            doc_node.chunk_node_list = [
                chunk_node for chunk_node in doc_node.chunk_node_list
                if id(chunk_node) in valid_chunk_ids
            ]

        valid_proto_nodes = []
        for proto_node in self.proto_nodes:
            keep_indices = [
                i for i, chunk_node in enumerate(proto_node.chunk_node_list)
                if id(chunk_node) in valid_chunk_ids
            ]
            proto_node.chunk_node_list = [proto_node.chunk_node_list[i] for i in keep_indices]
            proto_node.span_occurrences = [
                proto_node.span_occurrences[i]
                for i in keep_indices
            ] if getattr(proto_node, "span_occurrences", None) else []
            proto_node.chunk_edge_weight = [proto_node.chunk_edge_weight[i] for i in keep_indices]
            proto_node.retained_text_embeddings = [
                text_embedding for text_embedding in getattr(proto_node, "retained_text_embeddings", [])
                if id(text_embedding.chunk_node) in valid_chunk_ids
            ]
            proto_node.retained_text_embedding_source_count = max(
                proto_node.retained_text_embedding_source_count,
                len(proto_node.retained_text_embeddings),
            )
            if proto_node.chunk_node_list:
                valid_proto_nodes.append(proto_node)

        self.proto_nodes = valid_proto_nodes
        for index, proto_node in enumerate(self.proto_nodes):
            proto_node.proto_node_id = index
        self.next_proto_node_id = len(self.proto_nodes)

        valid_proto_ids = {id(proto_node) for proto_node in self.proto_nodes}
        valid_token_nodes = []
        phrase_token_nodes = []
        token_node_query = {}

        for token_node in self.token_nodes:
            token_node.proto_node_list = [
                proto_node for proto_node in token_node.proto_node_list
                if id(proto_node) in valid_proto_ids
            ]
            token_node.has_prototype = len(token_node.proto_node_list) > 0
            token_node.anomaly_section = [
                item for item in token_node.anomaly_section
                if id(item.text_embedding.chunk_node) in valid_chunk_ids
            ]
            token_node.span_occurrences = [
                item for item in token_node.span_occurrences
                if id(item.chunk_node) in valid_chunk_ids
            ]
            if token_node.has_prototype or token_node.embeds_buffer or token_node.anomaly_section:
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

        self.build_proto_waitlist = [
            token_node for token_node in self.build_proto_waitlist
            if token_node in self.token_nodes
        ]
        self.anomaly_waitlist = [
            token_node for token_node in self.anomaly_waitlist
            if token_node in self.token_nodes
        ]

        self.reset_chunk_node_id()
        self.reset_doc_node_id()

        self.chunk_avg_len = None
        if self.chunk_nodes:
            self.chunk_avg_len = sum(chunk_node.num_tokens for chunk_node in self.chunk_nodes) / len(self.chunk_nodes)

        for token_node in self.token_nodes:
            self.assign_idf(token_node)

        if self.chunk_avg_len is not None:
            self.get_proto_BM25()
        else:
            for proto_node in self.proto_nodes:
                proto_node.BM25 = {}

        self.build_chunk2proto_edge()
        self.phrase_index = defaultdict(set)
        self.build_phrase_query()

        if self.proto_nodes:
            self.build_query_database()
        else:
            self.query_database = None


    def get_max_sim_proto(self, token_node, embeds):
        proto_embeds = [proto.embed.to(self.device) for proto in token_node.proto_node_list]
        x = torch.stack(proto_embeds, 0).to(self.device)
        q = embeds.unsqueeze(0).expand_as(x)
        return int(F.cosine_similarity(x, q, dim=1).argmax().item())


    def index_document_parallel(self, doc_name, batch_size=8, queue_size=4):
        """
        Multithreaded version of index_document:
        - Stage 1 (CPU): build graph nodes + extract spans
        - Stage 2 (GPU thread): batch encode texts and push results to a bounded queue
        - Stage 3 (CPU main thread): consume results, build phrase/token embeddings, process embeds
        """
        self.start_time = time.perf_counter()
        document_start_time = time.perf_counter()

        # =============================
        # 准备：doc node & chunk list
        # =============================
        new_doc_node = self.create_doc_node(doc_name)
        chunk_list = split_doc(doc_name, self.nlp, max_tokens=self.chunk_size)
        total_chunks = len(chunk_list)

        if total_chunks == 0:
            # 仍然保持原来的收尾逻辑
            self.solve_proto()
            self.solve_anomaly()
            self.log_time(
                f"File {doc_name} completed. Index time: {time.perf_counter() - document_start_time:.4f}s"
            )
            return

        # =============================
        # 阶段 1：CPU 预处理（构图 + spans）
        # =============================
        # 这里把“每个 chunk 对应的 node/phrases/tokens”提前算好，避免 GPU 返回后再做耗时预处理
        chunk_meta = []  # [(chunk_node, phrases, tokens, chunk_text)]
        for chunk_text in chunk_list:
            new_chunk_node = self.create_chunk_node(chunk_text, new_doc_node)

            phrases, tokens = extract_important_spans(
                chunk_text,
                self.nlp,
                min_tokens=2,
                remove_duplicate=self.remove_duplicate_token
            )

            chunk_meta.append((new_chunk_node, phrases, tokens, chunk_text))

        # =============================
        # 阶段 2：构建 batch 列表
        # =============================
        batches = []
        for i in range(0, total_chunks, batch_size):
            batch_indices = list(range(i, min(i + batch_size, total_chunks)))
            batch_texts = [chunk_meta[idx][3] for idx in batch_indices]  # chunk_text
            batches.append((batch_texts, batch_indices))

        # =============================
        # 阶段 3：创建队列
        # =============================
        result_queue = queue.Queue(maxsize=queue_size)

        gpu_done = 0
        cpu_done = 0

        # =============================
        # GPU 生产者线程
        # =============================
        def gpu_worker():
            nonlocal gpu_done
            try:
                for batch_texts, batch_indices in batches:
                    token_embeddings_batch, offsets_batch = encode_chunk_batch(batch_texts, self.text_encoder, self.tokenizer, self.device)

                    result_queue.put((token_embeddings_batch, offsets_batch, batch_indices))
                    gpu_done += len(batch_indices)

                # 结束信号
                result_queue.put(None)

            except Exception as e:
                # 把异常传回主线程，避免主线程永远阻塞
                result_queue.put(("__EXCEPTION__", e))
                result_queue.put(None)

        gpu_thread = threading.Thread(target=gpu_worker, daemon=True)
        gpu_thread.start()

        # =============================
        # CPU 消费者（主线程）
        # =============================
        while True:
            item = result_queue.get()

            if item is None:
                break

            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__EXCEPTION__":
                # GPU 线程出错，直接抛出
                raise item[1]

            token_embeddings_batch, offsets_batch, batch_indices = item

            for i_in_batch, original_idx in enumerate(batch_indices):
                node, phrases, tokens, _ = chunk_meta[original_idx]

                phrase_embs, token_embs = get_token_embeds(
                    token_embeddings_batch[i_in_batch],
                    offsets_batch[i_in_batch],
                    phrases,
                    tokens
                )

                self.process_embeds(node, phrase_embs, token_embs)
                cpu_done += 1

                print(
                    f"\rGPU encoded: {gpu_done}/{total_chunks} | "
                    f"CPU processed: {cpu_done}/{total_chunks}",
                    end="",
                    flush=True
                )

        gpu_thread.join()
        print()

        # =============================
        # 收尾：prototype/anomaly + log
        # =============================
        self.solve_proto()
        self.solve_anomaly()
        self.log_time(
            f"File {doc_name} completed. Index time: {time.perf_counter() - document_start_time:.4f}s"
        )

    def chunk_id2text(self, ids):
        return [self.chunk_nodes[id].chunk_text for id in ids]

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

    def _collect_proto_sentence_examples(self, proto_node, token_text, max_sentences_per_proto=3):
        examples = []
        seen_sentences = set()

        for chunk_node in proto_node.chunk_node_list:
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

            if len(examples) >= max_sentences_per_proto:
                break

        return examples

    def inspect_multi_proto_token_nodes(self, min_proto_count=2, max_sentences_per_proto=3):
        result = {}

        for token_node in self.token_nodes:
            proto_count = len(token_node.proto_node_list)
            if proto_count < min_proto_count:
                continue

            result[token_node.token_text] = {
                "token_node_id": token_node.token_node_id,
                "proto_count": proto_count,
                "prototypes": [],
            }

            for proto_node in token_node.proto_node_list:
                result[token_node.token_text]["prototypes"].append({
                    "proto_node_id": proto_node.proto_node_id,
                    "description": proto_node.description,
                    "chunk_count": len(proto_node.chunk_node_list),
                    "sentence_examples": self._collect_proto_sentence_examples(
                        proto_node,
                        token_node.token_text,
                        max_sentences_per_proto=max_sentences_per_proto,
                    ),
                })

        return result

    def _select_multi_proto_token_nodes(
            self,
            inspect_data,
            token_contains=None,
            sort_by="proto_count",
            max_token_nodes=None,
            max_protos_per_token=None,
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
                key=lambda x: sum(proto["chunk_count"] for proto in x[1]["prototypes"]),
                reverse=True,
            )
        else:
            items.sort(key=lambda x: x[1]["proto_count"], reverse=True)

        if max_token_nodes is not None:
            items = items[:max_token_nodes]

        selected_data = {}
        for token_text, token_info in items:
            prototypes = token_info["prototypes"]
            if max_protos_per_token is not None:
                prototypes = sorted(
                    prototypes,
                    key=lambda x: x["chunk_count"],
                    reverse=True,
                )[:max_protos_per_token]

            selected_data[token_text] = {
                "token_node_id": token_info["token_node_id"],
                "proto_count": token_info["proto_count"],
                "displayed_proto_count": len(prototypes),
                "total_chunk_count": sum(proto["chunk_count"] for proto in token_info["prototypes"]),
                "prototypes": prototypes,
            }

        return selected_data

    def _limit_proto_sentence_examples(self, inspect_data, max_examples_per_token=None):
        if max_examples_per_token is None:
            return inspect_data

        limited_data = {}
        for token_text, token_info in inspect_data.items():
            remaining = max_examples_per_token
            limited_prototypes = []

            for proto_info in token_info["prototypes"]:
                if remaining <= 0:
                    limited_examples = []
                else:
                    limited_examples = proto_info["sentence_examples"][:remaining]
                remaining -= len(limited_examples)

                limited_proto = dict(proto_info)
                limited_proto["sentence_examples"] = limited_examples
                limited_prototypes.append(limited_proto)

            limited_token_info = dict(token_info)
            limited_token_info["prototypes"] = limited_prototypes
            limited_token_info["displayed_example_count"] = (
                max_examples_per_token - max(remaining, 0)
            )
            limited_data[token_text] = limited_token_info

        return limited_data

    def _format_multi_proto_token_nodes_text(self, inspect_data):
        lines = []

        for token_text, token_info in inspect_data.items():
            lines.append(
                f"[Token] {token_text} | token_node_id={token_info['token_node_id']} | "
                f"proto_count={token_info['proto_count']} | "
                f"displayed_proto_count={token_info['displayed_proto_count']} | "
                f"total_chunk_count={token_info['total_chunk_count']} | "
                f"displayed_example_count={token_info.get('displayed_example_count', 'all')}"
            )

            for proto_info in token_info["prototypes"]:
                description = proto_info.get("description") or "(none)"
                lines.append(
                    f"  [Proto] proto_node_id={proto_info['proto_node_id']} | "
                    f"description={description!r} | "
                    f"chunk_count={proto_info['chunk_count']}"
                )

                for idx, example in enumerate(proto_info["sentence_examples"], start=1):
                    lines.append(
                        f"    ({idx}) doc={example['doc_name']} | chunk_id={example['chunk_node_id']}"
                    )
                    lines.append(f"        {example['sentence_text']}")

            lines.append("")

        return "\n".join(lines).rstrip()

    def _build_multi_proto_token_nodes_html(self, inspect_data, open_details=False):
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
            token_meta = (
                f"<span style='color:#666;'>"
                f"token_node_id={token_info['token_node_id']} | "
                f"proto_count={token_info['proto_count']}, "
                f"displayed_proto_count={token_info['displayed_proto_count']}, "
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

            for proto_info in token_info["prototypes"]:
                description = proto_info.get("description") or "(none)"
                proto_header = (
                    f"proto_node_id={proto_info['proto_node_id']} | "
                    f"description={description} | "
                    f"chunk_count={proto_info['chunk_count']}"
                )
                html_parts.append(
                    "<div style='margin:10px 0 6px 16px; padding:8px 10px; "
                    "border-left:4px solid #4c78a8; background:#fff;'>"
                    f"<div style='font-weight:600; margin-bottom:6px;'>{escape(proto_header)}</div>"
                )

                for example in proto_info["sentence_examples"]:
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

    def show_multi_proto_token_nodes(
            self,
            min_proto_count=2,
            max_sentences_per_proto=3,
            as_html=True,
            token_contains=None,
            sort_by="proto_count",
            max_token_nodes=20,
            max_protos_per_token=5,
            max_examples_per_token=10,
            open_details=False,
    ):
        inspect_data = self.inspect_multi_proto_token_nodes(
            min_proto_count=min_proto_count,
            max_sentences_per_proto=max_sentences_per_proto,
        )
        inspect_data = self._select_multi_proto_token_nodes(
            inspect_data,
            token_contains=token_contains,
            sort_by=sort_by,
            max_token_nodes=max_token_nodes,
            max_protos_per_token=max_protos_per_token,
        )
        inspect_data = self._limit_proto_sentence_examples(
            inspect_data,
            max_examples_per_token=max_examples_per_token,
        )

        if not inspect_data:
            empty_text = "No token nodes matched the multi-prototype condition."
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
                return HTML(self._build_multi_proto_token_nodes_html(inspect_data, open_details=open_details))
            except ImportError:
                pass

        return self._format_multi_proto_token_nodes_text(inspect_data)


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


    def delete_chunk_node(self, chunk_node_to_delete):
        for node in chunk_node_to_delete.proto_node_list:
            index_to_delete = []
            for index, chunk_node in enumerate(node.chunk_node_list):
                if chunk_node is chunk_node_to_delete:
                    index_to_delete.append(index)
            index_to_delete = set(index_to_delete)
            node.chunk_node_list = [x for i, x in enumerate(node.chunk_node_list) if i not in index_to_delete]
            if getattr(node, "span_occurrences", None):
                node.span_occurrences = [
                    x for i, x in enumerate(node.span_occurrences)
                    if i not in index_to_delete
                ]
            node.chunk_edge_weight = [x for i, x in enumerate(node.chunk_edge_weight) if i not in index_to_delete]
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

    def reset_chunk_node_id(self):
        for index, chunk_node in enumerate(self.chunk_nodes):
            chunk_node.chunk_node_id = index
        self.next_chunk_node_id = len(self.chunk_nodes)

    def reset_doc_node_id(self):
        for index, doc_node in enumerate(self.doc_nodes):
            doc_node.doc_node_id = index
        self.next_doc_node_id = len(self.doc_nodes)

    def _load_nlp(self):
        #nlp = spacy.load("en_core_web_sm")
        nlp = spacy.load("en_core_web_lg")
        #nlp = spacy.load("en_core_web_trf")
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


    # ⭐ 控制 pickle 时保存什么
    def __getstate__(self):
        state = self.__dict__.copy()
        state["text_encoder"] = None
        state["tokenizer"] = None
        state["executor"] = None
        state["nlp"] = None
        state["reranker"] = None
        state["proto_description_model"] = None
        return state

    # ⭐ 反序列化后做什么
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.text_encoder = None
        self.tokenizer = None
        self.executor = None
        self.nlp = None
        self.reranker = None
        self.proto_description_model = None
        self._ensure_backward_compatible_attrs()

    def _restore_runtime_components(self, load_nlp=True, load_reranker=False):
        self._ensure_backward_compatible_attrs()
        if self.text_encoder is None or self.tokenizer is None:
            self._load_text_encoder()
        self.save_doc_to_json()
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=4)
        if load_nlp and self.nlp is None:
            self._load_nlp()
        if load_reranker and self.reranker is None:
            self.load_reranker()
        if self.proto_description_model is None:
            self._load_proto_description_model()

    def save_data(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def save_doc_to_json(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([x.doc_name for x in self.doc_nodes], f, ensure_ascii=False, indent=2)

    @classmethod
    def load_data(cls, path):
        with open(path, "rb") as f:
            obj= pickle.load(f)
        obj._restore_runtime_components(load_nlp=True, load_reranker=False)
        return obj

    def save_data_split(self, pkl_path: str):
        """
        生成两个文件：
          - pkl_path:                 保存结构（不含 tensor）
          - pkl_path.replace('.pkl','_tensors.pt'): 保存 query_database + proto embeds
        """
        pt_path = pkl_path.replace(".pkl", "_tensors.pt")
        self.node_instance2id()
        # 1) 收集 tensor 到 CPU（torch.save 很稳）
        tensor_pack = {
            "query_database": self.query_database.detach().cpu() if self.query_database is not None else None,
            # 用列表索引保存，最简单稳定
            "proto_embeds": [
                (p.embed.detach().cpu() if p.embed is not None else None)
                for p in self.proto_nodes
            ],
            "proto_retained_text_embeddings": [
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
                    }
                    for text_embedding in getattr(p, "retained_text_embeddings", [])
                ]
                for p in self.proto_nodes
            ],
        }
        torch.save(tensor_pack, pt_path)

        # 2) 临时把对象里的 tensor 清空，避免 pickle 遇到 torch.Storage
        qbak = self.query_database
        pbak = [p.embed for p in self.proto_nodes]
        retained_bak = [p.retained_text_embeddings for p in self.proto_nodes]

        self.query_database = None
        for p in self.proto_nodes:
            p.embed = None
            p.retained_text_embeddings = [
                TextEmbedding(
                    embed=None,
                    chunk_node=text_embedding.chunk_node,
                    span_start=text_embedding.span_start,
                    span_end=text_embedding.span_end,
                    span_text=text_embedding.span_text,
                )
                for text_embedding in getattr(p, "retained_text_embeddings", [])
            ]

        # 3) 临时去掉运行态组件（你之前 __getstate__ 里就有这些）
        exec_bak = getattr(self, "executor", None)
        enc_bak = getattr(self, "text_encoder", None)
        tok_bak = getattr(self, "tokenizer", None)
        nlp_bak = getattr(self, "nlp", None)
        reranker_bak = getattr(self, "reranker", None)

        if hasattr(self, "executor"): self.executor = None
        if hasattr(self, "text_encoder"): self.text_encoder = None
        if hasattr(self, "tokenizer"): self.tokenizer = None
        if hasattr(self, "nlp"): self.nlp = None
        if hasattr(self, "reranker"): self.reranker = None

        # 4) 把 pt 路径写进对象，便于 load
        tensors_path_bak = getattr(self, "tensors_path", None)
        self.tensors_path = pt_path
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            # 5) 恢复内存中的对象，保证系统继续可用
            self.query_database = qbak
            for p, e, retained in zip(self.proto_nodes, pbak, retained_bak):
                p.embed = e
                p.retained_text_embeddings = retained

            if hasattr(self, "executor"): self.executor = exec_bak
            if hasattr(self, "text_encoder"): self.text_encoder = enc_bak
            if hasattr(self, "tokenizer"): self.tokenizer = tok_bak
            if hasattr(self, "nlp"): self.nlp = nlp_bak
            if hasattr(self, "reranker"): self.reranker = reranker_bak
            self.node_id2instance()
            # tensors_path 也恢复（可选）
            self.tensors_path = tensors_path_bak


    @classmethod
    def load_data_split(cls, pkl_path: str):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        pt_path = getattr(obj, "tensors_path", pkl_path.replace(".pkl", "_tensors.pt"))
        tensor_pack = torch.load(pt_path)

        query_database = tensor_pack.get("query_database", None)
        obj.query_database = query_database.to(obj.device) if query_database is not None else None

        proto_embeds = tensor_pack.get("proto_embeds", [])
        proto_retained_text_embeddings = tensor_pack.get("proto_retained_text_embeddings", [])
        # 按索引回填
        for i, p in enumerate(obj.proto_nodes):
            embed = proto_embeds[i] if i < len(proto_embeds) else None
            p.embed = embed.to("cpu") if embed is not None else None
            retained_pack = proto_retained_text_embeddings[i] if i < len(proto_retained_text_embeddings) else []
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
                )
                for item in retained_pack
            ]
            p.retained_text_embedding_source_count = max(
                getattr(p, "retained_text_embedding_source_count", 0),
                len(p.retained_text_embeddings),
            )

        # 重建运行态（按你原逻辑）
        obj._restore_runtime_components(load_nlp=True, load_reranker=False)
        obj.node_id2instance()
        if obj.query_database is None and obj.proto_nodes and all(p.embed is not None for p in obj.proto_nodes):
            obj.build_query_database()
        obj.load_reranker()
        return obj

    def node_instance2id(self):
        for chunk_node in self.chunk_nodes:
            chunk_node.doc_node = chunk_node.doc_node.doc_node_id
            chunk_node.proto_node_list = [node.proto_node_id for node in chunk_node.proto_node_list]
        for doc_node in self.doc_nodes:
            doc_node.chunk_node_list = [chunk_node.chunk_node_id for chunk_node in doc_node.chunk_node_list]
        for proto_node in self.proto_nodes:
            proto_node.chunk_node_list = [chunk_node.chunk_node_id for chunk_node in proto_node.chunk_node_list]
            for text_embedding in getattr(proto_node, "retained_text_embeddings", []):
                text_embedding.chunk_node = text_embedding.chunk_node.chunk_node_id
            proto_node.token_node = proto_node.token_node.token_node_id
        for token_node in self.token_nodes:
            token_node.proto_node_list = [proto_node.proto_node_id for proto_node in token_node.proto_node_list]

    def node_id2instance(self):
        for chunk_node in self.chunk_nodes:
            chunk_node.doc_node = self.doc_nodes[chunk_node.doc_node]
            chunk_node.proto_node_list = [self.proto_nodes[idx] for idx in chunk_node.proto_node_list]
        for doc_node in self.doc_nodes:
            doc_node.chunk_node_list = [self.chunk_nodes[idx] for idx in doc_node.chunk_node_list]
        for proto_node in self.proto_nodes:
            proto_node.chunk_node_list = [self.chunk_nodes[idx] for idx in proto_node.chunk_node_list]
            proto_node.token_node = self.token_nodes[proto_node.token_node]
            if not hasattr(proto_node, "description"):
                proto_node.description = self._get_initial_proto_description(proto_node.token_node)
            if not hasattr(proto_node, "span_occurrences") or proto_node.span_occurrences is None:
                proto_node.span_occurrences = []
            if not proto_node.span_occurrences and proto_node.chunk_node_list:
                proto_node.span_occurrences = [
                    SpanOccurrence(chunk_node=chunk_node)
                    for chunk_node in proto_node.chunk_node_list
                ]
            if not hasattr(proto_node, "retained_text_embeddings") or proto_node.retained_text_embeddings is None:
                proto_node.retained_text_embeddings = []
            for text_embedding in proto_node.retained_text_embeddings:
                text_embedding.chunk_node = self.chunk_nodes[text_embedding.chunk_node]
            if not hasattr(proto_node, "retained_text_embedding_source_count"):
                proto_node.retained_text_embedding_source_count = len(proto_node.retained_text_embeddings)
            if not hasattr(proto_node, "pending_embed_rebuild"):
                proto_node.pending_embed_rebuild = False
        for token_node in self.token_nodes:
            token_node.proto_node_list = [self.proto_nodes[idx] for idx in token_node.proto_node_list]

    def index_json(self, chunk_list, batch_size=8, queue_size=4):
        self.start_time = time.perf_counter()
        self._reset_hdbscan_stats()
        total_chunks = len(chunk_list)
        doc_node_map = {}

        # =============================
        # 队列
        # =============================
        preprocess_queue = queue.Queue(maxsize=queue_size)
        result_queue = queue.Queue(maxsize=queue_size)

        # =============================
        # 进度计数
        # =============================
        progress_lock = threading.Lock()
        preprocess_done = 0
        gpu_done = 0
        cpu_done = 0

        # =============================
        # 异常传递
        # =============================
        worker_errors = []
        stop_event = threading.Event()

        def record_error(stage_name):
            if worker_errors:
                return
            worker_errors.append(
                (
                    stage_name,
                    traceback.format_exc()
                )
            )
            stop_event.set()

        def get_progress_snapshot():
            with progress_lock:
                return preprocess_done, gpu_done, cpu_done

        def inc_preprocess_done(n=1):
            nonlocal preprocess_done
            with progress_lock:
                preprocess_done += n

        def inc_gpu_done(n=1):
            nonlocal gpu_done
            with progress_lock:
                gpu_done += n

        def inc_cpu_done(n=1):
            nonlocal cpu_done
            with progress_lock:
                cpu_done += n

        last_progress_update_time = [0.0]

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

        def safe_queue_put(target_queue, item):
            while not stop_event.is_set():
                try:
                    target_queue.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        # =============================
        # 阶段 1：CPU 预处理线程
        # 输出到 preprocess_queue:
        # (idx, new_chunk_node, phrases, tokens, text)
        # =============================
        def cpu_preprocess_worker():
            try:
                for idx, chunk in enumerate(chunk_list):
                    if stop_event.is_set():
                        break

                    title = chunk["title"]
                    new_doc_node = doc_node_map.get(title)
                    if new_doc_node is None:
                        new_doc_node = self.create_doc_node(title)
                        doc_node_map[title] = new_doc_node
                    new_chunk_node = self.create_chunk_node(
                        chunk["text"], new_doc_node
                    )

                    phrases, tokens = extract_important_spans(
                        chunk["text"],
                        self.nlp,
                        min_tokens=2,
                        remove_duplicate=self.remove_duplicate_token,
                        discard_no_word=self.discard_no_word
                    )

                    if not safe_queue_put(
                        preprocess_queue,
                        (
                            idx,
                            new_chunk_node,
                            phrases,
                            tokens,
                            chunk["text"]
                        )
                    ):
                        break

                    inc_preprocess_done(1)
                    print_progress()

            except Exception:
                record_error("cpu_preprocess_worker")

            finally:
                safe_queue_put(preprocess_queue, None)

        # =============================
        # 阶段 2：GPU 线程
        # 从 preprocess_queue 取数据，凑 batch，
        # encode 后输出到 result_queue:
        # (
        #   batch_items,
        #   token_embeddings_batch,
        #   offsets_batch
        # )
        # 其中 batch_items 是:
        # [
        #   (idx, node, phrases, tokens),
        #   ...
        # ]
        # =============================
        def gpu_worker():
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

                # 刷掉最后不足一个 batch 的部分
                if batch_buffer:
                    flush_batch(batch_buffer)

            except Exception:
                record_error("gpu_worker")

            finally:
                safe_queue_put(result_queue, None)

        # =============================
        # 启动线程
        # =============================
        preprocess_thread = threading.Thread(target=cpu_preprocess_worker)
        gpu_thread = threading.Thread(target=gpu_worker)

        print_progress(force=True)
        preprocess_thread.start()
        gpu_thread.start()

        # =============================
        # 阶段 3：主线程消费 GPU 结果
        # =============================
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

        self.log_time(
            f"Index {total_chunks} documents. "
            f"Index time: {time.perf_counter() - self.start_time:.4f}s"
        )

        self.solve_proto()
        self.solve_anomaly()
        print(
            "HDBSCAN attempts: "
            f"{self.hdbscan_attempt_count}, "
            f"successes (n_clusters >= 1): {self.hdbscan_success_count}"
        )

class ListBatchExtractor:
    def __init__(self, list_of_lists, k, mode="round", exclude_list=None):
        """
        list_of_lists : 二维列表（不会改变）
        k             : round 模式下每轮每个子列表取K个
        mode          : "round" 或 "sequential"
        exclude_list  : 需要跳过的数字
        """
        if mode not in ("round", "sequential"):
            raise ValueError("mode must be 'round' or 'sequential'")

        self.list_of_lists = list_of_lists
        self.k = k
        self.mode = mode
        self.exclude_set = set(exclude_list) if exclude_list else set()

        # round 模式状态
        self.positions = [0] * len(list_of_lists)

        # sequential 模式状态
        self.seq_outer_idx = 0
        self.seq_inner_idx = 0

        self.finished = False

    def extract(self, N, result=None):
        if self.finished:
            return result if result else []

        if result is None:
            result = []

        if self.mode == "round":
            self._extract_round(N, result)
        else:
            self._extract_sequential(N, result)

        return result

    # =============================
    # round 模式
    # =============================
    def _extract_round(self, N, result):
        while len(result) < N:
            added = 0

            for idx, sublist in enumerate(self.list_of_lists):
                count = 0
                while (
                    self.positions[idx] < len(sublist)
                    and count < self.k
                    and len(result) < N
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

    # =============================
    # sequential 模式
    # =============================
    def _extract_sequential(self, N, result):
        while len(result) < N and self.seq_outer_idx < len(self.list_of_lists):

            sublist = self.list_of_lists[self.seq_outer_idx]

            while (
                self.seq_inner_idx < len(sublist)
                and len(result) < N
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

    # =============================
    # 状态管理
    # =============================
    def get_state(self):
        return {
            "positions": self.positions.copy(),
            "seq_outer_idx": self.seq_outer_idx,
            "seq_inner_idx": self.seq_inner_idx,
            "finished": self.finished,
            "mode": self.mode
        }

    def load_state(self, state):
        if state["mode"] != self.mode:
            raise ValueError("State mode does not match extractor mode")

        self.positions = state["positions"].copy()
        self.seq_outer_idx = state["seq_outer_idx"]
        self.seq_inner_idx = state["seq_inner_idx"]
        self.finished = state["finished"]

    def reset(self):
        self.positions = [0] * len(self.list_of_lists)
        self.seq_outer_idx = 0
        self.seq_inner_idx = 0
        self.finished = False
