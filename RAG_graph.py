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
    inspect_sem_nodes,
    load_wikidata_definition_candidates,
    plot_embeddings,
    print_size_mb,
    sem_embed_sim,
)

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

    # Build weighted co-occurrence links between semantic nodes that share chunks.
    def build_edges(self):
        for node_a, node_b in combinations(self.node_list, 2):
            weight = get_COG_edge_weight(node_a.sem_node , node_b.sem_node)
            if weight > 0:
                node_a.neighbor_node_list.append((node_b, weight))
                node_b.neighbor_node_list.append((node_a, weight))

        for node in self.node_list:
            if len(node.neighbor_node_list) > 0:
                self.connected_node_list.append(node)
                for _, weight in node.neighbor_node_list:
                    node.node_weight += weight
                                                                                                      
                node.node_weight = node.node_weight * node.node_level_weight
            else:
                self.isolate_node_list.append(node)

    # Aggregate semantic node weights into chunk-level retrieval scores.
    def assign_chunk_weight(self, avg_chunk_len, debug_mode=False):
                                                   
        if len(self.connected_node_list) > 0 :
            weight_map = {}
            token_record_map = defaultdict(list)
            token_weight_map = {}
            chunk_len_map = {}
            for node in self.connected_node_list:
                tf_chunk_dict = dict(Counter([chunk_node.chunk_node_id for chunk_node in node.sem_node.chunk_node_list]))
                for chunk_node in node.sem_node.chunk_node_list:
                    chunk_node_id = chunk_node.chunk_node_id
                    if node.sem_node.token_node.token_text not in token_record_map[chunk_node_id]:
                                                                                                                          
                        bm25_score = node.node_weight * node.sem_node.BM25[chunk_node_id]
                        weight_map[chunk_node_id] = weight_map.get(chunk_node_id, 0) + bm25_score
                        token_weight_map.setdefault(chunk_node_id, []).append(f"Token:{node.sem_node.token_node.token_text},Score:{(bm25_score ):.4f}")
                        token_record_map[chunk_node_id].append(node.sem_node.token_node.token_text)
                        chunk_len_map[chunk_node_id] = chunk_node.num_tokens
                                                                     
                                                                                    
            self.weighted_chunk_node_list = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)
        if debug_mode:
            print(token_weight_map)
            print(weight_map)
            print(self.weighted_chunk_node_list)
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
        self.ranked_sem_node_list = [[] for i in range(4)]
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

CoOccurrenceNode_query_weight = [1.2,1,0.8,0.5]

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
        self.neighbor_node_list = []
        self.node_weight = 0
        self.node_level_weight = CoOccurrenceNode_query_weight[self.node_level]


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


@dataclass
class TokenNode:
    token_text: str
    token_node_id: int
    node_type: str
    is_multi_semantic: bool = field(init=False)
    descriptions: list = field(init=False)
    wikidata_info_loaded: bool = field(init=False)
    has_semantic: bool = False
    sem_node_list: list = field(default_factory=list)
    embeds_buffer: list = field(default_factory=list)
    span_occurrences: list = field(default_factory=list)
    idf: float = 0
    df: int = 0
    anomaly_section: list = field(default_factory=list)

    # Initialize optional token metadata after dataclass construction.
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

    # Convert a retained text embedding into a span occurrence record.
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

    # Return the span boundaries when both endpoints are available.
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
    anomaly_threshold: float | None = None
    tf_dict_by_chunk_id: dict | None = None
    idf: float = 0
    df: int = 0
    chunk_len_dict_by_id: dict | None = None
    BM25: dict | None = None

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


globals()["".join(("Pro", "totype"))] = SemNode


class ProtoGraphRAG:
    # Initialize graph storage, model handles, retrieval settings, and runtime state.
    def __init__(self, text_embed_dim, df_ratio, buffer_size=100, anomaly_threshold_percentile=0.9,
                 anomaly_section_size=50,query_token_percentile=0.8,
                 retrieve_top_k=5, chunk_size=300, remove_duplicate_token=True, device="cuda",
                 discard_no_word=False, plot_embeds=False,
                 sem_description_prompt_context_mode="sentence_neighbors"):
        self.text_embed_dim = text_embed_dim
        self.df_ratio = df_ratio
        self.doc_nodes = []
        self.chunk_nodes = []
        self.token_nodes = []
        self.phrase_token_nodes = []
        self.sem_nodes = []
        self.next_doc_node_id = 0
        self.next_chunk_node_id = 0
        self.next_token_node_id = 0
        self.next_sem_node_id = 0
        self.buffer_size = buffer_size
        self.token_node_query = {}
        self.tau_conc = 0.90
        self.tau_disp = 0.78
        self.build_sem_node_waitlist = []
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
        self.sem_description_model_name = "cross-encoder/nli-deberta-v3-large"
        self.sem_description_model = None
        self._load_sem_description_model()
        self.sem_description_candidate_limit = 5
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
        self.plot_embeds = plot_embeds
        self.sem_description_prompt_context_mode = sem_description_prompt_context_mode
        self.predicted_sem_description_logs = []
        self.deleted_merged_sem_logs = []
        self.sem_description_operation_logs = []
        self.wikidata_no_result_logs = []
        self._wikidata_no_result_keys = set()
        self.hdbscan_attempt_count = 0
        self.hdbscan_success_count = 0

    # Initialize or disable the reranker used after graph retrieval.
    def load_reranker(self):
                                            
        self.reranker = None

    # Load the cross-encoder used to predict semantic node descriptions.
    def _load_sem_description_model(self):
        from sentence_transformers import CrossEncoder

        self.sem_description_model = CrossEncoder(self.sem_description_model_name)

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

    # Append a structured semantic-description operation log entry.
    def _log_sem_description_operation(self, event_type, **payload):
        self.sem_description_operation_logs.append(
            {
                "event_type": event_type,
                **payload,
            }
        )

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

    # Reset HDBSCAN attempt and success counters.
    def _reset_hdbscan_stats(self):
        self.hdbscan_attempt_count = 0
        self.hdbscan_success_count = 0

    # Track whether an HDBSCAN clustering attempt produced clusters.
    def _record_hdbscan_attempt(self, n_clusters):
        self.hdbscan_attempt_count += 1
        if n_clusters >= 1:
            self.hdbscan_success_count += 1

    # Create and register a document node.
    def create_doc_node(self, doc_name):
        new_doc_node = DocumentNode(doc_name, self._new_node_id("doc"))
        self.doc_nodes.append(new_doc_node)
        return new_doc_node

    # Create and register a chunk node under a document node.
    def create_chunk_node(self, chunk_text, doc_node):
        new_chunk_node = ChunkNode(chunk_text, self._new_node_id("chunk"), doc_node)
        new_chunk_node.num_tokens = get_num_tokens(chunk_text, self.nlp)
        self.chunk_nodes.append(new_chunk_node)
        doc_node.chunk_node_list.append(new_chunk_node)
        return new_chunk_node

    # Create and register a token or phrase token node.
    def create_token_node(self, token_text):
        node_type = 'phrase' if len(token_text.split()) >= 2 else 'token'
        new_token_node = TokenNode(token_text, self._new_node_id("token"), node_type)
        self.token_nodes.append(new_token_node)
        if new_token_node.node_type == "phrase":
            self.phrase_token_nodes.append(new_token_node)
        self.token_node_query[token_text] = new_token_node
        return new_token_node

    # Create and register a semantic node for a token node.
    def create_sem_node(self, token_node):
        new_sem_node = SemNode(self._new_node_id("sem"), token_node)
        new_sem_node.description = self._get_initial_sem_description(token_node)
        self.sem_nodes.append(new_sem_node)
        token_node.has_semantic = True
        return new_sem_node

    # Wrap an embedding with chunk and optional span metadata.
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
        sem_node.chunk_node_list.append(text_embedding.chunk_node)
        sem_node.span_occurrences.append(text_embedding.to_span_occurrence())
        self._retain_text_embedding_for_sem(sem_node, text_embedding)
        if edge_weight is not None:
            sem_node.chunk_edge_weight.append(edge_weight)

    # Attach a span occurrence to a semantic node and retain its embedding when available.
    def _append_span_occurrence_to_sem(
        self,
        sem_node,
        span_occurrence,
        edge_weight=None,
        retained_text_embedding=None,
    ):
        sem_node.chunk_node_list.append(span_occurrence.chunk_node)
        sem_node.span_occurrences.append(
            SpanOccurrence(
                chunk_node=span_occurrence.chunk_node,
                span_start=span_occurrence.span_start,
                span_end=span_occurrence.span_end,
                span_text=span_occurrence.span_text,
            )
        )
        if retained_text_embedding is not None:
            self._retain_text_embedding_for_sem(sem_node, retained_text_embedding)
            if getattr(sem_node, "pending_embed_rebuild", False):
                sem_node.chunk_node_embed.append(retained_text_embedding.embed)
        if edge_weight is not None:
            sem_node.chunk_edge_weight.append(edge_weight)

    # Return the initial description assigned to a new semantic node.
    def _get_initial_sem_description(self, token_node):
        return None

    # Create one semantic node from all buffered embeddings for a token node.
    def create_basic_sem_node(self, token_node):
        new_sem_node = self.create_sem_node(token_node)
        new_sem_node.chunk_node_embed = ([i.embed for i in token_node.embeds_buffer])
        new_sem_node.embed = average_embeds(new_sem_node.chunk_node_embed)
        new_sem_node.chunk_node_list = [k.chunk_node for k in token_node.embeds_buffer]
        new_sem_node.span_occurrences = [k.to_span_occurrence() for k in token_node.embeds_buffer]
        self._initialize_sem_retained_text_embeddings(new_sem_node, token_node.embeds_buffer)
        new_sem_node.chunk_edge_weight = sem_embed_sim(new_sem_node).cpu().tolist()
        new_sem_node.anomaly_threshold = get_anomaly_threshold(new_sem_node.chunk_edge_weight,
                                                                 self.anomaly_threshold_percentile)
        new_sem_node.chunk_node_embed.clear()

        token_node.sem_node_list.append(new_sem_node)

    # Cluster buffered token embeddings and build semantic nodes or anomaly records.
    def build_sem_node(self, token_node):
        if token_node.node_type == "token" and not self.semantic_type_cls(token_node):
            self.create_basic_sem_node(token_node)
        else:
            n_clusters, clusters, cluster_centers = hdbscan_cluster([(k.embed.cpu(),k.chunk_node) for k in token_node.embeds_buffer],
                                                                    min_cluster_size=int(len(token_node.embeds_buffer)/20),
                                                                    percentile=self.anomaly_threshold_percentile, merge_chunks=False)
            self._record_hdbscan_attempt(n_clusters)
            if n_clusters >= 1:
                if self.plot_embeds:
                    self.plot_embed_distribution(token_node, clusters)
                for clusters_label in range(n_clusters):
                    new_sem_node = self.create_sem_node(token_node)
                    new_sem_node.embed = torch.from_numpy(cluster_centers[clusters_label])
                    cluster_text_embeddings = []
                    for idx in clusters[clusters_label]:
                        text_embedding = token_node.embeds_buffer[idx]
                        new_sem_node.chunk_node_list.append(text_embedding.chunk_node)
                        new_sem_node.span_occurrences.append(text_embedding.to_span_occurrence())
                        new_sem_node.chunk_node_embed.append(text_embedding.embed)
                        cluster_text_embeddings.append(text_embedding)
                    self._initialize_sem_retained_text_embeddings(new_sem_node, cluster_text_embeddings)
                    new_sem_node.chunk_edge_weight = sem_embed_sim(new_sem_node).cpu().tolist()
                    new_sem_node.anomaly_threshold = get_anomaly_threshold(new_sem_node.chunk_edge_weight,
                                                                             self.anomaly_threshold_percentile)
                    new_sem_node.chunk_node_embed.clear()
                    token_node.sem_node_list.append(new_sem_node)
                anomaly_idx = clusters.get(-1)
                if anomaly_idx is not None:
                    for idx in anomaly_idx:
                        text_embedding = token_node.embeds_buffer[idx]
                        max_val, max_idx = inspect_sem_nodes(text_embedding.embed, token_node.sem_node_list)
                        token_node.anomaly_section.append(
                            AnomalyTextEmbedding(
                                text_embedding=text_embedding,
                                max_val=max_val,
                                max_idx=max_idx,
                            )
                        )
            else:
                self.create_basic_sem_node(token_node)
        token_node.embeds_buffer.clear()

    # Build semantic nodes for all token nodes waiting in the build queue.
    def solve_sem_nodes(self):
        for token_node in self.build_sem_node_waitlist:
            self.build_sem_node(token_node)
        self.build_sem_node_waitlist = []

    # Cluster or reassign anomaly embeddings collected during semantic node creation.
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
                    new_sem_node = self.create_sem_node(token_node)
                    new_sem_node.embed = torch.from_numpy(cluster_centers[clusters_label]).to(self.device)
                    cluster_text_embeddings = []
                    for idx in clusters[clusters_label]:
                        text_embedding = token_node.anomaly_section[idx].text_embedding
                        new_sem_node.chunk_node_list.append(text_embedding.chunk_node)
                        new_sem_node.span_occurrences.append(text_embedding.to_span_occurrence())
                        new_sem_node.chunk_node_embed.append(text_embedding.embed)
                        cluster_text_embeddings.append(text_embedding)
                    self._initialize_sem_retained_text_embeddings(new_sem_node, cluster_text_embeddings)
                    new_sem_node.chunk_edge_weight = sem_embed_sim(new_sem_node).cpu().tolist()
                    new_sem_node.anomaly_threshold = get_anomaly_threshold(new_sem_node.chunk_edge_weight,
                                                                             self.anomaly_threshold_percentile)
                    new_sem_node.chunk_node_embed.clear()
                anomaly_idx = clusters.get(-1)
                if anomaly_idx is None:
                    token_node.anomaly_section = []
                else:
                    token_node.anomaly_section = [token_node.anomaly_section[i] for i in anomaly_idx]
            else:
                for item in token_node.anomaly_section:
                    self._append_sem_occurrence(
                        token_node.sem_node_list[item.max_idx],
                        item.text_embedding,
                        edge_weight=item.max_val,
                    )
        self.anomaly_waitlist = []

                                             
                                       
                                           
                                                                          

    # Print an elapsed-time progress message for the current operation.
    def log_time(self, msg):
        print(f"[{time.perf_counter() - self.start_time:.4f}s] {msg}")

    # Create a throttled console progress updater for long loops.
    def _make_progress_updater(self, label, total, min_interval=0.2):
        last_update_time = [0.0]

        # Print progress when enough time has elapsed or a forced update is requested.
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
        if token_node.df > len(self.chunk_nodes) * self.df_ratio:
            return False
        s_mean = get_s_mean([i.embed for i in token_node.embeds_buffer])
        if s_mean > self.tau_conc:
            return False
        else:
            return True

    # Look up a token node by its surface text.
    def query_token_node(self, text):
        return self.token_node_query.get(text, None)

    # Populate missing attributes when loading graphs saved by older code versions.
    def _ensure_backward_compatible_attrs(self):
        if not hasattr(self, "sem_nodes") and hasattr(self, "proto_nodes"):
            self.sem_nodes = self.proto_nodes
        if not hasattr(self, "next_sem_node_id") and hasattr(self, "next_proto_node_id"):
            self.next_sem_node_id = self.next_proto_node_id
        if not hasattr(self, "build_sem_node_waitlist") and hasattr(self, "build_proto_waitlist"):
            self.build_sem_node_waitlist = self.build_proto_waitlist
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
        if not hasattr(self, "remove_duplicate_token"):
            self.remove_duplicate_token = True
        if not hasattr(self, "discard_no_word"):
            self.discard_no_word = False
        if not hasattr(self, "plot_embeds"):
            self.plot_embeds = False
        if not hasattr(self, "sem_description_prompt_context_mode"):
            self.sem_description_prompt_context_mode = "sentence_neighbors"
        if not hasattr(self, "sem_description_candidate_limit"):
            self.sem_description_candidate_limit = 5
        if not hasattr(self, "sem_description_batch_size"):
            self.sem_description_batch_size = 32
        if not hasattr(self, "sem_description_use_detailed_description"):
            self.sem_description_use_detailed_description = False
        if not hasattr(self, "sem_description_require_detailed_description"):
            self.sem_description_require_detailed_description = True
        if not hasattr(self, "sem_description_exact_match_text"):
            self.sem_description_exact_match_text = False
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
        if not hasattr(self, "sem_description_model_name"):
            self.sem_description_model_name = "cross-encoder/nli-deberta-v3-large"
        if not hasattr(self, "sem_description_model"):
            self.sem_description_model = None
        for token_node in getattr(self, "token_nodes", []):
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
        for chunk_node in getattr(self, "chunk_nodes", []):
            if not hasattr(chunk_node, "sem_node_list") and hasattr(chunk_node, "proto_node_list"):
                chunk_node.sem_node_list = chunk_node.proto_node_list
        for sem_node in getattr(self, "sem_nodes", []):
            if not hasattr(sem_node, "sem_node_id") and hasattr(sem_node, "proto_node_id"):
                sem_node.sem_node_id = sem_node.proto_node_id
            if not hasattr(sem_node, "span_occurrences"):
                sem_node.span_occurrences = []
            if not hasattr(sem_node, "retained_text_embeddings") or sem_node.retained_text_embeddings is None:
                sem_node.retained_text_embeddings = []
            if not hasattr(sem_node, "retained_text_embedding_source_count"):
                sem_node.retained_text_embedding_source_count = len(sem_node.retained_text_embeddings)
            if not hasattr(sem_node, "pending_embed_rebuild"):
                sem_node.pending_embed_rebuild = False
            for text_embedding in sem_node.retained_text_embeddings:
                if not hasattr(text_embedding, "span_start"):
                    text_embedding.span_start = None
                if not hasattr(text_embedding, "span_end"):
                    text_embedding.span_end = None
                if not hasattr(text_embedding, "span_text"):
                    text_embedding.span_text = None

    # Run span extraction with optional cleanup and debug output.
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

    # Build the normalized semantic-node embedding matrix used for similarity search.
    def build_query_database(self):
        embeds_list = [sem_node.embed for sem_node in self.sem_nodes]
        database = torch.stack(embeds_list).to(self.device)
        self.query_database = F.normalize(database, dim=1)

    # Clean a query and extract token, phrase, and entity embeddings for matching.
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

    # Resolve exact, fuzzy, and embedding-based semantic node matches for a query.
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
            exact_sem_node = None
            fuzzy_sem_nodes = []
            retrieved_sem = None
            semantic_weight = None

            if token_node is not None:
                exact_sem_node = token_node.sem_node_list[self.get_max_sim_sem(token_node, token_embed)]
                if token_type in {"phrase", "ent"} and search_mode != "broad":
                    tokens_in_phrase.extend(token.split(" "))

            fuzzy_query_list = self.phrase_fuzzy_query(token)
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
                "token_embed": token_embed,
                "exact_sem_node": exact_sem_node,
                "fuzzy_sem_nodes": fuzzy_sem_nodes,
                "retrieved_sem": retrieved_sem,
                "semantic_weight": semantic_weight,
            })

        return query_text, query_tokens, resolved_matches

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
        retrieved_chunk_ids = list(set(retrieved_chunk_ids))
        retrieved_chunks = self.chunk_id2text(retrieved_chunk_ids)[:candidate]
        rerank_chunks, rerank_chunk_index= self.reranker.rerank(query_text, retrieved_chunks, top_k=top_k)
        rerank_chunk_ids = [retrieved_chunk_ids[i] for i in rerank_chunk_index]

        return rerank_chunks, rerank_chunk_ids

    # Retrieve chunks using exact, fuzzy, and semantic matches organized by match level.
    def multi_level_query(self, query_text, top_k_chunk=10, top_k_each_isolated_chunk=2, isolate_chunk_ratio=0.2, isolate_retrieve_mode='sequential',print_important_tokens=True, search_mode='broad'):
        query_text, query_tokens, resolved_matches = self._resolve_query_matches(
            query_text,
            search_mode=search_mode,
            print_important_tokens=print_important_tokens,
        )
        query_tokens = []
        low_level_tokens = []
        high_level_tokens = []
        low_level_sems = []
        high_level_sems = []
        tokens_in_phrase = []
        for match in resolved_matches:
            token_type = match["token_type"]
            token = match["token"]
            if token_type == "token" and token in tokens_in_phrase:
                continue
            query_tokens.append(token)
            exact_match = match["exact_sem_node"] is not None
            fuzzy_match = len(match["fuzzy_sem_nodes"]) > 0
            if exact_match:
                max_sem_node = match["exact_sem_node"]
                low_level_tokens.append(token + f"(exact matched)")
                if token_type == 'ent':
                    low_level_sems.append((max_sem_node, 0, 1))
                else:
                    low_level_sems.append((max_sem_node, 1, 1))
                if (token_type == 'phrase' or token_type == 'ent') and search_mode != 'broad':
                    tokens_in_phrase.extend(token.split(" "))
            if fuzzy_match:
                for sem_node in match["fuzzy_sem_nodes"][:1]:
                    weight = count_words(token) / count_words(sem_node.token_node.token_text)
                    if exact_match:
                        if token_type == 'ent':
                            high_level_sems.append((sem_node,1, weight))
                        else:
                            high_level_sems.append((sem_node, 2, weight))
                        high_level_tokens.append(
                            sem_node.token_node.token_text + f"(partial matched)")
                    else:
                        if token_type == 'ent':
                            low_level_sems.append((sem_node,1, weight))
                        else:
                            low_level_sems.append((sem_node,2, weight))
                        low_level_tokens.append(
                            sem_node.token_node.token_text + f"(partial matched)")
            if not (exact_match or fuzzy_match):
                low_level_sems.append((match["retrieved_sem"], 3, match["semantic_weight"]))
                low_level_tokens.append(match["retrieved_sem"].token_node.token_text + f"(sim matched)")

                high_level_tokens.append(['N/A'])

        if print_important_tokens:
            print(f"query tokens: {[text for text in query_tokens]}")
            print(f"low level tokens: {[text for text in low_level_tokens]}")
            print(f"high level tokens: {[text for text in high_level_tokens]}")

        co_occurrence_graph = CoOccurrenceGraph(low_level_sems + high_level_sems)
        co_occurrence_graph.build_edges()
        co_occurrence_graph.assign_chunk_weight(self.chunk_avg_len,print_important_tokens)
        co_occurrence_graph.rank_sem_node()
        co_occurrence_graph.rank_chunk_by_BM25()


        num_isolate_chunk = math.floor(top_k_chunk * isolate_chunk_ratio)
        num_connected_chunk = top_k_chunk - num_isolate_chunk

        retrieved_connected_chunk = [
            chunk_id for chunk_id, _ in co_occurrence_graph.weighted_chunk_node_list[:num_connected_chunk]
        ]
        connect_chunk_full = True if len(retrieved_connected_chunk) == num_connected_chunk else False
                                                                                                                                                              

        isolate_chunk_extractor = ListBatchExtractor(
            co_occurrence_graph.ranked_chunk_BM25,
            mode=isolate_retrieve_mode,
            k=top_k_each_isolated_chunk,
            exclude_list=retrieved_connected_chunk,
        )
        retrieved_isolated_chunk = isolate_chunk_extractor.extract(num_isolate_chunk, [])
        isolate_chunk_full = not isolate_chunk_extractor.finished

        if connect_chunk_full != isolate_chunk_full:
            if not connect_chunk_full:
                retrieved_isolated_chunk = isolate_chunk_extractor.extract(top_k_chunk-len(retrieved_connected_chunk), retrieved_isolated_chunk)
            else:
                retrieved_connected_chunk = [chunk_id for chunk_id, _ in
                                             co_occurrence_graph.weighted_chunk_node_list[:(top_k_chunk - num_isolate_chunk)]]


        return (
            self.chunk_id2text(retrieved_connected_chunk + retrieved_isolated_chunk),
            retrieved_connected_chunk + retrieved_isolated_chunk,
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
        query_tensor = torch.stack(query_embeds).to(self.device)          
        query_tensor = F.normalize(query_tensor, dim=1)
        sims = torch.matmul(self.query_database, query_tensor.T)
        best_scores, best_indices = torch.max(sims, dim=0)

        return best_indices.tolist(), best_scores.tolist()

    # Finalize graph metadata, semantic descriptions, retrieval indexes, and document output.
    def finalize(self):
        self._reset_sem_description_logs()
        self.log_time("Finalize started.")
        self.chunk_avg_len = sum([chunk_node.num_tokens for chunk_node in self.chunk_nodes])/len(self.chunk_nodes)
        self.log_time("Computed average chunk length.")
        self.finalize_token_nodes()
        self.log_time("Finished token node finalization.")
        self.merge_duplicate_description_sem_nodes()
        self.log_time("Finished merging sem nodes by description.")
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
        self.log_time("Finalizing completed.")

    # Ensure every token node has semantic nodes and absorb pending anomalies.
    def finalize_token_nodes(self):
        total_token_nodes = len(self.token_nodes)
        if total_token_nodes == 0:
            return

        progress_update = self._make_progress_updater("finalize_token_nodes", total_token_nodes)
        progress_update(0, force=True)
        for index, token_node in enumerate(self.token_nodes, start=1):
            if not token_node.has_semantic:
                self.create_basic_sem_node(token_node)
            elif len(token_node.anomaly_section) > 0:
                for item in token_node.anomaly_section:
                    self._append_sem_occurrence(
                        token_node.sem_node_list[item.max_idx],
                        item.text_embedding,
                        edge_weight=item.max_val,
                    )
                token_node.anomaly_section.clear()
            token_node.embeds_buffer.clear()
            progress_update(index)
        progress_update(total_token_nodes, force=True)
        print()

    # Sample unique chunks connected to a semantic node.
    def _sample_sem_chunk_nodes(self, sem_node, max_samples=10):
        unique_chunk_nodes = []
        seen_chunk_ids = set()
        for chunk_node in sem_node.chunk_node_list:
            if chunk_node.chunk_node_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_node.chunk_node_id)
            unique_chunk_nodes.append(chunk_node)
        if max_samples is None or max_samples <= 0:
            return unique_chunk_nodes
        if len(unique_chunk_nodes) <= max_samples:
            return unique_chunk_nodes
        return random.sample(unique_chunk_nodes, max_samples)

    # Sample span occurrences for semantic-description prediction.
    def _sample_sem_span_occurrences(self, sem_node, max_samples=10):
        span_occurrences = list(getattr(sem_node, "span_occurrences", []))
        if span_occurrences:
            if max_samples is None or max_samples <= 0:
                return span_occurrences
            if len(span_occurrences) <= max_samples:
                return span_occurrences
            return random.sample(span_occurrences, max_samples)

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

    # Extract the sentence containing the target token for description prediction.
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

    # Extract the target sentence plus neighboring sentence context for description prediction.
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

    # Use the full chunk as context for semantic-description prediction.
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
    def _load_sem_description_candidate_bank(self, sem_node, candidate_bank_cache):
        token_text = sem_node.token_node.token_text
        cached_candidate_bank = candidate_bank_cache.get(token_text)
        if cached_candidate_bank is not None:
            return cached_candidate_bank

        max_candidate_count = max(1, len(sem_node.token_node.sem_node_list) + 1)
        try:
            candidates_df, definition_column = load_wikidata_definition_candidates(
                token_text,
                use_detailed_description=self.sem_description_use_detailed_description,
                limit=self.sem_description_candidate_limit,
                target_candidate_count=max_candidate_count,
            )
        except ValueError:
            candidate_bank_cache[token_text] = []
            self._log_wikidata_no_result(
                token_text,
                "sem_description",
                "no_candidate_definitions",
            )
            return []

        candidate_bank = build_wikidata_candidate_bank(candidates_df, definition_column=definition_column)
        candidate_bank_cache[token_text] = candidate_bank
        if candidate_bank:
            self._log_sem_description_operation(
                "wikidata_candidate_lookup",
                token_text=token_text,
                candidate_limit=self.sem_description_candidate_limit,
                max_candidate_count=max_candidate_count,
                candidate_count=len(candidate_bank),
            )
            return candidate_bank

        self._log_wikidata_no_result(
            token_text,
            "sem_description",
            "empty_candidate_bank",
        )
        return []

    # Predict a semantic node description by scoring sampled contexts against candidates.
    def _predict_sem_description_from_samples(
        self,
        sem_node,
        model,
        candidate_bank_cache,
        max_samples=10,
    ):
        token_text = sem_node.token_node.token_text
        candidate_bank = self._load_sem_description_candidate_bank(sem_node, candidate_bank_cache)

        if not candidate_bank:
            self._log_wikidata_no_result(
                token_text,
                "sem_description",
                "empty_candidate_bank",
            )
            return None

        description_vote_map = {}
        sample_prediction_records = []
        sample_span_occurrences = self._sample_sem_span_occurrences(sem_node, max_samples=max_samples)
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

        return self._find_description_match_span(chunk_text, token_text)

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
        )

    # Rebuild the embedding and threshold for a semantic node created by splitting.
    def _finalize_split_sem_node_embed(self, sem_node):
        available_embeds = list(sem_node.chunk_node_embed)
        if available_embeds:
            sem_node.embed = average_embeds(available_embeds)
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
                    sem_node.embed = average_embeds(
                        [text_embedding.embed for text_embedding in regenerated_text_embeddings]
                    )
                    self._initialize_sem_retained_text_embeddings(
                        sem_node,
                        regenerated_text_embeddings,
                    )

        if sem_node.embed is None:
            raise ValueError(
                f"Failed to rebuild embed for split sem node {sem_node.sem_node_id} "
                f"({sem_node.token_node.token_text!r}, description={sem_node.description!r})."
            )

        if sem_node.chunk_edge_weight:
            sem_node.anomaly_threshold = get_anomaly_threshold(
                sem_node.chunk_edge_weight,
                self.anomaly_threshold_percentile,
            )
        else:
            sem_node.anomaly_threshold = None
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
        new_sem_node.anomaly_threshold = None
        new_sem_node.tf_dict_by_chunk_id = None
        new_sem_node.chunk_len_dict_by_id = None
        new_sem_node.BM25 = None
        new_sem_node.df = 0
        new_sem_node.idf = 0
        return new_sem_node

    # Merge semantic nodes that share the same predicted description.
    def _merge_sem_node_group(self, sem_group):
        primary_sem = sem_group[0]
        merged_chunk_nodes = []
        merged_chunk_edge_weights = []
        merged_span_occurrences = []
        merged_embeds = []
        merged_retained_text_embeddings = []
        merged_retained_source_count = 0

        for sem_node in sem_group:
            merged_chunk_nodes.extend(sem_node.chunk_node_list)
            merged_chunk_edge_weights.extend(sem_node.chunk_edge_weight)
            merged_span_occurrences.extend(getattr(sem_node, "span_occurrences", []))
            merged_retained_text_embeddings.extend(getattr(sem_node, "retained_text_embeddings", []))
            merged_retained_source_count += getattr(
                sem_node,
                "retained_text_embedding_source_count",
                len(getattr(sem_node, "retained_text_embeddings", [])),
            )
            if sem_node.embed is not None:
                merged_embeds.append(sem_node.embed)

        primary_sem.chunk_node_list = merged_chunk_nodes
        primary_sem.chunk_edge_weight = merged_chunk_edge_weights
        primary_sem.span_occurrences = merged_span_occurrences
        primary_sem.chunk_node_embed = []
        primary_sem.retained_text_embeddings = self._sample_text_embeddings(
            merged_retained_text_embeddings,
            max_samples=self.sem_retained_embed_limit,
        )
        primary_sem.retained_text_embedding_source_count = max(
            merged_retained_source_count,
            len(merged_retained_text_embeddings),
        )
        primary_sem.pending_embed_rebuild = False
        retained_embeds = [
            text_embedding.embed
            for text_embedding in merged_retained_text_embeddings
            if text_embedding.embed is not None
        ]
        if retained_embeds:
            primary_sem.embed = average_embeds(retained_embeds)
        elif merged_embeds:
            primary_sem.embed = torch.stack(merged_embeds).mean(dim=0)
        if primary_sem.chunk_edge_weight:
            primary_sem.anomaly_threshold = get_anomaly_threshold(
                primary_sem.chunk_edge_weight,
                self.anomaly_threshold_percentile,
            )
        else:
            primary_sem.anomaly_threshold = None
        primary_sem.tf_dict_by_chunk_id = None
        primary_sem.chunk_len_dict_by_id = None
        primary_sem.BM25 = None
        primary_sem.df = 0
        primary_sem.idf = 0
        return primary_sem

    # Assign descriptions, split ambiguous semantic nodes, and merge duplicates by description.
    def merge_duplicate_description_sem_nodes(self):
        consensus_ratio_threshold = 0.8
        redundant_sem_ids = set()
        sem_nodes_changed = False
        target_token_nodes = [
            token_node for token_node in self.token_nodes
            if len(token_node.sem_node_list) > 1
        ]
        candidate_bank_cache = {}

        total_token_nodes = len(target_token_nodes)
        progress_update = self._make_progress_updater(
            "merge_duplicate_description_sem_nodes",
            total_token_nodes,
        ) if total_token_nodes > 0 else None
        if progress_update is not None:
            progress_update(0, force=True)
        for index, token_node in enumerate(target_token_nodes, start=1):
            ordered_sem_list = []
            seen_sem_ids = set()
            for sem_node in token_node.sem_node_list:
                sem_id = id(sem_node)
                if sem_id in seen_sem_ids:
                    continue
                seen_sem_ids.add(sem_id)
                ordered_sem_list.append(sem_node)

            new_sem_list = []
            description_sem_map = defaultdict(list)
            for sem_node in ordered_sem_list:
                if sem_node.description:
                    new_sem_list.append(sem_node)
                    description_sem_map[sem_node.description].append(sem_node)
                    continue

                if self.sem_description_model is None:
                    new_sem_list.append(sem_node)
                    continue

                prediction_result = self._predict_sem_description_from_samples(
                    sem_node,
                    model=self.sem_description_model,
                    candidate_bank_cache=candidate_bank_cache,
                    max_samples=10,
                )
                if prediction_result is None:
                    new_sem_list.append(sem_node)
                    continue

                if (
                    prediction_result["sample_count"] > 0
                    and prediction_result["top_description_ratio"] >= consensus_ratio_threshold
                ):
                    predicted_description = prediction_result["description"]
                    sem_node.description = predicted_description
                    new_sem_list.append(sem_node)
                    description_sem_map[predicted_description].append(sem_node)
                    self.predicted_sem_description_logs.append(
                        {
                            "sem_node_id": sem_node.sem_node_id,
                            "token_text": sem_node.token_node.token_text,
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
                    continue

                exhaustive_prediction_result = self._predict_sem_description_from_samples(
                    sem_node,
                    model=self.sem_description_model,
                    candidate_bank_cache=candidate_bank_cache,
                    max_samples=None,
                )
                if exhaustive_prediction_result is None:
                    new_sem_list.append(sem_node)
                    continue

                sem_nodes_changed = True
                redundant_sem_ids.add(id(sem_node))
                self._log_sem_description_operation(
                    "split_sem_node_by_sample_labels",
                    token_text=token_node.token_text,
                    source_sem_node_id=sem_node.sem_node_id,
                    source_chunk_count=len(sem_node.chunk_node_list),
                    description_counts=exhaustive_prediction_result.get("description_counts", {}),
                )
                retained_text_embedding_lookup = self._build_retained_text_embedding_lookup(sem_node)
                self.deleted_merged_sem_logs.append(
                    {
                        "deleted_sem_node_id": sem_node.sem_node_id,
                        "kept_sem_node_id": None,
                        "token_text": token_node.token_text,
                        "description": "split_by_sample_label",
                        "deleted_chunk_count": len(sem_node.chunk_node_list),
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
                    target_sem_group = description_sem_map.get(predicted_description)
                    created_new_sem = False
                    if target_sem_group:
                        target_sem = target_sem_group[0]
                    else:
                        target_sem = self._create_split_sem_node(token_node, predicted_description)
                        description_sem_map[predicted_description].append(target_sem)
                        new_sem_list.append(target_sem)
                        created_new_sem = True
                        self._log_sem_description_operation(
                            "create_split_sem_node",
                            token_text=token_node.token_text,
                            sem_node_id=target_sem.sem_node_id,
                            description=predicted_description,
                        )
                    self._append_span_occurrence_to_sem(
                        target_sem,
                        span_occurrence,
                        edge_weight=0.0,
                        retained_text_embedding=retained_text_embedding,
                    )
                    self._log_sem_description_operation(
                        "assign_sample_to_sem_node",
                        token_text=token_node.token_text,
                        source_sem_node_id=sem_node.sem_node_id,
                        target_sem_node_id=target_sem.sem_node_id,
                        description=predicted_description,
                        chunk_node_id=span_occurrence.chunk_node.chunk_node_id,
                        span_start=span_occurrence.span_start,
                        span_end=span_occurrence.span_end,
                        span_text=span_occurrence.span_text,
                        used_retained_embedding=retained_text_embedding is not None,
                        created_new_sem=created_new_sem,
                    )

            for sem_node in new_sem_list:
                if getattr(sem_node, "pending_embed_rebuild", False):
                    self._finalize_split_sem_node_embed(sem_node)
                    self._log_sem_description_operation(
                        "finalize_split_sem_node_embed",
                        token_text=token_node.token_text,
                        sem_node_id=sem_node.sem_node_id,
                        description=sem_node.description,
                        retained_embed_count=len(getattr(sem_node, "retained_text_embeddings", [])),
                    )

            merged_sem_map = {}
            for description, sem_group in defaultdict(list, {
                description: [sem_node for sem_node in new_sem_list if sem_node.description == description]
                for description in {sem_node.description for sem_node in new_sem_list if sem_node.description}
            }).items():
                if len(sem_group) < 2:
                    continue
                merged_sem = self._merge_sem_node_group(sem_group)
                merged_sem_map[description] = merged_sem
                self._log_sem_description_operation(
                    "merge_sem_nodes_with_same_description",
                    token_text=token_node.token_text,
                    description=description,
                    kept_sem_node_id=merged_sem.sem_node_id,
                    merged_sem_node_ids=[sem_node.sem_node_id for sem_node in sem_group],
                    retained_embed_count=len(getattr(merged_sem, "retained_text_embeddings", [])),
                )
                for redundant_sem in sem_group[1:]:
                    sem_nodes_changed = True
                    redundant_sem_ids.add(id(redundant_sem))
                    self.deleted_merged_sem_logs.append(
                        {
                            "deleted_sem_node_id": redundant_sem.sem_node_id,
                            "kept_sem_node_id": merged_sem.sem_node_id,
                            "token_text": token_node.token_text,
                            "description": description,
                            "deleted_chunk_count": len(redundant_sem.chunk_node_list),
                        }
                    )

            if merged_sem_map:
                deduped_sem_list = []
                added_sem_ids = set()
                for sem_node in new_sem_list:
                    if id(sem_node) in redundant_sem_ids:
                        continue
                    target_sem = merged_sem_map.get(sem_node.description, sem_node)
                    target_sem_id = id(target_sem)
                    if target_sem_id in added_sem_ids:
                        continue
                    added_sem_ids.add(target_sem_id)
                    deduped_sem_list.append(target_sem)
                new_sem_list = deduped_sem_list

            token_node.sem_node_list = new_sem_list
            token_node.has_semantic = len(token_node.sem_node_list) > 0
            progress_update(index)

        if not redundant_sem_ids and not sem_nodes_changed:
            if total_token_nodes > 0:
                progress_update(total_token_nodes, force=True)
                print()
            return

        self.sem_nodes = [
            sem_node for sem_node in self.sem_nodes
            if id(sem_node) not in redundant_sem_ids
        ]
        for index, sem_node in enumerate(self.sem_nodes):
            sem_node.sem_node_id = index
        self.next_sem_node_id = len(self.sem_nodes)
        if total_token_nodes > 0:
            progress_update(total_token_nodes, force=True)
            print()

    # Format semantic-description logs as plain text.
    def _format_sem_description_logs_text(self):
        lines = []

        lines.append("Sem description operations:")
        if not self.sem_description_operation_logs:
            lines.append("  (none)")
        else:
            for item in self.sem_description_operation_logs:
                event_type = item.get("event_type", "unknown")
                payload = ", ".join(
                    f"{key}={value!r}"
                    for key, value in item.items()
                    if key != "event_type"
                )
                lines.append(f"  [{event_type}] {payload}")

        lines.append("")
        lines.append("Sem nodes assigned predicted descriptions:")
        if not self.predicted_sem_description_logs:
            lines.append("  (none)")
        else:
            for item in self.predicted_sem_description_logs:
                lines.append(
                    "[Sem] sem_node_id={} | token={!r} | description={!r} | chunk_count={}".format(
                        item["sem_node_id"],
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

        lines.append("Sem nodes deleted by description merge:")
        if not self.deleted_merged_sem_logs:
            lines.append("  (none)")
        else:
            for item in self.deleted_merged_sem_logs:
                lines.append(
                    "  deleted_sem_node_id={} | kept_sem_node_id={} | token={!r} | description={!r} | deleted_chunk_count={}".format(
                        item["deleted_sem_node_id"],
                        item["kept_sem_node_id"],
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

    # Write semantic-description logs and return the output path.
    def print_sem_description_logs(self, output_path=None):
        return self.show_sem_description_logs(output_path=output_path)

    # Persist semantic-description logs to a text file.
    def show_sem_description_logs(self, as_html=True, open_details=False, output_path=None):
        output_path = self.sem_description_log_path if output_path is None else output_path
        log_text = self._format_sem_description_logs_text()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(log_text)
        return output_path


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

    # Route extracted token embeddings into buffers, semantic nodes, or anomaly queues.
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
                if token_node.has_semantic:
                    max_val, max_idx = inspect_sem_nodes(embed, token_node.sem_node_list)
                    if max_val >= token_node.sem_node_list[max_idx].anomaly_threshold:
                        self._append_sem_occurrence(
                            token_node.sem_node_list[max_idx],
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
                self.build_sem_node_waitlist.append(token_node)

    # Index one document using the selected single or multi-processing path.
    def index_document(self, doc_name, multiprocessing=True):
        self.start_time = time.perf_counter()
        document_start_time = time.perf_counter()
        if multiprocessing:
            self.index_document_multi_processing(doc_name)
        else:
            self.index_document_single_processing(doc_name)
        self.log_time(f"File {doc_name} completed. Index time: {time.perf_counter() - document_start_time:.4f}s")


    # Index one document by encoding chunks in a single processing flow.
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
        self.solve_sem_nodes()
        self.solve_anomaly()

    # Index one document by submitting chunk encoding work to the executor.
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
        self.solve_sem_nodes()
        self.solve_anomaly()

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
        for chunk_node in self.chunk_nodes:
            chunk_node.sem_node_list = []
        for sem_node in self.sem_nodes:
            for chunk_node in sem_node.chunk_node_list:
                if not id(sem_node) in (id(x) for x in chunk_node.sem_node_list):
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
            sem_node.chunk_node_list = [sem_node.chunk_node_list[i] for i in keep_indices]
            sem_node.span_occurrences = [
                sem_node.span_occurrences[i]
                for i in keep_indices
            ] if getattr(sem_node, "span_occurrences", None) else []
            sem_node.chunk_edge_weight = [sem_node.chunk_edge_weight[i] for i in keep_indices]
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
        for index, sem_node in enumerate(self.sem_nodes):
            sem_node.sem_node_id = index
        self.next_sem_node_id = len(self.sem_nodes)

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
            token_node.anomaly_section = [
                item for item in token_node.anomaly_section
                if id(item.text_embedding.chunk_node) in valid_chunk_ids
            ]
            token_node.span_occurrences = [
                item for item in token_node.span_occurrences
                if id(item.chunk_node) in valid_chunk_ids
            ]
            if token_node.has_semantic or token_node.embeds_buffer or token_node.anomaly_section:
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

        self.build_sem_node_waitlist = [
            token_node for token_node in self.build_sem_node_waitlist
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
            self.get_sem_BM25()
        else:
            for sem_node in self.sem_nodes:
                sem_node.BM25 = {}

        self.build_chunk2sem_edge()
        self.phrase_index = defaultdict(set)
        self.build_phrase_query()

        if self.sem_nodes:
            self.build_query_database()
        else:
            self.query_database = None


    # Return the index of the semantic node most similar to an embedding.
    def get_max_sim_sem(self, token_node, embeds):
        sem_embeds = [sem_node.embed.to(self.device) for sem_node in token_node.sem_node_list]
        x = torch.stack(sem_embeds, 0).to(self.device)
        q = embeds.unsqueeze(0).expand_as(x)
        return int(F.cosine_similarity(x, q, dim=1).argmax().item())


    # Index one document with staged CPU preprocessing, GPU encoding, and CPU consumption.
    def index_document_parallel(self, doc_name, batch_size=8, queue_size=4):
        self.start_time = time.perf_counter()
        document_start_time = time.perf_counter()

                                       
                                  
                                       
        new_doc_node = self.create_doc_node(doc_name)
        chunk_list = split_doc(doc_name, self.nlp, max_tokens=self.chunk_size)
        total_chunks = len(chunk_list)

        if total_chunks == 0:
                         
            self.solve_sem_nodes()
            self.solve_anomaly()
            self.log_time(
                f"File {doc_name} completed. Index time: {time.perf_counter() - document_start_time:.4f}s"
            )
            return

                                       
                                  
                                       
                                                                     
        chunk_meta = []                                               
        for chunk_text in chunk_list:
            new_chunk_node = self.create_chunk_node(chunk_text, new_doc_node)

            phrases, tokens = extract_important_spans(
                chunk_text,
                self.nlp,
                min_tokens=2,
                remove_duplicate=self.remove_duplicate_token
            )

            chunk_meta.append((new_chunk_node, phrases, tokens, chunk_text))

                                       
                          
                                       
        batches = []
        for i in range(0, total_chunks, batch_size):
            batch_indices = list(range(i, min(i + batch_size, total_chunks)))
            batch_texts = [chunk_meta[idx][3] for idx in batch_indices]              
            batches.append((batch_texts, batch_indices))

                                       
                   
                                       
        result_queue = queue.Queue(maxsize=queue_size)

        gpu_done = 0
        cpu_done = 0

                                       
                   
                                       
        # Encode prepared chunk batches on the GPU worker thread.
        def gpu_worker():
            nonlocal gpu_done
            try:
                for batch_texts, batch_indices in batches:
                    token_embeddings_batch, offsets_batch = encode_chunk_batch(batch_texts, self.text_encoder, self.tokenizer, self.device)

                    result_queue.put((token_embeddings_batch, offsets_batch, batch_indices))
                    gpu_done += len(batch_indices)

                      
                result_queue.put(None)

            except Exception as e:
                                    
                result_queue.put(("__EXCEPTION__", e))
                result_queue.put(None)

        gpu_thread = threading.Thread(target=gpu_worker, daemon=True)
        gpu_thread.start()

                                       
                      
                                       
        while True:
            item = result_queue.get()

            if item is None:
                break

            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__EXCEPTION__":
                               
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

                                       
                                   
                                       
        self.solve_sem_nodes()
        self.solve_anomaly()
        self.log_time(
            f"File {doc_name} completed. Index time: {time.perf_counter() - document_start_time:.4f}s"
        )

    # Map chunk IDs to their stored chunk texts.
    def chunk_id2text(self, ids):
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

    # Format semantic-node inspection results as plain text.
    def _format_multi_sem_token_nodes_text(self, inspect_data):
        lines = []

        for token_text, token_info in inspect_data.items():
            lines.append(
                f"[Token] {token_text} | token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']} | "
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
            token_meta = (
                f"<span style='color:#666;'>"
                f"token_node_id={token_info['token_node_id']} | "
                f"sem_count={token_info['sem_count']}, "
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
                return HTML(self._build_multi_sem_token_nodes_html(inspect_data, open_details=open_details))
            except ImportError:
                pass

        return self._format_multi_sem_token_nodes_text(inspect_data)

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


                       
    # Prepare the graph for pickling by removing runtime-only components.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["text_encoder"] = None
        state["tokenizer"] = None
        state["executor"] = None
        state["nlp"] = None
        state["reranker"] = None
        state["sem_description_model"] = None
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
        self._ensure_backward_compatible_attrs()

    # Reload runtime components needed after deserialization.
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
        if self.sem_description_model is None:
            self._load_sem_description_model()

    # Save the full graph object to a pickle file.
    def save_data(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

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
        return obj

    # Save graph structure and tensors into separate files.
    def save_data_split(self, pkl_path: str):
        pt_path = pkl_path.replace(".pkl", "_tensors.pt")
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
                )
                for text_embedding in getattr(p, "retained_text_embeddings", [])
            ]

                                              
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

                                
        tensors_path_bak = getattr(self, "tensors_path", None)
        self.tensors_path = pt_path
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
                                  
            self.query_database = qbak
            for p, e, retained in zip(self.sem_nodes, pbak, retained_bak):
                p.embed = e
                p.retained_text_embeddings = retained

            if hasattr(self, "executor"): self.executor = exec_bak
            if hasattr(self, "text_encoder"): self.text_encoder = enc_bak
            if hasattr(self, "tokenizer"): self.tokenizer = tok_bak
            if hasattr(self, "nlp"): self.nlp = nlp_bak
            if hasattr(self, "reranker"): self.reranker = reranker_bak
            self.node_id2instance()
                                  
            self.tensors_path = tensors_path_bak


    # Load split graph structure and tensor files, then restore runtime components.
    @classmethod
    def load_data_split(cls, pkl_path: str):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

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
                )
                for item in retained_pack
            ]
            p.retained_text_embedding_source_count = max(
                getattr(p, "retained_text_embedding_source_count", 0),
                len(p.retained_text_embeddings),
            )

                      
        obj._restore_runtime_components(load_nlp=True, load_reranker=False)
        obj.node_id2instance()
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

    # Convert serialized integer IDs back into object references.
    def node_id2instance(self):
        for chunk_node in self.chunk_nodes:
            chunk_node.doc_node = self.doc_nodes[chunk_node.doc_node]
            chunk_node.sem_node_list = [self.sem_nodes[idx] for idx in chunk_node.sem_node_list]
        for doc_node in self.doc_nodes:
            doc_node.chunk_node_list = [self.chunk_nodes[idx] for idx in doc_node.chunk_node_list]
        for sem_node in self.sem_nodes:
            sem_node.chunk_node_list = [self.chunk_nodes[idx] for idx in sem_node.chunk_node_list]
            sem_node.token_node = self.token_nodes[sem_node.token_node]
            if not hasattr(sem_node, "description"):
                sem_node.description = self._get_initial_sem_description(sem_node.token_node)
            if not hasattr(sem_node, "span_occurrences") or sem_node.span_occurrences is None:
                sem_node.span_occurrences = []
            if not sem_node.span_occurrences and sem_node.chunk_node_list:
                sem_node.span_occurrences = [
                    SpanOccurrence(chunk_node=chunk_node)
                    for chunk_node in sem_node.chunk_node_list
                ]
            if not hasattr(sem_node, "retained_text_embeddings") or sem_node.retained_text_embeddings is None:
                sem_node.retained_text_embeddings = []
            for text_embedding in sem_node.retained_text_embeddings:
                text_embedding.chunk_node = self.chunk_nodes[text_embedding.chunk_node]
            if not hasattr(sem_node, "retained_text_embedding_source_count"):
                sem_node.retained_text_embedding_source_count = len(sem_node.retained_text_embeddings)
            if not hasattr(sem_node, "pending_embed_rebuild"):
                sem_node.pending_embed_rebuild = False
        for token_node in self.token_nodes:
            token_node.sem_node_list = [self.sem_nodes[idx] for idx in token_node.sem_node_list]

    # Index a list of document records with a staged CPU/GPU pipeline.
    def index_json(self, chunk_list, batch_size=8, queue_size=4):
        self.start_time = time.perf_counter()
        self._reset_hdbscan_stats()
        total_chunks = len(chunk_list)
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

            except Exception:
                record_error("gpu_worker")

            finally:
                safe_queue_put(result_queue, None)

                                       
              
                                       
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

        self.log_time(
            f"Index {total_chunks} documents. "
            f"Index time: {time.perf_counter() - self.start_time:.4f}s"
        )

        self.solve_sem_nodes()
        self.solve_anomaly()
        print(
            "HDBSCAN attempts: "
            f"{self.hdbscan_attempt_count}, "
            f"successes (n_clusters >= 1): {self.hdbscan_success_count}"
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

    # Extract up to N values while preserving previous extraction state.
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

                                   
              
                                   
    # Extract values in round-robin order across source lists.
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

                                   
                   
                                   
    # Extract values from source lists sequentially.
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
