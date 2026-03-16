import json
import math
import pickle
import time
import traceback
from collections import Counter, defaultdict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import queue
import threading

import spacy
import torch
import torch.nn.functional as F
from localizeJina import LocalJinaReranker
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
    get_anomaly_threshold,
    get_s_mean,
    hdbscan_cluster,
    inspect_prototypes,
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
    is_semantic: bool = field(init=False)
    has_prototype: bool = False
    proto_node_list: list = field(default_factory=list)
    embeds_buffer: list = field(default_factory=list)
    idf: float = 0
    df: int = 0
    anomaly_section: list = field(default_factory=list)

    def __post_init__(self):
        self.is_semantic = self.node_type == "phrase"


@dataclass
class TextEmbedding:
    embed: object
    chunk_node: ChunkNode


@dataclass
class Prototype:
    proto_node_id: int
    token_node: TokenNode
    chunk_node_list: list = field(default_factory=list)
    chunk_node_embed: list = field(default_factory=list)
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
                 retrieve_top_k=5, chunk_size=300, remove_duplicate_token=True, device="cuda", discard_no_word=False):
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
        #self.load_reranker()
        self.chunk_avg_len = None
        self.discard_no_word = discard_no_word

    def load_reranker(self):
        self.reranker = LocalJinaReranker()

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
        self.proto_nodes.append(new_proto_node)
        token_node.has_prototype = True
        return new_proto_node

    def create_basic_proto_node(self, token_node):
        new_proto_node = self.create_proto_node(token_node)
        new_proto_node.chunk_node_embed = ([i.embed for i in token_node.embeds_buffer])
        new_proto_node.embed = average_embeds(new_proto_node.chunk_node_embed)
        new_proto_node.chunk_node_list = [k.chunk_node for k in token_node.embeds_buffer]
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
            if n_clusters >= 1:
                self.plot_embed_distribution(token_node, clusters)
                for clusters_label in range(n_clusters):
                    new_proto_node = self.create_proto_node(token_node)
                    new_proto_node.embed = torch.from_numpy(cluster_centers[clusters_label])
                    for idx in clusters[clusters_label]:
                        new_proto_node.chunk_node_list.append(token_node.embeds_buffer[idx].chunk_node)
                        new_proto_node.chunk_node_embed.append(token_node.embeds_buffer[idx].embed)
                    new_proto_node.chunk_edge_weight = proto_embed_sim(new_proto_node).cpu().tolist()
                    new_proto_node.anomaly_threshold = get_anomaly_threshold(new_proto_node.chunk_edge_weight,
                                                                             self.anomaly_threshold_percentile)
                    new_proto_node.chunk_node_embed.clear()
                    token_node.proto_node_list.append(new_proto_node)
                anomaly_idx = clusters.get(-1)
                if anomaly_idx is not None:
                    for idx in anomaly_idx:
                        max_val, max_idx = inspect_prototypes(token_node.embeds_buffer[idx].embed, token_node.proto_node_list)
                        token_node.anomaly_section.append((token_node.embeds_buffer[idx].embed, token_node.embeds_buffer[idx].chunk_node, max_val, max_idx))
                token_node.is_semantic = True
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
                [(embed.cpu(), chunk_node) for embed, chunk_node, _, _ in token_node.anomaly_section],
                min_cluster_size=10, percentile=self.anomaly_threshold_percentile)
            if n_clusters >= 1:
                for clusters_label in range(n_clusters):
                    new_proto_node = self.create_proto_node(token_node)
                    new_proto_node.embed = torch.from_numpy(cluster_centers[clusters_label]).to(self.device)
                    for idx in clusters[clusters_label]:
                        new_proto_node.chunk_node_list.append(token_node.anomaly_section[idx][1])
                        new_proto_node.chunk_node_embed.append(token_node.anomaly_section[idx][0])
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
                for embed, chunk_node, max_val, max_idx in token_node.anomaly_section:
                    token_node.proto_node_list[max_idx].chunk_node_list.append(chunk_node)
                    token_node.proto_node_list[max_idx].chunk_edge_weight.append(max_val)
        self.anomaly_waitlist = []

    # def assign_edge_weight(self, proto_node):
    #     sim = proto_embed_sim(proto_node)
    #     proto_node.chunk_node_embed.clear()
    #     proto_node.chunk_edge_weight = proto_node_combine_sim(sim, proto_node)

    def log_time(self, msg):
        print(f"[{time.perf_counter() - self.start_time:.4f}s] {msg}")

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
                #low_level_tokens.append(token + f"(exact matched), idf:{max_proto_node.token_node.idf}")
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
                        #high_level_tokens.append(proto_node.token_node.token_text + f"(partial matched), idf:{proto_node.token_node.idf}")
                        high_level_tokens.append(
                            proto_node.token_node.token_text + f"(partial matched)")
                    else:
                        if token_type == 'ent':
                            low_level_protos.append((proto_node,1, weight))
                        else:
                            low_level_protos.append((proto_node,2, weight))
                        #low_level_tokens.append(proto_node.token_node.token_text + f"(partial matched), idf:{proto_node.token_node.idf}")
                        low_level_tokens.append(
                            proto_node.token_node.token_text + f"(partial matched)")
            if not (exact_match or fuzzy_match):
                low_level_protos.append((match["semantic_proto"], 3, match["semantic_weight"]))
                #low_level_tokens.append(self.proto_nodes[proto_node_indices[0]].token_node.token_text  + f"(sim matched), idf:{self.proto_nodes[proto_node_indices[0]].token_node.idf}")
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
        self.chunk_avg_len = sum([chunk_node.num_tokens for chunk_node in self.chunk_nodes])/len(self.chunk_nodes)
        for token_node in self.token_nodes:
            if not token_node.has_prototype:
                self.create_basic_proto_node(token_node)
            elif len(token_node.anomaly_section) > 0:
                for embed, chunk_node, max_val, max_idx in token_node.anomaly_section:
                    token_node.proto_node_list[max_idx].chunk_node_list.append(chunk_node)
                    token_node.proto_node_list[max_idx].chunk_edge_weight.append(max_val)
                token_node.anomaly_section.clear()
            token_node.embeds_buffer.clear()
            self.assign_idf(token_node)
        self.get_proto_BM25()
        self.build_query_database()
        self.build_phrase_query()
        self.build_chunk2proto_edge()
        self.save_doc_to_json()
        self.log_time("Finalizing completed.")

    def get_proto_BM25(self):
        for proto_node in self.proto_nodes:
            proto_node.get_BM25(self.chunk_avg_len)

    def process_embeds(self, new_chunk_node, phrase_embs, token_embs):
        for text, embed in phrase_embs + token_embs:
            token_node = self.query_token_node(text)
            if token_node is None:
                token_node = self.create_token_node(text)
                token_node.embeds_buffer.append(TextEmbedding(embed, new_chunk_node))
            else:
                if token_node.has_prototype:
                    max_val, max_idx = inspect_prototypes(embed, token_node.proto_node_list)
                    if max_val >= token_node.proto_node_list[max_idx].anomaly_threshold:
                        token_node.proto_node_list[max_idx].chunk_node_list.append(new_chunk_node)
                        token_node.proto_node_list[max_idx].chunk_edge_weight.append(max_val)
                    else:
                        token_node.anomaly_section.append((embed, new_chunk_node, max_val, max_idx))
                else:
                    token_node.embeds_buffer.append(TextEmbedding(embed, new_chunk_node))
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
            proto_node.chunk_edge_weight = [proto_node.chunk_edge_weight[i] for i in keep_indices]
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
                if id(item[1]) in valid_chunk_ids
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
            node.chunk_edge_weight = [x for i, x in enumerate(node.chunk_edge_weight) if i not in index_to_delete]
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
            fix_mistral_regex=True
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
        return state

    # ⭐ 反序列化后做什么
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_text_encoder()
        self.save_doc_to_json()
        self.executor = ThreadPoolExecutor(max_workers=4)

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
        obj._load_nlp()
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
        }
        torch.save(tensor_pack, pt_path)

        # 2) 临时把对象里的 tensor 清空，避免 pickle 遇到 torch.Storage
        qbak = self.query_database
        pbak = [p.embed for p in self.proto_nodes]

        self.query_database = None
        for p in self.proto_nodes:
            p.embed = None

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
            for p, e in zip(self.proto_nodes, pbak):
                p.embed = e

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

        obj.query_database = tensor_pack.get("query_database", None).to(obj.device)

        proto_embeds = tensor_pack.get("proto_embeds", [])
        # 按索引回填
        for i, p in enumerate(obj.proto_nodes):
            p.embed = proto_embeds[i].to("cpu") if i < len(proto_embeds) else None

        # 重建运行态（按你原逻辑）
        obj._load_text_encoder()
        obj.save_doc_to_json()
        obj.executor = ThreadPoolExecutor(max_workers=4)
        obj._load_nlp()
        obj.node_id2instance()
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
        for token_node in self.token_nodes:
            token_node.proto_node_list = [self.proto_nodes[idx] for idx in token_node.proto_node_list]

    def index_json(self, chunk_list, batch_size=8, queue_size=4):
        self.start_time = time.perf_counter()
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

        def record_error(stage_name):
            worker_errors.append(
                (
                    stage_name,
                    traceback.format_exc()
                )
            )

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

        def print_progress():
            p_done, g_done, c_done = get_progress_snapshot()
            print(
                f"\rCPU preprocessed: {p_done}/{total_chunks} | "
                f"GPU encoded: {g_done}/{total_chunks} | "
                f"CPU processed: {c_done}/{total_chunks}",
                end="",
                flush=True
            )

        # =============================
        # 阶段 1：CPU 预处理线程
        # 输出到 preprocess_queue:
        # (idx, new_chunk_node, phrases, tokens, text)
        # =============================
        def cpu_preprocess_worker():
            try:
                for idx, chunk in enumerate(chunk_list):
                    title = chunk["title"]
                    new_doc_node = doc_node_map.get(title)
                    if new_doc_node is None:
                        new_doc_node = self.create_doc_node(title)
                        doc_node_map[title] = new_doc_node
                    new_chunk_node = self.create_chunk_node(
                        chunk["text"], new_doc_node
                    )

                    phrases, tokens = extract_important_spans(
                        clean_text(chunk["text"]),
                        self.nlp,
                        min_tokens=2,
                        remove_duplicate=self.remove_duplicate_token,
                        discard_no_word=self.discard_no_word
                    )

                    preprocess_queue.put(
                        (
                            idx,
                            new_chunk_node,
                            phrases,
                            tokens,
                            chunk["text"]
                        )
                    )

                    inc_preprocess_done(1)

            except Exception:
                record_error("cpu_preprocess_worker")

            finally:
                # 无论成功还是失败，都要发结束信号，避免下游卡死
                preprocess_queue.put(None)

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
                    return

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

                result_queue.put(
                    (
                        batch_items,
                        token_embeddings_batch,
                        offsets_batch
                    )
                )

                inc_gpu_done(len(batch_buffer))

            batch_buffer = []

            try:
                while True:
                    item = preprocess_queue.get()

                    if item is None:
                        break

                    batch_buffer.append(item)

                    if len(batch_buffer) >= batch_size:
                        flush_batch(batch_buffer)
                        batch_buffer = []

                # 刷掉最后不足一个 batch 的部分
                if batch_buffer:
                    flush_batch(batch_buffer)

            except Exception:
                record_error("gpu_worker")

            finally:
                result_queue.put(None)

        # =============================
        # 启动线程
        # =============================
        preprocess_thread = threading.Thread(target=cpu_preprocess_worker)
        gpu_thread = threading.Thread(target=gpu_worker)

        preprocess_thread.start()
        gpu_thread.start()

        # =============================
        # 阶段 3：主线程消费 GPU 结果
        # =============================
        try:
            while True:
                item = result_queue.get()

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
            preprocess_thread.join()
            gpu_thread.join()

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


