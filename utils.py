from __future__ import annotations
import numpy as np
import hdbscan
from collections import defaultdict
import torch, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pympler import asizeof
import math
from typing import Sequence, Optional, Set

def mrr_for_one_query_titles(
    ranked_titles: Sequence[str],
    gold_titles: Sequence[str],
    k: Optional[int] = 10,
    normalize: bool = True,
) -> float:
    """
    MRR@k for one query, where both retrieved items and gold labels are title strings.

    - Multi-gold: the first occurrence of ANY gold title determines reciprocal rank.
    - If no gold appears in top-k, returns 0.0
    """

    def norm(s: str, do_norm: bool = normalize) -> str:
        s = str(s).strip()
        if do_norm:
            s = s.lower()
            s = " ".join(s.split())  # collapse multiple spaces/tabs/newlines
        return s

    gold_set: Set[str] = {norm(t) for t in gold_titles if t is not None and str(t).strip()}
    if not gold_set:
        return 0.0

    end = len(ranked_titles) if k is None else min(k, len(ranked_titles))
    for idx in range(end):
        t = ranked_titles[idx]
        if t is None:
            continue
        if norm(t) in gold_set:
            return 1.0 / (idx + 1)  # 1-indexed rank
    return 0.0

time_stamp = datetime.now().strftime("%m-%d-%H-%M")

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n + eps)

# 其实在 100 样本规模：
#
# 👉 HDBSCAN 不一定是最优
# 👉 有时层次聚类（Agglomerative）更清晰
# from sklearn.cluster import AgglomerativeClustering
#
# clustering = AgglomerativeClustering(
#     n_clusters=3,
#     affinity='euclidean',
#     linkage='ward'
# )


def hdbscan_cluster(
    embeds_list,
    min_cluster_size=10,
    percentile=0.9,
    merge_chunks=False
    ):
    """
    embeds_list: List[(embedding_tensor, chunk_node)]
    merge_chunks: 是否按照 chunk_node_id 聚合
    """

    # Kept for backward compatibility with existing callers.
    _ = percentile
    original_count = len(embeds_list)

    merged_embeds = []
    merged_to_original = []

    # =========================
    # 1️⃣ 聚合 or 不聚合
    # =========================
    if merge_chunks:

        chunk_groups = defaultdict(list)

        for idx, (embed, chunk_node) in enumerate(embeds_list):
            chunk_id = chunk_node.chunk_node_id
            chunk_groups[chunk_id].append((idx, embed))

        for items in chunk_groups.values():

            indices = []
            embeds = []

            for idx, embed in items:
                indices.append(idx)
                embeds.append(embed.detach().cpu().numpy())

            embeds = np.array(embeds)

            merged_embed = np.mean(embeds, axis=0)

            merged_embeds.append(merged_embed)
            merged_to_original.append(indices)

    else:

        for idx, (embed, _) in enumerate(embeds_list):

            merged_embeds.append(embed.detach().cpu().numpy())
            merged_to_original.append([idx])

    merged_embeds = np.array(merged_embeds)
    merged_count = len(merged_embeds)

    # =========================
    # 2️⃣ min_cluster_size 缩放
    # =========================
    if merge_chunks:
        k = merged_count / original_count
    else:
        k = 1

    scaled_min_cluster_size = max(2, math.ceil(min_cluster_size * k))

    # =========================
    # 3️⃣ L2 normalize
    # =========================
    norms = np.linalg.norm(merged_embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = merged_embeds / norms

    # =========================
    # 4️⃣ PCA
    # =========================
    pca = PCA(n_components=min(50, X_norm.shape[1]))
    X_reduced = pca.fit_transform(X_norm)

    # =========================
    # 5️⃣ HDBSCAN
    # =========================
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=scaled_min_cluster_size,
        # min_samples=max(2, scaled_min_cluster_size // 2),
        metric="euclidean",
        # cluster_selection_method='eom',
        # cluster_selection_epsilon=0.05
    )

    cluster_labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    # =========================
    # 6️⃣ 映射回原始 index
    # =========================
    clusters = defaultdict(list)

    for merged_idx, label in enumerate(cluster_labels):

        original_indices = merged_to_original[merged_idx]

        clusters[label].extend(original_indices)

    # =========================
    # 7️⃣ cluster centers
    # =========================
    cluster_centers = {}

    for label in clusters.keys():

        merged_indices = [
            i for i, l in enumerate(cluster_labels) if l == label
        ]

        cluster_embeds = X_norm[merged_indices]

        center = np.mean(cluster_embeds, axis=0)

        norm = np.linalg.norm(center)
        if norm > 0:
            center /= norm

        cluster_centers[label] = center

    return n_clusters, clusters, cluster_centers

def get_anomaly_threshold(values, percentile):
    q = 1 - percentile
    return float(np.percentile(values, q * 100))

# def get_anomaly_threshold(center, embeds_list, percentile):
#     embeds = torch.stack(embeds_list)
#
#     center = torch.nn.functional.normalize(center, p=2, dim=0)
#     embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
#
#     similarities = embeds @ center
#
#     threshold = torch.quantile(similarities, 1 - percentile)
#
#     return threshold.item()

# def hdbscan_cluster(embeds_list, min_size=5):
#     X = np.array(embeds_list)
#
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=min_size,
#         metric='cosine'
#     )
#
#     cluster_labels = clusterer.fit_predict(X)
#     n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
#
#     clusters = defaultdict(list)
#
#     for idx, label in enumerate(cluster_labels):
#         clusters[label].append(idx)
#
#     clusters.pop(-1, None)
#     cluster_centers = {}
#
#     for label, indices in clusters.items():
#         cluster_embeds = X[indices]
#         center = np.mean(cluster_embeds, axis=0)
#         norm = np.linalg.norm(center)
#         if norm > 0:
#             center = center / norm
#
#         cluster_centers[label] = center
#     return n_clusters, clusters, cluster_centers

def e5_average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# def get_s_mean(embeds_buffer):
#     embeds_mat = np.stack(embeds_buffer, axis=0)
#     mu = l2_normalize(embeds_mat.mean(axis=0))
#     E_norm = embeds_mat / (np.linalg.norm(embeds_mat, axis=1, keepdims=True) + 1e-12)
#     s_mean = float(E_norm @ mu).mean()
#     return s_mean

def get_s_mean(embeds_buffer):
    embeds_mat = torch.stack(embeds_buffer)  # [N, 1024]
    mu = F.normalize(embeds_mat.mean(dim=0), dim=0)
    E_norm = F.normalize(embeds_mat, dim=1)

    s_mean = torch.matmul(E_norm, mu).mean()
    return s_mean.item()


def sorted_indices(values: Sequence[float]) -> list[int]:
    """
    Sorts the list of values and returns the indices of the sorted order.

    Args:
        values (Sequence[float]): The input list of values to be sorted.

    Returns:
        list[int]: Indices of the values in sorted order.
    """
    return sorted(range(len(values)), key=lambda i: values[i])

def average_embeds(tensors, eps=1e-12):
    x = torch.stack(tensors, dim=0)
    x = F.normalize(x, p=2, dim=1)
    m = x.mean(dim=0)
    return F.normalize(m, p=2, dim=0)


def proto_embed_sim(proto_node):
    proto_embed = proto_node.embed
    chunk_node_embed = torch.stack(proto_node.chunk_node_embed)
    proto_embed = proto_embed.unsqueeze(0)  # [1, d]
    sim = F.cosine_similarity(chunk_node_embed, proto_embed, dim=1)

    return sim

def proto_node_combine_sim(sim, proto_node, r=3):
    """
    sim: list/iterable of similarity floats (same length as proto_node.chunk_node)
    proto_node.chunk_node: list of chunk nodes (may contain duplicates)
    r: top-r to average
    """
    bucket = defaultdict(list)

    # collect sims per chunk
    for s, chunk in zip(sim, proto_node.chunk_node_list):
        bucket[chunk].append(float(s))

    # unique chunks + top-r mean
    new_chunks = []
    combined_sim = []
    for chunk, sims in bucket.items():
        sims.sort(reverse=True)
        top = sims[:r]
        combined_sim.append(sum(top) / len(top))
        new_chunks.append(chunk)

    proto_node.chunk_node_list = new_chunks
    return combined_sim


def inspect_prototypes(embed, prototype_list):
    B = torch.stack([prototype.embed for prototype in prototype_list])  # [N, D]
    A = embed.unsqueeze(0)  # [1, D]
    similarities = F.cosine_similarity(A, B, dim=1)
    max_val, max_idx = torch.max(similarities, dim=0)
    return max_val, max_idx


def plot_embeddings(embed, token, clusters):
    folder_name = time_stamp
    current_dir = os.getcwd()
    base_dir = os.path.join(current_dir, "model_embed_dis")
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    embeddings = torch.stack(embed)  # shape: (N, D)
    embeddings_np = embeddings.detach().cpu().numpy()

    # tsne = TSNE(n_components=2, random_state=42)
    # reduced = tsne.fit_transform(embeddings_np)

    pca = PCA(n_components=min(50, embeddings_np.shape[1]))
    reduced = pca.fit_transform(embeddings_np)

    N = len(embed)

    # 为每个点建立cluster标签
    labels = [-1] * N
    for cid, idx_list in clusters.items():
        for idx in idx_list:
            if idx < N:
                labels[idx] = cid

    unique_clusters = sorted(set(labels))

    plt.figure()

    # 颜色映射
    cmap = plt.cm.get_cmap("tab20", len(unique_clusters))

    for i, cid in enumerate(unique_clusters):
        indices = [j for j, l in enumerate(labels) if l == cid]
        if len(indices) == 0:
            continue

        points = reduced[indices]

        if cid == -1:
            plt.scatter(points[:, 0], points[:, 1], color="gray", label="unassigned")
        else:
            plt.scatter(points[:, 0], points[:, 1], color=cmap(i), label=f"cluster {cid}")

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"{token} PCA Visualization")

    plt.legend()
    plt.savefig(f"{folder_path}/{token}_PCA_embedding.png", dpi=300, bbox_inches="tight")
    plt.close()

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings_np)

    N = len(embed)

    # 为每个点建立cluster标签
    labels = [-1] * N
    for cid, idx_list in clusters.items():
        for idx in idx_list:
            if idx < N:
                labels[idx] = cid

    unique_clusters = sorted(set(labels))

    plt.figure()

    # 颜色映射
    cmap = plt.cm.get_cmap("tab20", len(unique_clusters))

    for i, cid in enumerate(unique_clusters):
        indices = [j for j, l in enumerate(labels) if l == cid]
        if len(indices) == 0:
            continue

        points = reduced[indices]

        if cid == -1:
            plt.scatter(points[:, 0], points[:, 1], color="gray", label="unassigned")
        else:
            plt.scatter(points[:, 0], points[:, 1], color=cmap(i), label=f"cluster {cid}")

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"{token} t-SNE Visualization")

    plt.legend()
    plt.savefig(f"{folder_path}/{token}_TSNE_embedding.png", dpi=300, bbox_inches="tight")
    plt.close()



def print_size_mb(obj, precision=2):
    size_bytes = asizeof.asizeof(obj)
    size_mb = size_bytes / (1024 * 1024)
    print(f"memory size: {size_mb:.{precision}f} MB")



def max_cosine_similarity_index(query_tensor, tensor_list):
    """
    query_tensor: shape (d,)
    tensor_list:  list of tensors, each shape (d,)
    return: index of tensor with max cosine similarity
    """

    if len(tensor_list) == 0:
        return None

    matrix = torch.stack(tensor_list)  # shape: (n, d)

    query_norm = F.normalize(query_tensor.unsqueeze(0), dim=1)  # (1, d)
    matrix_norm = F.normalize(matrix, dim=1)  # (n, d)

    similarities = torch.mm(query_norm, matrix_norm.t()).squeeze(0)  # (n,)

    max_index = torch.argmax(similarities).item()

    return max_index

import json
from pathlib import Path

def load_hotpot_distractor(file_path):
    """
    Load HotpotQA distractor setting.

    Returns:
        samples (list of dict)
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for item in data:
        sample = {}

        # 基本字段
        sample["id"] = item["_id"]
        sample["question"] = item["question"]
        sample["answer"] = item["answer"]

        # supporting facts
        # 格式: [["Page_Title", sentence_id], ...]
        supporting_pages = set()
        supporting_sentences = {}

        for title, sent_id in item["supporting_facts"]:
            supporting_pages.add(title)
            supporting_sentences.setdefault(title, []).append(sent_id)

        sample["supporting_pages"] = list(supporting_pages)
        sample["supporting_sentences"] = supporting_sentences

        # context (10篇文章)
        # 格式: [["Page_Title", ["sent1", "sent2", ...]], ...]
        documents = []

        for title, sentences in item["context"]:
            doc = {
                "title": title,
                "sentences": sentences,
                "text": " ".join(sentences)  # 拼成整篇文章
            }
            documents.append(doc)

        sample["documents"] = documents

        samples.append(sample)

    return samples

def build_global_document_list(file_path):
    """
    从 HotpotQA distractor 数据构建全局去重文档列表

    返回:
        documents: list[dict]
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_store = {}

    for item in data:
        for title, sentences in item["context"]:

            # 拼接成完整文本
            text = " ".join(sentences).strip()

            # 用 title 做唯一键（HotpotQA 基本可行）
            if title not in doc_store:
                doc_store[title] = {
                    "doc_id": title,
                    "title": title,
                    "text": text,
                    "sentences": sentences
                }
            else:
                # 如果出现重复 title 但文本更长，可以选择替换
                if len(text) > len(doc_store[title]["text"]):
                    doc_store[title]["text"] = text
                    doc_store[title]["sentences"] = sentences

    documents = list(doc_store.values())

    print(f"Total unique documents: {len(documents)}")

    return documents


import json
import pickle
from pathlib import Path


def build_hotpot_retrieval_dataset(file_path, num_samples=None):
    """
    构建 HotpotQA retrieval dataset，并带缓存机制

    返回:
        documents, samples
    """

    file_path = Path(file_path)

    # -----------------------------
    # cache 目录
    # -----------------------------

    cache_dir = Path("./hotpot_QA")
    cache_dir.mkdir(exist_ok=True)

    tag = "all" if num_samples is None else str(num_samples)

    documents_cache = cache_dir / f"hotpot_documents_{tag}.pkl"
    samples_cache = cache_dir / f"hotpot_samples_{tag}.pkl"

    # -----------------------------
    # 如果缓存存在，直接读取
    # -----------------------------

    if documents_cache.exists() and samples_cache.exists():
        print("Loading cached dataset...")

        with open(documents_cache, "rb") as f:
            documents = pickle.load(f)

        with open(samples_cache, "rb") as f:
            samples = pickle.load(f)

        print(f"Loaded {len(documents)} documents")
        print(f"Loaded {len(samples)} samples")

        return documents, samples

    # -----------------------------
    # 否则重新构建
    # -----------------------------

    print("Building dataset from raw file...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]

    # -----------------------------
    # Step 1: 构建全局去重文档库
    # -----------------------------

    title_to_doc = {}

    for item in data:
        for title, sentences in item["context"]:

            text = " ".join(sentences).strip()

            if title not in title_to_doc:
                title_to_doc[title] = {
                    "title": title,
                    "text": text,
                    "sentences": sentences
                }
            else:
                if len(text) > len(title_to_doc[title]["text"]):
                    title_to_doc[title]["text"] = text
                    title_to_doc[title]["sentences"] = sentences

    documents = []
    title_to_id = {}

    for idx, (title, doc) in enumerate(title_to_doc.items()):
        doc_entry = {
            "doc_id": idx,
            "title": title,
            "text": doc["text"],
            "sentences": doc["sentences"]
        }

        documents.append(doc_entry)
        title_to_id[title] = idx

    print(f"Total unique documents: {len(documents)}")

    # -----------------------------
    # Step 2: 构建 samples
    # -----------------------------

    samples = []

    for item in data:

        gold_titles = set([title for title, _ in item["supporting_facts"]])

        gold_doc_ids = []

        for title in gold_titles:
            if title in title_to_id:
                gold_doc_ids.append(title_to_id[title])

        sample_entry = {
            "sample_id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "gold_doc_ids": gold_doc_ids
        }

        samples.append(sample_entry)

    print(f"Total samples: {len(samples)}")

    # -----------------------------
    # 保存缓存
    # -----------------------------

    with open(documents_cache, "wb") as f:
        pickle.dump(documents, f)

    with open(samples_cache, "wb") as f:
        pickle.dump(samples, f)

    print(f"Dataset cached to: {cache_dir}")

    return documents, samples
