"""A/B evaluation of query-side method improvements on HotpotQA (distractor 500),
with indexing and evaluation in the same process.

The fast_index graph is built in memory and evaluated directly without writing a
pickle to disk, which avoids stack overflows when pickling large objects.
Only the query-side method is changed:
  A0 baseline_off : subtoken fallback off + IDF pruning off (roughly the
                    "before improvement" setup in the report)
  A1 subtoken     : subtoken fallback on (current default) + IDF pruning off
  A2 subtoken_idf : subtoken fallback on + generic-term IDF pruning
                    tau/floor/keep_top (recommended FEVER setting)

PRESERVE_CASE is enabled by default (Option A). Metrics are
Recall@10(pq/micro)/MRR@10/full_hit, with separate slices for comparison and
bridge questions to check whether single-hop improvements hurt multi-hop
behavior.
"""
from __future__ import annotations

import os
os.environ.setdefault("LITESEM_PRESERVE_CASE", "1")

import json
import sys
import time
from pathlib import Path

import RAG_graph
from RAG_graph import COOCCUR_LAMBDA, COOCCUR_TOP_K_PAIR_BOOSTS
from utils import build_hotpot_retrieval_dataset, mrr_for_one_query_titles

NUM_SAMPLES = 500
TOP_K = 10
CHUNK_SIZE = 256
HOTPOT_FILE = "jupyter_notebooks/hotpot_dev_distractor_v1.json"
OUT = Path("/tmp/hotpot_litesem_ab.json")

IDF_TAU = 0.70
IDF_FLOOR = 6.5
IDF_KEEP_TOP = 2

CONFIGS = [
    ("A0_baseline_off", False, None),
    ("A1_subtoken",     True,  None),
    ("A2_subtoken_idf", True,  IDF_TAU),
]


def log(msg):
    print(msg, flush=True)


def load_type_map(file_path, num_samples):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    if num_samples is not None:
        data = data[:num_samples]
    return {item["_id"]: item.get("type", "unknown") for item in data}


def retrieved_titles(db, question, idf_prune_tau):
    _, chunk_ids, _ = db.chunk_cooccur_query(
        question,
        top_k_chunk=TOP_K,
        top_k_pair_boosts=COOCCUR_TOP_K_PAIR_BOOSTS,
        lambda_boost=COOCCUR_LAMBDA,
        print_important_tokens=False,
        idf_prune_tau=idf_prune_tau,
        idf_prune_min_max_idf=(IDF_FLOOR if idf_prune_tau is not None else None),
        idf_prune_keep_top=IDF_KEEP_TOP,
    )
    titles, seen = [], set()
    for cid in chunk_ids:
        t = db.chunk_nodes[cid].doc_node.doc_name
        if t not in seen:
            seen.add(t)
            titles.append(t)
    return titles[:TOP_K]


def evaluate(db, samples, gold_sets, gold_titles_list, type_list, idf_prune_tau):
    agg = {}

    def bump(bucket, recall_pq, n_hit, n_gold, rr, is_full):
        s = agg.setdefault(bucket, [0.0, 0, 0, 0.0, 0, 0])
        s[0] += recall_pq; s[1] += n_hit; s[2] += n_gold
        s[3] += rr; s[4] += int(is_full); s[5] += 1

    start = time.time()
    for sample, gold_set, gold_titles, qtype in zip(samples, gold_sets, gold_titles_list, type_list):
        if not gold_set:
            continue
        ranked = retrieved_titles(db, sample["question"], idf_prune_tau)
        rset = {t.strip().lower() for t in ranked}
        n_hit = len(gold_set & rset)
        n_gold = len(gold_set)
        recall_pq = n_hit / n_gold
        is_full = gold_set <= rset
        rr = mrr_for_one_query_titles(ranked, gold_titles, k=TOP_K)
        bump("all", recall_pq, n_hit, n_gold, rr, is_full)
        bump(qtype, recall_pq, n_hit, n_gold, rr, is_full)
    elapsed = time.time() - start

    out = {}
    for bucket, (rpq, hit, gold, mrr, full, n) in agg.items():
        out[bucket] = {
            "n": n,
            "full_hit": full,
            "recall@10_pq": round(rpq / n, 4),
            "recall@10_micro": round(hit / gold, 4),
            "mrr@10": round(mrr / n, 4),
        }
    out["sec"] = round(elapsed, 1)
    return out


def main():
    documents, samples = build_hotpot_retrieval_dataset(HOTPOT_FILE, num_samples=NUM_SAMPLES)
    type_map = load_type_map(HOTPOT_FILE, NUM_SAMPLES)
    log(f"documents={len(documents)} samples={len(samples)}")

    gold_sets, gold_titles_list, type_list = [], [], []
    for s in samples:
        titles = [documents[i]["title"] for i in s["gold_doc_ids"]]
        gold_titles_list.append(titles)
        gold_sets.append({t.strip().lower() for t in titles if t and t.strip()})
        type_list.append(type_map.get(s["sample_id"], "unknown"))

    # Build the index in memory.
    db = RAG_graph.LiteSemRAG(chunk_size=CHUNK_SIZE, device="cuda", fast_index=True)
    db.json_path = "/tmp/hotpot_fastidx_documents.json"
    t0 = time.time()
    db.index_json(documents, batch_size=64, queue_size=16, sample_count=len(samples))
    log(f"index_json done in {time.time()-t0:.1f}s")
    t1 = time.time()
    db.finalize()
    log(f"finalize done in {time.time()-t1:.1f}s")
    log(f"graph: docs={len(db.doc_nodes)} chunks={len(db.chunk_nodes)} "
        f"tokens={len(db.token_nodes)} sem_nodes={len(db.sem_nodes)} fast_index={db.fast_index}")

    orig_subtoken = RAG_graph.LiteSemRAG._build_subtoken_fallback_units
    results = {}
    for name, subtoken_on, idf_tau in CONFIGS:
        if subtoken_on:
            RAG_graph.LiteSemRAG._build_subtoken_fallback_units = orig_subtoken
        else:
            RAG_graph.LiteSemRAG._build_subtoken_fallback_units = lambda *a, **k: []
        res = evaluate(db, samples, gold_sets, gold_titles_list, type_list, idf_tau)
        results[name] = res
        log(f"\n=== {name} (subtoken={subtoken_on}, idf_tau={idf_tau}) ===")
        log(json.dumps(res, ensure_ascii=False, indent=2))

    RAG_graph.LiteSemRAG._build_subtoken_fallback_units = orig_subtoken
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log(f"\nwrote {OUT}")
    try:
        db.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
