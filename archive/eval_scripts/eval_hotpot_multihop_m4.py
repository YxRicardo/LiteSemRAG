"""Milestone 4 ablation: sentence-level evidence graph on HotpotQA (distractor 500).

Loads the existing fastidx cache (no rebuild), builds the global co-occurrence
index (M2) and the sentence-level evidence index (M4), then compares query-time
methods with sentence evidence OFF vs ON to isolate the M4 effect:

  baseline    : chunk_cooccur_query()                       - single-hop baseline
  path_off    : multihop_path_query(use_sentence_evidence=False)
  path_on     : multihop_path_query(use_sentence_evidence=True)   (M4)
  routed_off  : multihop_query(use_sentence_evidence=False)
  routed_on   : multihop_query(use_sentence_evidence=True)        (M4)

Reports full_hit / recall@10_pq / recall@10_micro / mrr@10 for all / bridge /
comparison slices, plus avg query latency. Results -> JSON for the report.
"""
from __future__ import annotations

import os
os.environ.setdefault("LITESEM_PRESERVE_CASE", "1")

import json
import time
from pathlib import Path

import RAG_graph
from utils import build_hotpot_retrieval_dataset, mrr_for_one_query_titles

NUM_SAMPLES = 500
TOP_K = 10
HOTPOT_FILE = "jupyter_notebooks/hotpot_dev_distractor_v1.json"
PKL = "cache/hotpotqa_latest_framework_index/litesemrag_hotpotqa_500_fastidx.pkl"
OUT = Path("reports/multihop_m4_eval_results.json")


def log(msg):
    print(msg, flush=True)


def load_type_map(file_path, num_samples):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    if num_samples is not None:
        data = data[:num_samples]
    return {item["_id"]: item.get("type", "unknown") for item in data}


def titles_from_chunk_ids(db, chunk_ids, k=TOP_K):
    titles, seen = [], set()
    for cid in chunk_ids:
        t = db.chunk_nodes[cid].doc_node.doc_name
        if t not in seen:
            seen.add(t)
            titles.append(t)
    return titles[:k]


def baseline_titles(db, question):
    _, chunk_ids, _ = db.chunk_cooccur_query(
        question, top_k_chunk=TOP_K, print_important_tokens=False
    )
    return titles_from_chunk_ids(db, chunk_ids)


def make_path_fn(use_sentence_evidence):
    def fn(db, question):
        _, chunk_ids, _ = db.multihop_path_query(
            question, top_k_chain=TOP_K, first_hop_k=20, bridge_top_k=8,
            second_hop_k=20, use_sentence_evidence=use_sentence_evidence,
        )
        return titles_from_chunk_ids(db, chunk_ids)
    return fn


def make_routed_fn(use_sentence_evidence):
    def fn(db, question):
        _, chunk_ids, _ = db.multihop_query(
            question, top_k_chain=TOP_K, first_hop_k=20, bridge_top_k=8,
            second_hop_k=20, use_sentence_evidence=use_sentence_evidence,
        )
        return titles_from_chunk_ids(db, chunk_ids)
    return fn


def evaluate(db, method_fn, samples, gold_sets, gold_titles_list, type_list):
    agg = {}

    def bump(bucket, recall_pq, n_hit, n_gold, rr, is_full):
        s = agg.setdefault(bucket, [0.0, 0, 0, 0.0, 0, 0])
        s[0] += recall_pq; s[1] += n_hit; s[2] += n_gold
        s[3] += rr; s[4] += int(is_full); s[5] += 1

    start = time.time()
    for sample, gold_set, gold_titles, qtype in zip(
        samples, gold_sets, gold_titles_list, type_list
    ):
        if not gold_set:
            continue
        ranked = method_fn(db, sample["question"])
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
            "full_hit_rate": round(full / n, 4),
            "recall@10_pq": round(rpq / n, 4),
            "recall@10_micro": round(hit / gold, 4),
            "mrr@10": round(mrr / n, 4),
        }
    n_all = agg.get("all", [0, 0, 0, 0, 0, 0])[5]
    out["sec"] = round(elapsed, 1)
    out["avg_query_ms"] = round(1000.0 * elapsed / n_all, 2) if n_all else None
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

    log(f"loading {PKL} ...")
    t0 = time.time()
    db = RAG_graph.LiteSemRAG.load_data(PKL)
    log(f"loaded in {time.time()-t0:.1f}s: docs={len(db.doc_nodes)} chunks={len(db.chunk_nodes)} "
        f"sems={len(db.sem_nodes)} fast_index={getattr(db,'fast_index',None)}")

    t0 = time.time()
    db.build_sem_cooccur_index()
    log(f"sem_cooccur_index built in {time.time()-t0:.1f}s: {db._sem_cooccur_index_params}")

    t0 = time.time()
    db.build_sentence_evidence_index()
    log(f"sentence_evidence_index built in {time.time()-t0:.1f}s: {db._sentence_evidence_params}")

    results = {
        "index_params": db._sem_cooccur_index_params,
        "sentence_evidence_params": db._sentence_evidence_params,
    }
    methods = [
        ("baseline", baseline_titles),
        ("path_off", make_path_fn(False)),
        ("path_on", make_path_fn(True)),
        ("routed_off", make_routed_fn(False)),
        ("routed_on", make_routed_fn(True)),
    ]
    for name, fn in methods:
        res = evaluate(db, fn, samples, gold_sets, gold_titles_list, type_list)
        results[name] = res
        log(f"\n=== {name} ===")
        log(json.dumps(res, ensure_ascii=False, indent=2))

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log(f"\nwrote {OUT}")
    try:
        db.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
