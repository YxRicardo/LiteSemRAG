"""Run LiteSemRAG on the SciFact 1000-doc subset, dump per-qid results.

Mirrors jupyter_notebooks/scifact_litesemrag_index_eval.ipynb:
  - 1000-doc subset index, gold from queries.jsonl metadata.
  - eval only queries with gold ⊆ indexed subset (=> 142 queries).
  - chunk_cooccur_query default weights, TOP_K=10. No PRESERVE_CASE override.
Join key with NoLLMRAG = BEIR corpus_id; qid = BEIR query id.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path

import RAG_graph
from RAG_graph import COOCCUR_LAMBDA, COOCCUR_TOP_K_PAIR_BOOSTS

SCIFACT_DIR = Path("/home/xiaoyue/ProtoGraphRAG/datasets/scifact")
INDEX_PKL = Path("cache/scifact_litesemrag_index/litesemrag_scifact_1000_api.pkl")
INDEX_DOCS_JSON = Path("cache/scifact_litesemrag_index/litesemrag_scifact_1000_api_documents.json")
TOP_K = 10


# --- IDF generic-word pruning (NoLLMRAG IS-threshold borrow), A/B via env ---
def _env_float(name):
    v = os.environ.get(name, "").strip()
    return float(v) if v else None


IDF_PRUNE_TAU = _env_float("LITESEM_IDF_PRUNE_TAU")
IDF_PRUNE_MIN_MAX_IDF = _env_float("LITESEM_IDF_PRUNE_MIN_MAX_IDF")
IDF_PRUNE_KEEP_TOP = int(os.environ.get("LITESEM_IDF_PRUNE_KEEP_TOP", "2"))

_tag = "baseline" if IDF_PRUNE_TAU is None else (
    f"prune_tau{IDF_PRUNE_TAU:g}"
    + (f"_floor{IDF_PRUNE_MIN_MAX_IDF:g}" if IDF_PRUNE_MIN_MAX_IDF is not None else "")
    + f"_keep{IDF_PRUNE_KEEP_TOP}"
)
OUT = Path(f"/tmp/litesem_scifact_results_{_tag}.json")


def load_queries_with_metadata(path):
    queries, gold = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["_id"])
            queries[qid] = obj["text"]
            meta = obj.get("metadata", {}) or {}
            gold[qid] = {str(k) for k in meta.keys()}
    return queries, gold


def main():
    subset_ids = set(str(x) for x in json.loads(INDEX_DOCS_JSON.read_text()))
    queries, gold_docs = load_queries_with_metadata(SCIFACT_DIR / "queries.jsonl")

    eval_items = []
    for qid, text in queries.items():
        g = gold_docs.get(qid, set())
        if not g or not g <= subset_ids:
            continue
        eval_items.append((qid, text, g))
    print(f"eval_items: {len(eval_items)}")

    try:
        db = RAG_graph.LiteSemRAG.load_data_split(str(INDEX_PKL))
    except Exception:
        db = RAG_graph.LiteSemRAG.load_data(str(INDEX_PKL))
    db.json_path = str(INDEX_DOCS_JSON)

    from utils import mrr_for_one_query_titles

    results = {}
    full_hit = 0
    recall_pq_sum = 0.0
    n_hit_total = 0
    n_gold_total = 0
    mrr_sum = 0.0
    t0 = time.perf_counter()
    for qid, qtext, gold in eval_items:
        _, chunk_ids, _ = db.chunk_cooccur_query(
            qtext, top_k_chunk=TOP_K,
            top_k_pair_boosts=COOCCUR_TOP_K_PAIR_BOOSTS,
            lambda_boost=COOCCUR_LAMBDA,
            print_important_tokens=False,
            idf_prune_tau=IDF_PRUNE_TAU,
            idf_prune_min_max_idf=IDF_PRUNE_MIN_MAX_IDF,
            idf_prune_keep_top=IDF_PRUNE_KEEP_TOP,
        )
        ranked, seen = [], set()
        for cid in chunk_ids:
            docid = db.chunk_nodes[cid].doc_node.doc_name
            if docid not in seen:
                seen.add(docid)
                ranked.append(docid)
        ranked = ranked[:TOP_K]
        rset = set(ranked)
        n_hit = len(gold & rset)
        recall_pq_sum += n_hit / len(gold)
        n_hit_total += n_hit
        n_gold_total += len(gold)
        is_full = gold <= rset
        full_hit += int(is_full)
        rr = mrr_for_one_query_titles(ranked, list(gold), k=TOP_K)
        mrr_sum += rr
        results[qid] = {
            "claim": qtext, "gold": sorted(gold), "ranked": ranked,
            "n_gold": len(gold), "n_hit": n_hit, "full_hit": is_full, "rr": rr,
        }
    elapsed = time.perf_counter() - t0

    n = len(results)
    summary = {
        "config": _tag,
        "idf_prune_tau": IDF_PRUNE_TAU,
        "idf_prune_min_max_idf": IDF_PRUNE_MIN_MAX_IDF,
        "idf_prune_keep_top": IDF_PRUNE_KEEP_TOP,
        "n": n, "full_hit": full_hit, "fail": n - full_hit,
        "recall@10_per_query": recall_pq_sum / n,
        "recall@10_micro": n_hit_total / n_gold_total,
        "mrr@10": mrr_sum / n, "sec": elapsed,
    }
    OUT.write_text(json.dumps({"summary": summary, "per_qid": results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    try:
        db.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
