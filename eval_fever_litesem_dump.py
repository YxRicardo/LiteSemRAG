"""Run LiteSemRAG (caseA index + sub-token fallback) on FEVER, dump per-qid results.

Mirrors jupyter_notebooks/fever_litesemrag_index_eval.ipynb eval path.
Join key with NoLLMRAG = corpus_id (BEIR _id). qid = BEIR query id.
"""
from __future__ import annotations

import os
os.environ.setdefault("LITESEM_PRESERVE_CASE", "1")  # caseA index is cased

import csv
import json
import time
from pathlib import Path

import RAG_graph
from RAG_graph import COOCCUR_LAMBDA, COOCCUR_TOP_K_PAIR_BOOSTS

FEVER_DIR = Path("/home/xiaoyue/ProtoGraphRAG/fever_subset")
QRELS_SPLIT = "test.tsv"
INDEX_PKL = Path("cache/fever_litesemrag_index/litesemrag_fever_all_caseA.pkl")
INDEX_DOCS_JSON = Path("cache/fever_litesemrag_index/litesemrag_fever_all_caseA_documents.json")
TOP_K = 10
INDEX_INCLUDE_TITLE = True

# --- IDF generic-word pruning (NoLLMRAG IS-threshold borrow), A/B via env ---
# LITESEM_IDF_PRUNE_TAU unset/empty -> pruning OFF (baseline, identical to before).
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
OUT = Path(f"/tmp/litesem_fever_results_{_tag}.json")


def load_corpus(path):
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[str(obj["_id"])] = {"title": obj.get("title", ""), "text": obj.get("text", "")}
    return corpus


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[str(obj["_id"])] = obj["text"]
    return queries


def load_qrels(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, docid, score = row[0], row[1], row[2]
            try:
                if int(score) <= 0:
                    continue
            except ValueError:
                pass
            gold.setdefault(str(qid), set()).add(str(docid))
    return gold


def main():
    corpus = load_corpus(FEVER_DIR / "corpus.jsonl")
    queries = load_queries(FEVER_DIR / "queries.jsonl")
    gold_docs = load_qrels(FEVER_DIR / "qrels" / QRELS_SPLIT)
    indexed_doc_ids = set(corpus.keys())

    eval_items = []
    for qid, text in queries.items():
        g = gold_docs.get(qid, set())
        if not g:
            continue
        if not g <= indexed_doc_ids:
            continue
        eval_items.append((qid, text, g))
    print(f"eval_items: {len(eval_items)}")

    db = RAG_graph.LiteSemRAG.load_data(str(INDEX_PKL))
    db.json_path = str(INDEX_DOCS_JSON)

    from utils import mrr_for_one_query_titles

    results = {}
    full_hit = 0
    recall_sum = 0.0
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
        ranked = []
        seen = set()
        for cid in chunk_ids:
            docid = db.chunk_nodes[cid].doc_node.doc_name
            if docid not in seen:
                seen.add(docid)
                ranked.append(docid)
        ranked = ranked[:TOP_K]
        rset = set(ranked)
        n_hit = len(gold & rset)
        recall_sum += n_hit / len(gold)
        is_full = gold <= rset
        full_hit += int(is_full)
        rr = mrr_for_one_query_titles(ranked, list(gold), k=TOP_K)
        mrr_sum += rr
        results[qid] = {
            "claim": qtext,
            "gold": sorted(gold),
            "ranked": ranked,
            "n_gold": len(gold),
            "n_hit": n_hit,
            "full_hit": is_full,
            "rr": rr,
        }
    elapsed = time.perf_counter() - t0

    n = len(results)
    summary = {
        "config": _tag,
        "idf_prune_tau": IDF_PRUNE_TAU,
        "idf_prune_min_max_idf": IDF_PRUNE_MIN_MAX_IDF,
        "idf_prune_keep_top": IDF_PRUNE_KEEP_TOP,
        "n": n,
        "full_hit": full_hit,
        "fail": n - full_hit,
        "recall@10_per_query": recall_sum / n,
        "mrr@10": mrr_sum / n,
        "sec": elapsed,
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
