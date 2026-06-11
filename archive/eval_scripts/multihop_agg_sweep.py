"""聚合权重小扫描:在保住 bridge full_hit 的同时尽量收回 MRR。

加载 db 一次,跑 baseline + 若干 multihop 聚合配置,打印 all/bridge/comparison 的
full_hit / recall@10_pq / mrr@10。
"""
from __future__ import annotations
import os
os.environ.setdefault("LITESEM_PRESERVE_CASE", "1")
import json, time
import RAG_graph
from utils import build_hotpot_retrieval_dataset, mrr_for_one_query_titles

NUM_SAMPLES = 500
TOP_K = 10
HOTPOT_FILE = "jupyter_notebooks/hotpot_dev_distractor_v1.json"
PKL = "cache/hotpotqa_latest_framework_index/litesemrag_hotpotqa_500_fastidx.pkl"

# (name, hop1_w, hop2_w, min_chain_bonus_score)
CONFIGS = [
    ("h1=1.0,h2=1.0,min=0.0", 1.0, 1.0, 0.0),   # 原始(全 re-sort)
    ("h1=0.5,h2=1.0,min=0.0", 0.5, 1.0, 0.0),   # 弱 hop1
    ("h1=0.5,h2=1.0,min=0.1", 0.5, 1.0, 0.1),   # 弱 hop1 + 门槛
    ("h1=0.3,h2=1.0,min=0.1", 0.3, 1.0, 0.1),
    ("h1=0.5,h2=1.0,min=0.2", 0.5, 1.0, 0.2),
]


def titles_from_chunk_ids(db, chunk_ids, k=TOP_K):
    titles, seen = [], set()
    for cid in chunk_ids:
        t = db.chunk_nodes[cid].doc_node.doc_name
        if t not in seen:
            seen.add(t); titles.append(t)
    return titles[:k]


def evaluate(db, fn, samples, gold_sets, gold_titles_list, type_list):
    agg = {}
    def bump(bucket, rpq, rr, full):
        s = agg.setdefault(bucket, [0.0, 0.0, 0, 0])
        s[0] += rpq; s[1] += rr; s[2] += int(full); s[3] += 1
    for sample, gold_set, gold_titles, qtype in zip(samples, gold_sets, gold_titles_list, type_list):
        if not gold_set: continue
        ranked = fn(db, sample["question"])
        rset = {t.strip().lower() for t in ranked}
        rpq = len(gold_set & rset) / len(gold_set)
        full = gold_set <= rset
        rr = mrr_for_one_query_titles(ranked, gold_titles, k=TOP_K)
        bump("all", rpq, rr, full); bump(qtype, rpq, rr, full)
    out = {}
    for b, (rpq, rr, full, n) in agg.items():
        out[b] = {"n": n, "full": full, "fhr": round(full/n,4),
                  "r@10": round(rpq/n,4), "mrr": round(rr/n,4)}
    return out


def fmt(res):
    parts = []
    for b in ("all", "bridge", "comparison"):
        if b in res:
            r = res[b]
            parts.append(f"{b}: full={r['full']}/{r['n']}({r['fhr']}) r@10={r['r@10']} mrr={r['mrr']}")
    return "\n    " + "\n    ".join(parts)


def main():
    documents, samples = build_hotpot_retrieval_dataset(HOTPOT_FILE, num_samples=NUM_SAMPLES)
    with open(HOTPOT_FILE, encoding="utf-8") as f:
        type_map = {it["_id"]: it.get("type","unknown") for it in json.load(f)[:NUM_SAMPLES]}
    gold_sets, gold_titles_list, type_list = [], [], []
    for s in samples:
        titles = [documents[i]["title"] for i in s["gold_doc_ids"]]
        gold_titles_list.append(titles)
        gold_sets.append({t.strip().lower() for t in titles if t and t.strip()})
        type_list.append(type_map.get(s["sample_id"], "unknown"))

    db = RAG_graph.LiteSemRAG.load_data(PKL)
    print(f"loaded: chunks={len(db.chunk_nodes)}")

    def baseline_fn(db, q):
        _, cids, _ = db.chunk_cooccur_query(q, top_k_chunk=TOP_K, print_important_tokens=False)
        return titles_from_chunk_ids(db, cids)

    t0=time.time()
    print("\n[baseline]" + fmt(evaluate(db, baseline_fn, samples, gold_sets, gold_titles_list, type_list)) + f"\n  ({time.time()-t0:.0f}s)")

    for name, h1, h2, mn in CONFIGS:
        def mh_fn(db, q, h1=h1, h2=h2, mn=mn):
            _, cids, _ = db.multihop_bridge_query(
                q, top_k_chain=TOP_K, first_hop_k=20, bridge_top_k=8, second_hop_k=20,
                hop1_bonus_weight=h1, hop2_bonus_weight=h2, min_chain_bonus_score=mn,
                print_important_tokens=False)
            return titles_from_chunk_ids(db, cids)
        t0=time.time()
        print(f"\n[multihop {name}]" + fmt(evaluate(db, mh_fn, samples, gold_sets, gold_titles_list, type_list)) + f"\n  ({time.time()-t0:.0f}s)")

    db.shutdown()


if __name__ == "__main__":
    main()
