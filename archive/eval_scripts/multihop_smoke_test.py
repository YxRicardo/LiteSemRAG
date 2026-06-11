"""Validate four smoke-test cases for Plan-1 multihop_bridge_query (plan §11.1).

Load an existing fastidx cache and check:
  - whether the bridge is extracted
  - whether both gold titles appear in the returned chunks/docs
  - comparison-case behavior
"""
import os
os.environ.setdefault("LITESEM_PRESERVE_CASE", "1")

import RAG_graph

PKL = "cache/hotpotqa_latest_framework_index/litesemrag_hotpotqa_500_fastidx.pkl"

CASES = [
    {
        "name": "Kiss and Tell -> Shirley Temple (bridge)",
        "q": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "gold": ["Kiss and Tell (1945 film)", "Shirley Temple"],
        "want_bridge": "shirley temple",
    },
    {
        "name": "Lewiston Maineiacs -> Androscoggin Bank Colisee (bridge)",
        "q": "The arena where the Lewiston Maineiacs played their home games can seat how many people?",
        "gold": ["Lewiston Maineiacs", "Androscoggin Bank Colisée"],
        "want_bridge": "androscoggin bank colisée",
    },
    {
        "name": "University of Kansas -> Kansas Song (bridge)",
        "q": "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?",
        "gold": ["University of Kansas", "Kansas Song"],
        "want_bridge": "kansas song",
    },
    {
        "name": "Scott Derrickson vs Ed Wood (comparison)",
        "q": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "gold": ["Scott Derrickson", "Ed Wood"],
        "want_bridge": None,
    },
]


def titles_from_chunk_ids(db, chunk_ids, k=10):
    titles, seen = [], set()
    for cid in chunk_ids:
        t = db.chunk_nodes[cid].doc_node.doc_name
        if t not in seen:
            seen.add(t)
            titles.append(t)
        if len(titles) >= k:
            break
    return titles


def main():
    print(f"loading {PKL} ...")
    db = RAG_graph.LiteSemRAG.load_data(PKL)
    print(f"loaded: docs={len(db.doc_nodes)} chunks={len(db.chunk_nodes)} "
          f"sems={len(db.sem_nodes)} fast_index={getattr(db,'fast_index',None)}\n")

    for case in CASES:
        print("=" * 100)
        print(f"CASE: {case['name']}")
        print(f"Q: {case['q']}")
        gold = case["gold"]
        print(f"gold titles: {gold}")

        # baseline
        _, base_cids, _ = db.chunk_cooccur_query(case["q"], top_k_chunk=10, print_important_tokens=False)
        base_titles = titles_from_chunk_ids(db, base_cids, 10)
        base_hit = [g for g in gold if g in base_titles]
        print(f"\n[baseline chunk_cooccur] top10 titles: {base_titles}")
        print(f"[baseline] gold hit: {len(base_hit)}/{len(gold)} -> {base_hit}")

        # multihop
        chains, mh_cids, dbg = db.multihop_bridge_query(case["q"], top_k_chain=10)
        mh_titles = titles_from_chunk_ids(db, mh_cids, 10)
        mh_hit = [g for g in gold if g in mh_titles]
        print(f"\n[multihop_bridge] top10 titles: {mh_titles}")
        print(f"[multihop] gold hit: {len(mh_hit)}/{len(gold)} -> {mh_hit}")

        bridges = [b["bridge"] for b in dbg["bridge_candidates"]]
        print(f"[multihop] bridge candidates: {bridges}")
        if case["want_bridge"]:
            found = case["want_bridge"] in bridges
            print(f"[multihop] expected bridge '{case['want_bridge']}' extracted: {found}")
        print(f"[multihop] top chains:")
        for c in dbg["chains"][:5]:
            print(f"   score={c['score']:.5f} bridge='{c['bridge_text']}' "
                  f"hop1='{c['hop1_doc']}' -> hop2='{c['hop2_doc']}' "
                  f"(fhs={c['first_hop_score']:.3f} bp={c['bridge_prior']:.3f} "
                  f"lbe={c['local_bridge_evidence']:.2f} shs={c['second_hop_score']:.3f} "
                  f"cc={c['constraint_coverage']:.3f})")
        print()

    db.shutdown()


if __name__ == "__main__":
    main()
