"""最小冒烟测试(对应审查报告整改清单第 8 项 / §2.2-F)。

本仓库是 run-as-script 的研究代码,没有 pytest 之类的测试框架,所以这里用一个
自包含脚本 + 纯 assert + 退出码自检的方式,覆盖三类"契约存在但缺机器防护"的不变量:

  1. 端到端:index_json -> finalize -> multi_level_query 在玩具语料上跑通,
     返回的 chunk 文本/id 数量一致、id 落在合法区间。
  2. 持久化往返:save_data_split -> load_data_split 后,结构字段逐项相等,
     且对同一 query 返回相同的 chunk id。
  3. 删除重建:delete_by_document 后,chunk_nodes 索引不变量
     (chunk_nodes[i].chunk_node_id == i)、sem 节点的 chunk 引用一致性、
     以及查询仍可用且 id 合法。

运行:
    /home/xiaoyue/anaconda3/envs/llm_graph/bin/python smoke_test.py

注意:构造 LiteSemRAG 会在 CUDA 上加载 DeBERTa 编码器与 cross-encoder(与本仓库
其余流程一致,假设有可用 GPU)。全程 no-LLM,不需要 API key。
"""

import os
import sys
import tempfile
import traceback

from RAG_graph import LiteSemRAG, CURRENT_SCHEMA_VERSION


# ----------------------------------------------------------------------------
# 玩具语料:10 篇短文档,词汇有意重叠(bank/apple/python 等多义词 + 共现实体),
# 让共现图能建出边、查询能命中,同时规模足够小,finalize 在秒级完成。
# ----------------------------------------------------------------------------
def make_corpus():
    return [
        {"title": "river_bank",
         "text": "The fisherman sat on the river bank and watched the water flow. "
                 "Reeds grew along the muddy bank near the slow river."},
        {"title": "money_bank",
         "text": "She walked into the bank to deposit her paycheck. "
                 "The bank approved the loan after reviewing her credit."},
        {"title": "apple_fruit",
         "text": "He picked a ripe red apple from the tree in the orchard. "
                 "The apple tasted sweet and crisp after the autumn harvest."},
        {"title": "apple_company",
         "text": "Apple released a new phone at its annual product event. "
                 "The Apple engineers designed a faster chip for the laptop."},
        {"title": "python_snake",
         "text": "The python coiled silently around the branch in the jungle. "
                 "A reticulated python can grow longer than most other snakes."},
        {"title": "python_language",
         "text": "She wrote the data pipeline in Python for the research project. "
                 "The Python script parsed the documents and built the index."},
        {"title": "mercury_planet",
         "text": "Mercury is the smallest planet and the closest one to the sun. "
                 "The surface of Mercury is covered with craters and bare rock."},
        {"title": "mercury_metal",
         "text": "Liquid mercury was once used inside old glass thermometers. "
                 "Exposure to mercury vapor is dangerous to human health."},
        {"title": "spring_season",
         "text": "In spring the garden filled with tulips and fresh green leaves. "
                 "Warm spring rain helped the new seeds grow quickly."},
        {"title": "spring_coil",
         "text": "The metal spring in the old clock had finally lost its tension. "
                 "He replaced the worn spring inside the mechanical watch."},
    ]


QUERY = "the fisherman watched the river"


def build_finalized():
    """构造实例、索引玩具语料并 finalize 一次,返回可查询实例。"""
    db = LiteSemRAG()
    db.index_json(make_corpus())
    db.finalize()
    return db


# ----------------------------------------------------------------------------
# 测试 1:端到端
# ----------------------------------------------------------------------------
def test_end_to_end(db):
    corpus = make_corpus()
    assert len(db.doc_nodes) == len(corpus), \
        f"doc 节点数 {len(db.doc_nodes)} != 语料数 {len(corpus)}"
    assert len(db.chunk_nodes) > 0, "没有索引出任何 chunk"
    assert len(db.sem_nodes) > 0, "finalize 后 sem 节点为空"

    texts, chunk_ids, graph = db.multi_level_query(QUERY, top_k_chunk=5)
    assert len(texts) == len(chunk_ids), \
        f"返回文本数 {len(texts)} != id 数 {len(chunk_ids)}"
    assert len(chunk_ids) > 0, "查询没有命中任何 chunk"
    assert len(chunk_ids) == len(set(chunk_ids)), "返回的 chunk id 有重复"
    for cid in chunk_ids:
        assert 0 <= cid < len(db.chunk_nodes), f"chunk id {cid} 越界"
    # 返回文本必须与 id 指向的 chunk 文本一致
    assert texts == [db.chunk_nodes[cid].chunk_text for cid in chunk_ids], \
        "返回文本与 chunk id 对应的文本不一致"
    print(f"  [1] 端到端 OK:{len(db.chunk_nodes)} chunks / "
          f"{len(db.sem_nodes)} sem 节点,query 命中 {len(chunk_ids)} chunks")


# ----------------------------------------------------------------------------
# 测试 2:持久化往返
# ----------------------------------------------------------------------------
def _structural_snapshot(db):
    """抽取一组可逐项比较的结构性字段(不含运行期句柄/张量)。"""
    return {
        "schema_version": getattr(db, "schema_version", None),
        "n_docs": len(db.doc_nodes),
        "n_chunks": len(db.chunk_nodes),
        "n_tokens": len(db.token_nodes),
        "n_sems": len(db.sem_nodes),
        "doc_names": sorted(d.doc_name for d in db.doc_nodes),
        "chunk_ids": [c.chunk_node_id for c in db.chunk_nodes],
        "chunk_texts": [c.chunk_text for c in db.chunk_nodes],
        "sem_ids": [s.sem_node_id for s in db.sem_nodes],
        "token_surfaces": sorted(t.token_text for t in db.token_nodes),
    }


def test_persistence_roundtrip(db):
    before = _structural_snapshot(db)
    before_query = db.multi_level_query(QUERY, top_k_chunk=5)[1]

    tmpdir = tempfile.mkdtemp(prefix="litesemrag_smoke_")
    pkl_path = os.path.join(tmpdir, "smoke.pkl")
    try:
        db.save_data_split(pkl_path)
        assert os.path.exists(pkl_path), "save_data_split 未写出 pkl"
        assert os.path.exists(pkl_path.replace(".pkl", "_tensors.pt")), \
            "save_data_split 未写出 _tensors.pt"

        loaded = LiteSemRAG.load_data_split(pkl_path)
        after = _structural_snapshot(loaded)

        for key in before:
            assert before[key] == after[key], \
                f"往返后字段 '{key}' 不一致:\n  before={before[key]!r}\n  after={after[key]!r}"

        assert after["schema_version"] == CURRENT_SCHEMA_VERSION, \
            f"schema 版本 {after['schema_version']} != 当前 {CURRENT_SCHEMA_VERSION}"

        after_query = loaded.multi_level_query(QUERY, top_k_chunk=5)[1]
        assert before_query == after_query, \
            f"往返后查询结果不一致:before={before_query} after={after_query}"

        loaded.shutdown()
    finally:
        for fn in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, fn))
        os.rmdir(tmpdir)
    print(f"  [2] 持久化往返 OK:{len(before)} 组结构字段逐项相等,查询结果一致")


# ----------------------------------------------------------------------------
# 测试 3:删除重建
# ----------------------------------------------------------------------------
def test_delete_rebuild(db):
    docs_before = sorted(d.doc_name for d in db.doc_nodes)
    chunks_before = len(db.chunk_nodes)
    to_delete = ["apple_company", "spring_coil"]

    db.delete_by_document(to_delete)

    docs_after = sorted(d.doc_name for d in db.doc_nodes)
    assert docs_after == sorted(set(docs_before) - set(to_delete)), \
        f"删除后文档集合错误:{docs_after}"
    assert len(db.chunk_nodes) < chunks_before, "删除后 chunk 数没有减少"

    # chunk_nodes 索引不变量:位置 i 的节点 id 必须为 i(chunk_id2text 依赖此点)
    for i, c in enumerate(db.chunk_nodes):
        assert c.chunk_node_id == i, \
            f"chunk_nodes 索引不变量被破坏:位置 {i} 的 id 是 {c.chunk_node_id}"
    assert db.next_chunk_node_id == len(db.chunk_nodes), \
        "next_chunk_node_id 与 chunk_nodes 长度不一致"

    # chunk_id2text 对全量 id 不应抛异常(它内部会校验上述不变量)
    all_ids = list(range(len(db.chunk_nodes)))
    db.chunk_id2text(all_ids)

    # sem 节点的 chunk 引用必须都指向仍存在的 chunk 节点
    live_chunks = set(id(c) for c in db.chunk_nodes)
    for s in db.sem_nodes:
        for c in s.chunk_node_list:
            assert id(c) in live_chunks, \
                f"sem 节点 {s.sem_node_id} 仍引用已删除的 chunk"

    # 删除后查询仍可用,且 id 合法
    texts, chunk_ids, _ = db.multi_level_query(QUERY, top_k_chunk=5)
    assert len(texts) == len(chunk_ids)
    for cid in chunk_ids:
        assert 0 <= cid < len(db.chunk_nodes), f"删除后查询返回越界 id {cid}"
    print(f"  [3] 删除重建 OK:删 {len(to_delete)} 文档后 "
          f"{len(db.chunk_nodes)} chunks,索引不变量 + 引用一致性通过")


# ----------------------------------------------------------------------------
# 运行器
# ----------------------------------------------------------------------------
def main():
    failures = []

    # finalize 必须每实例只调一次,所以删除测试用独立实例,避免污染前两个测试。
    print("[build] 索引玩具语料并 finalize(共享实例,供测试 1/2)...")
    shared = build_finalized()
    for name, fn in [("end_to_end", test_end_to_end),
                     ("persistence_roundtrip", test_persistence_roundtrip)]:
        try:
            fn(shared)
        except Exception:
            failures.append(name)
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    shared.shutdown()

    print("[build] 为删除测试构造独立实例...")
    try:
        delete_db = build_finalized()
        try:
            test_delete_rebuild(delete_db)
        finally:
            delete_db.shutdown()
    except Exception:
        failures.append("delete_rebuild")
        print("  [FAIL] delete_rebuild")
        traceback.print_exc()

    print("-" * 60)
    if failures:
        print(f"冒烟测试失败:{', '.join(failures)}")
        return 1
    print("全部冒烟测试通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
