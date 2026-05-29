# archive/ — 废弃实验归档

本目录存放与当前 `LiteSemRAG` 主线（基于 Wikidata 候选库 + 语义节点 +
cross-encoder/LLM 标注）已脱钩的历史脚本与 notebook。归档而非删除，便于追溯。

- 归档日期：2026-05-29
- 归档基准提交：`2ddf77a`（归档前所有内容均已同步到 GitHub）
- 这些文件**不被 `RAG_graph.py` 运行时引用**，迁移不影响核心检索引擎。
- 用 `git mv` 迁移，历史可用 `git log --follow <file>` 查看。

> 注意：归档后的 notebook 若仍 `from wikidata_utils import ...` 等引用根目录模块，
> 在本目录原地运行会因工作目录变化而找不到模块。它们仅作参考保留，不保证可直接运行。

## wikipedia_redirects/ — Wikipedia 同义词/重定向索引实验线

早期尝试用 Wikipedia 重定向构建同义词索引，当前主线未采用。

- `wikipedia_redirects.py` — 重定向索引构建与查询
- `prepare_wikipedia_redirects.py` — 重定向数据准备脚本
- `inspect_redirect_index.py` — 索引检视脚本
- `wikipedia_redirect_index_demo.ipynb` — 演示 notebook

## wordnet_semcor/ — WordNet / SemCor 语义消歧探索线

早期基于 WordNet/SemCor 的词义消歧探索，已被 Wikidata 候选库方案取代。

- `wordnet_explorer.py`、`prepare_semcor.py` — 数据导出/探索脚本
- `explore_wordnet.ipynb`、`semcor_demo.ipynb`、`semcor_cross_encoder.ipynb`、
  `semcor_embeds_explore.ipynb`、`semcor_embeds_qwen.ipynb` — 实验 notebook

## hotpotqa_legacy/ — HotpotQA 旧实验 notebook

均为 2026-05-21 及更早版本，已被
`jupyter_notebooks/hotpotqa_latest_framework_index.ipynb`（当前主索引入口）取代。

- `HotpotQA.ipynb`、`farthest_first_traversal_test.ipynb`、`hotpotQA_cluster.ipynb`
- `hotpotqa_embeds_explore.ipynb`、`hotpotqa_embeds_qwen.ipynb`
- `hotpotqa_cross_encoder.ipynb`、`hotpotqa_llm_model_compare.ipynb`
- `hotpotqa_scan_cache_explore.ipynb`
