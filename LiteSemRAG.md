# LiteSemRAG

`LiteSemRAG` is defined in `RAG_graph.py`. It builds a semantic-aware retrieval
index over documents without requiring a generative LLM during online
retrieval. The graph stores documents, chunks, normalized token/phrase nodes,
and semantic nodes that represent contextual meanings of a token or phrase.

Retrieval combines exact token/phrase matches, fuzzy phrase lookup, dense
embedding fallback, modifier-aware scoring for compositional phrases, BM25, and
a chunk co-occurrence graph.

This document describes the current code surface used to build, finalize,
query, persist, and inspect a `LiteSemRAG` instance.

---

## 1. Conceptual Model

The graph has four main node types:

| Node | Dataclass | Purpose |
| --- | --- | --- |
| Document | `DocumentNode` | One indexed source document; owns chunk nodes. |
| Chunk | `ChunkNode` | A text window cut from a document; stores chunk text and token count. |
| Token | `TokenNode` | A normalized surface form for a token or phrase. |
| Semantic node | `SemNode` | One contextual meaning of a token/phrase, with centroid embedding, chunk links, spans, description, IDF, and BM25 scores. |

Supporting records:

- `TextEmbedding` stores an embedding tied to a chunk span and optional
  phrase/head/modifier metadata.
- `SpanOccurrence` stores the persisted chunk/span location for a token or
  phrase occurrence.
- `CoOccurrenceGraph` and `CoOccurrenceNode` are query-time structures used to
  rank chunks by matched semantic nodes and their shared chunk evidence.

`Prototype = SemNode` remains as a backward-compatible alias for old imports and
pickles.

---

## 2. Construction

Current constructor:

```python
LiteSemRAG(
    min_occurrences_for_description=20,
    retrieve_top_k=5,
    chunk_size=300,
    device="cuda",
    discard_no_word=False,
    sem_description_prompt_context_mode="sentence_neighbors",
    consensus_ratio_threshold=0.8,
    min_description_candidates=3,
    use_llm_candidate_filter=False,
    use_llm_semantic_labeler=False,
    disambiguate_query_sense=True,
    sem_assignment_method="Anchor-F-mutual [ce_fallback]",
    anchor_fraction=0.15,
    anchor_fft_ratio=0.70,
    anchor_min_count=2,
    prop_knn_k=8,
)
```

Important parameters:

- `min_occurrences_for_description` gates "does this token get a description /
  sense split". Indexing builds no semantic nodes — every embedding is retained on
  its `TokenNode` until the corpus is fully scanned. At `finalize()`, any
  token/phrase whose accumulated occurrence count is `>=` this value runs the full
  description-prediction / sense-split path; rarer tokens fall back to a single
  description-less basic node. Set to `None` to give every token a basic node with
  no description; set to `1` to describe every token. Larger values trade recall of
  sense-aware nodes for fewer cross-encoder / Wikidata / LLM calls at finalize.
- `retrieve_top_k` is the default cap for `get_top_k_chunks_for_sem_node`.
- `chunk_size` is passed to `split_doc()` as the maximum chunk token count.
- `device` is used by the Transformer encoder and query database.
- `discard_no_word` is passed into span extraction filters.
- `sem_description_prompt_context_mode` controls how much chunk context is used
  when building semantic-description prompts.
- `consensus_ratio_threshold` controls whether cross-encoder description
  predictions have enough sample agreement to be accepted.
- `min_description_candidates` controls candidate-bank padding/fallback
  behavior.
- `use_llm_candidate_filter` merges/filters the raw Wikidata candidates with an
  LLM **and** classifies the FFT samples in a single call, following the combined
  prompt in `wikidata_llm_candidate_merge_experiment.ipynb`. When enabled, the
  merged senses become the candidate bank and the per-sample `sample_judgments`
  directly supply each FFT sample's sense — no separate cross-encoder or
  per-sample LLM judgment runs for the sampled occurrences. FFT samples the LLM
  marks `unsupported`/`ambiguous` get no sense here and are treated like ordinary
  non-sampled occurrences downstream (no KNN safety check). Non-sampled and
  danger-zone occurrences are still judged the usual way (cross-encoder, or the
  labeler when `use_llm_semantic_labeler` is on). The standalone, samples-free
  `WikidataDefinitionFilter.filter_definitions()` is retained as an extra utility.
- `use_llm_semantic_labeler` switches semantic-description sample judgments
  from the cross-encoder to per-sample LLM calls. FFT samples and fallback
  samples are judged one prompt at a time, using `llm_semantic_labeler.py`.
  (When `use_llm_candidate_filter` is on, the sampled FFT occurrences are already
  judged by the combined merge call; the labeler then only applies to the
  remaining non-sampled / danger-zone occurrences.)
- `sem_assignment_method` selects how `finalize()` assigns descriptions / splits
  senses for tokens that pass the occurrence gate. Two families:
  - **FFT family** (`"FFT-CE"` / `"FFT-LLM"`, or any non-`Anchor-*` name): the
    original path — FFT sampling → per-sample judgment → consensus gate →
    d1/d2 medoid split, driven by `use_llm_candidate_filter` /
    `use_llm_semantic_labeler` as before. Unchanged and never removed.
  - **Anchor family** (the new default): ports
    `hotpotqa_anchor_propagation_compare.ipynb`. A small `anchor_fraction` of
    occurrences (70% FFT-selected + 30% random) is labeled by the existing
    candidate-merge LLM path (or the cross-encoder when no LLM is enabled), then
    those labels propagate over a `(mutual-)kNN` graph; the cross-encoder acts as
    a "veto". Records grouped by propagated description become the split senses.
    All twelve notebook variants are selectable by name:
    `Anchor-C [ce_fallback|re_llm]`, `Anchor-D [...]`,
    `Anchor-E-{plain,mutual} [...]`, `Anchor-F-{plain,mutual} [...]`
    (C=plain/D=mutual kNN; E adds CE margin veto; F adds anchor-center + rare-class
    conservatism). `[ce_fallback]` resolves leftover `uncertain` records with the
    CE top-1; `[re_llm]` re-asks the LLM per occurrence. Default:
    `"Anchor-F-mutual [ce_fallback]"`.
- `anchor_fraction`, `anchor_fft_ratio`, `anchor_min_count`, `prop_knn_k` are the
  core tunable anchor hyper-parameters. The remaining notebook hyper-parameters
  (vote ratios, `high_margin`, `ce_oppose_gap`, rare thresholds, `max_rounds`)
  keep their notebook defaults in `ANCHOR_PROP_DEFAULTS` and can be overridden on
  `self.anchor_prop_params` after construction.
- `disambiguate_query_sense` enables query-time sense disambiguation. When a
  query span resolves to a token that built multiple *described* semantic nodes
  during indexing (a genuinely multi-sense token), the exact match is chosen by
  re-running the index-time semantic judgment (cross-encoder, or LLM when
  `use_llm_semantic_labeler` is on) over the query context against those senses'
  candidate definitions, instead of by embedding similarity. Tokens with a
  single semantic node skip the judgment and resolve directly.

The constructor initializes graph lists, ID counters, phrase/modifier indexes,
runtime model handles, a `ThreadPoolExecutor`, and a schema version marker. It
loads:

- a local DeBERTa encoder/tokenizer from
  `/home/xiaoyue/ProtoGraphRAG/deberta-v3-large`;
- a spaCy English pipeline via `_load_nlp()`;
- a cross-encoder semantic-description model named
  `cross-encoder/nli-deberta-v3-large`;
- an optional reranker, currently disabled by default in `load_reranker()`.

Call `shutdown()` to release the executor when a long-lived process is done.

---

## 3. Indexing Pipeline

### 3.1 Entry Points

`index_document(doc_name, multiprocessing=True)` is the public document entry
point. The `multiprocessing` argument is retained for API compatibility, but the
method now delegates to `index_document_parallel(doc_name)`.

These legacy methods are compatibility wrappers around the same staged pipeline:

- `index_document_single_processing(doc_name)`
- `index_document_threaded(doc_name)`
- `index_document_multi_processing(doc_name)`

`index_json(chunk_list, batch_size=8, queue_size=4, sample_count=None)` indexes
pre-built chunk records instead of reading and splitting a PDF.

### 3.2 Staged Pipeline

`index_document_parallel()` builds `{"doc_name": ..., "text": ...}` chunk
records from `split_doc()`, then calls `_index_chunk_records_pipeline()`.

`_index_chunk_records_pipeline()` has three coordinated stages:

1. CPU preprocessing creates `DocumentNode` / `ChunkNode` records and extracts
   spans with `extract_important_spans()`.
2. Phrase routing calls `_prepare_index_phrase_spans()`, which uses
   `PhraseAnalyzer` to classify phrases and route compositional phrases to both
   the full phrase and semantic head.
3. GPU batching calls `encode_chunk_batch()`, then CPU consumption calls
   `get_token_embeds()` and `process_embeds()`.

Indexing builds no semantic nodes and runs no clustering: after chunks are
consumed the pipeline only logs timing. All semantic-node construction is
deferred to `finalize()`.

### 3.3 Embedding Ingestion

`process_embeds(new_chunk_node, phrase_embs, token_embs)` handles every
extracted phrase/token embedding:

1. `_make_text_embedding()` wraps the tensor and occurrence metadata.
2. `_get_or_create_token_node()` resolves or creates the `TokenNode`.
3. Atomic phrases set `force_single_semantic=True`.
4. `_register_compositional_phrase_relation()` records full phrase, head, and
   modifier relationships on token nodes.
5. The embedding is appended to `TokenNode.embeds_buffer`. Every occurrence is
   retained — the buffer is never flushed or capped during indexing, so by the
   end of the run each token holds all of its embeddings.
6. `_append_token_occurrence()` records the occurrence on the token itself.

---

## 4. Phrase And Modifier Handling

`phrase_analysis.py` defines:

- `PHRASE_TYPE_ATOMIC`
- `PHRASE_TYPE_COMPOSITIONAL`
- `PHRASE_TYPE_SINGLE_TOKEN`
- `PHRASE_TYPE_UNKNOWN`
- `PhraseAnalysis`
- `PhraseAnalyzer`

Atomic phrases include non-trivial named entities, configured fixed
collocations, and title aliases. They are forced into a single semantic node
because splitting them by head/modifier would usually be wrong.

Compositional phrases expose:

- `head_text` / `head_text_norm`
- `modifier_texts` / `modifier_texts_norm`
- `modifier_spans`
- `atomic_modifier_spans`
- dependency labels for inspection

During indexing, compositional phrase occurrences store both the original
surface text and normalized head/modifier metadata. During querying,
`_prepare_query_tokens(..., expand_compositional=True)` expands a compositional
query phrase into grouped members: full phrase, head, and modifiers. The group
roles use `COMPOSITIONAL_QUERY_ROLE_WEIGHT`:

```python
{
    "phrase": 1.0,
    "head": 0.7,
    "modifier": 0.2,
}
```

`build_modifier_postings()` creates a `(head, modifier) -> sem_node_id counts`
index from final `SemNode.span_occurrences`. Query-time modifier boosts are
computed by `_score_chunk_modifier_boosts()` and applied to broad-search chunk
lists or co-occurrence scores.

---

## 5. Semantic Node Construction

All semantic nodes are built in `finalize()`. `finalize_token_nodes()` walks
every `TokenNode` and, using the token's fully accumulated `embeds_buffer`,
routes it by occurrence count: tokens reaching `min_occurrences_for_description`
go to `build_sem_node()`, the rest to `create_basic_sem_node()` (one
description-less node). After a node is built the buffer is cleared.

`build_sem_node(token_node)` decides whether the token collapses to a single
basic node or runs the description-driven sense split:

- `force_single_semantic` (entities / atomic phrases) routes straight to
  `create_basic_sem_node()`.
- `semantic_type_cls(token_node)` returns false for highly concentrated tokens
  (high `get_s_mean()`), which also collapse to a single basic node.
- Otherwise the token enters the sense-split path via
  `_build_sem_node_from_cluster()` (despite the legacy name, **no HDBSCAN runs**).

`_build_sem_node_from_cluster()` creates a `SemNode`, initializes retained sample
embeddings, computes a centroid, and calls `_assign_sem_description_on_build()`,
which does the actual sense splitting: it FFT-samples the occurrences down to
`fft_max_samples`, predicts a description per sample, and may split one token into
multiple described `SemNode` objects when samples disagree coherently. Every
operation over the buffer is linear in the number of occurrences.

`SemNode.retained_text_embeddings` keeps a reservoir-sampled subset of source
embeddings (`sem_retained_embed_limit`, default 10). These samples are used for
centroid recomputation, semantic-description prompts, split/merge decisions, and
inspection output.

---

## 6. Semantic Descriptions

Semantic descriptions are optional labels assigned from a Wikidata-style
candidate bank. They improve interpretability and support merging or splitting
semantic nodes that share or mix meanings.

Main steps:

1. `_load_sem_description_candidate_bank()` loads and caches candidates using
   `load_wikidata_definition_candidates()` and
   `build_wikidata_candidate_bank()`.
2. `_predict_sem_description_from_samples()` FFT-samples the occurrences, builds a
   prompt per sample, and judges each one. In the default path this scores the
   sample against candidate definitions with the cross-encoder (or, when
   `use_llm_semantic_labeler` is on, with a per-sample LLM call). When
   `use_llm_candidate_filter` is on, this step instead issues **one combined LLM
   call** (`_run_combined_merge_for_samples()` →
   `WikidataDefinitionFilter.filter_definitions_with_samples()`) that both merges
   the Wikidata candidates and returns `sample_judgments`; the matched-sense
   judgments become the sample senses and `unsupported`/`ambiguous` samples drop
   out of the vote.
3. `_assign_sem_description_on_build()` applies consensus rules. If a token's
   FFT samples yield multiple coherent predicted descriptions, it can split the
   token's occurrences into multiple `SemNode` objects; non-sampled and
   danger-zone occurrences are then assigned by medoid / d1-d2 distance (falling
   back to the cross-encoder or labeler as configured). When `sem_assignment_method`
   names an `Anchor-*` variant, this method instead dispatches to
   `_assign_sem_description_anchor_propagation()`, which labels an
   `anchor_fraction` subset (via the same merge / cross-encoder path), propagates
   the labels over a `(mutual-)kNN` graph with an optional cross-encoder veto, and
   groups records by propagated description into the split `SemNode` objects
   (steps 1–2 above still supply the candidate bank and anchor labels).
4. `merge_duplicate_description_sem_nodes()` later merges same-token semantic
   nodes with matching descriptions during `finalize()`.

Prompt context is controlled by `sem_description_prompt_context_mode`; supported
helpers include sentence-only, neighboring-sentence, full-chunk, and
boundary-extended context extraction.

Description, split, merge, and no-result events are stored in memory and written
by `save_sem_description_logs()` or automatically by `finalize()` under
`logs/sem_description_*.log`.

---

## 7. Finalization

Call `finalize()` after all indexing calls and before normal querying.

Current finalize order:

1. Reset semantic-description logs.
2. Validate the graph is non-empty.
3. Recompute average chunk length for BM25.
4. Remove empty placeholder token nodes.
5. `finalize_token_nodes()` builds every token's semantic node(s) from its fully
   accumulated `embeds_buffer` (sense split or basic node, per
   `min_occurrences_for_description`), then clears the buffer.
6. `merge_duplicate_description_sem_nodes()` merges same-description semantic
   nodes.
7. Validate semantic nodes exist.
8. `build_modifier_postings()` builds head/modifier postings.
9. `assign_idf()` computes token and semantic-node IDF.
10. `get_sem_BM25()` computes per-sem-node chunk BM25 scores.
11. `build_query_database()` stacks normalized semantic-node embeddings.
12. `build_phrase_query()` builds the word-to-phrase inverted index.
13. `build_chunk2sem_edge()` rebuilds chunk-to-sem reverse edges.
14. `save_doc_to_json()` writes `index_documents.json`.
15. `_save_sem_description_logs_to_timestamped_file()` writes a timestamped log
    under `logs/`.

`finalize()` raises a `ValueError` for empty graphs or missing semantic-node
embeddings.

---

## 8. Querying

### 8.1 Query Preparation

`_prepare_query_tokens(query_text, print_important_tokens=False,
expand_compositional=False)`:

- cleans the query;
- encodes it with `encode_text()`;
- extracts entities/phrases/tokens with `extract_important_phrases()` and
  `extract_important_tokens()`;
- analyzes phrase type through `PhraseAnalyzer`;
- optionally expands compositional phrases into grouped phrase/head/modifier
  query units.

`_resolve_query_matches()` then produces, for each query unit:

- `exact_sem_node` from the existing token node. For a token that built
  multiple described semantic nodes, `_disambiguate_query_sem_node_by_description()`
  picks the sense by re-running the index-time cross-encoder / LLM judgment over
  the query context (controlled by `disambiguate_query_sense`); otherwise, or on
  an inconclusive judgment, it falls back to the highest centroid similarity;
- `fuzzy_sem_nodes` from `phrase_word_intersection_query()` and
  `max_cosine_sem_nodes()`;
- `retrieved_sem` plus `semantic_weight` from `query_by_sim()` when exact and
  fuzzy matching fail.

### 8.2 Retrieval Methods

`broad_search_query(query_text, top_k=10, candidate=30)`:

- resolves query matches;
- gathers chunks connected to matched semantic nodes;
- deduplicates chunk IDs in match order;
- applies modifier boosts when available;
- optionally reranks with `self.reranker` if enabled;
- returns `(chunks, chunk_ids)`.

`multi_level_query(query_text, top_k_chunk=10, top_k_each_isolated_chunk=2,
isolate_chunk_ratio=0.2, isolate_retrieve_mode="sequential",
print_important_tokens=True, search_mode="broad")` is the main graph retrieval
method:

1. Resolves exact, fuzzy, and similarity matches with compositional expansion.
2. Assigns match levels using `CO_OCCURRENCE_NODE_QUERY_WEIGHT`:
   `(1.2, 1, 0.6, 0.3)` for entity/exact, token/exact, partial, and
   similarity-only matches.
3. Builds a `CoOccurrenceGraph`.
4. Connects semantic nodes that share chunks.
5. Aggregates weighted chunk scores with BM25 and query-role weights.
6. Splits the output budget between connected chunks and isolated BM25 chunks
   through `ListBatchExtractor`.
7. Returns `(chunks, chunk_ids, co_occurrence_graph)`.

Other query helpers:

- `query_by_sim(query_embeds)` searches the normalized semantic-node embedding
  matrix.
- `get_top_k_chunks_for_sem_node(sem_node, top_k=None, retrieve_all=False)`
  returns chunks connected to one semantic node by descending edge weight.
- `chunk_id2text(ids)` maps stored chunk IDs back to chunk text and validates
  the list invariant.

---

## 9. Persistence

`LiteSemRAG` is pickleable but runtime handles are stripped and restored:

- Runtime fields include encoder, tokenizer, executor, spaCy pipeline, reranker,
  semantic-description model, encoder lock, and phrase analyzer.
- `__getstate__()` / `__setstate__()` use `_snapshot_runtime_fields()`,
  `_clear_runtime_fields()`, and `_restore_runtime_fields_from_snapshot()`.
- `save_data(path)` / `load_data(path)` persist the full object.
- `save_data_split(pkl_path)` / `load_data_split(pkl_path)` store graph
  structure separately from tensors in a companion `_tensors.pt` file.
- `node_instance2id()` and `node_id2instance()` convert references for
  inspection or storage.

Backward compatibility is handled by `_ensure_backward_compatible_attrs()` and
node-specific helpers. `CURRENT_SCHEMA_VERSION = 5`; older pickles are upgraded
for renamed `proto_*` fields, missing occurrence metadata, retained embeddings,
compositional phrase metadata, and modifier postings, and have the removed
`anomaly_section` / `anomaly_threshold` attributes stripped (schema 5).

`save_doc_to_json()` writes only indexed document names to
`index_documents.json`.

---

## 10. Deletion And Rebuild

`delete_by_document(doc_name_list)` removes documents and their chunks, then
calls `rebuild_metadata_after_deletion()`.

The rebuild path:

- removes invalid chunk references from documents, semantic nodes, token nodes,
  anomalies, retained embeddings, and headed phrase records;
- removes semantic nodes with no remaining chunks;
- resets semantic, token, chunk, and document IDs;
- recomputes chunk average length, IDF, BM25, chunk-to-sem edges, phrase index,
  modifier postings, and query database.

Use this path instead of manually mutating graph lists.

---

## 11. Inspection And Debugging

Useful methods:

- `debug_extract_important_spans()` exposes the same span extractor used during
  indexing.
- `inspect_multi_sem_token_nodes()` returns tokens with multiple semantic nodes
  and example sentences.
- `inspect_described_sem_token_nodes()` returns semantic nodes with
  descriptions and examples.
- `show_multi_sem_token_nodes()` and `show_described_sem_token_nodes()` produce
  HTML reports.
- `print_sem_description_logs()`, `show_sem_description_logs()`, and
  `save_sem_description_logs()` inspect semantic-description decisions.
- `print_wikidata_no_result_logs()` summarizes failed candidate-bank lookups.
- `get_modifier_postings()` returns a plain-dict copy of the modifier postings.
- `print_memory_size()` reports approximate graph object size.

---

## 12. Typical Usage

```python
from RAG_graph import LiteSemRAG

rag = LiteSemRAG(
    chunk_size=300,
    device="cuda",
)

for doc_path in doc_paths:
    rag.index_document(doc_path)

rag.finalize()

chunks, chunk_ids, cog = rag.multi_level_query(
    "What did Marie Curie discover?",
    top_k_chunk=10,
)

rag.save_data("litesemrag.pkl")
rag.shutdown()
```

Loading:

```python
from RAG_graph import LiteSemRAG

rag = LiteSemRAG.load_data("litesemrag.pkl")
chunks, chunk_ids = rag.broad_search_query("polonium discovery", top_k=5)
rag.shutdown()
```

---

## 13. File Map

- `RAG_graph.py`：core graph, indexing, semantic-node construction,
  description assignment, query ranking, persistence, deletion, inspection.
- `text_processing.py`：cleaning, chunking, span extraction, Transformer
  encoding, span embedding alignment, BM25 helpers.
- `phrase_analysis.py`：phrase classification and head/modifier extraction.
- `utils.py`：embedding math (`get_s_mean` / `average_embeds` / `sem_embed_sim`),
  farthest-first-traversal sampling, Wikidata/Wikipedia fetch helpers, Wikidata
  candidate loading, cross-encoder score extraction, dataset/eval utilities.
  (`hdbscan_cluster` remains but is legacy — unused by the engine.)
- `wikidata_definition_filter.py`：LLM-based candidate-sense merging/filtering.
  `filter_definitions()` is the samples-free, SQLite-cached path;
  `filter_definitions_with_samples()` adds the FFT samples to the prompt and
  returns per-sample `sample_judgments` (no cache), powering
  `use_llm_candidate_filter`.
- `llm_semantic_labeler.py`：LLM candidate selection/cache experiments.
- `local_llm.py`：OpenAI-compatible chat client wrapper.
- `simple_rag.py`：standalone baseline RAG used for HotpotQA comparison; not
  part of the `LiteSemRAG` runtime.
- `index_documents.json`：document-name index written by `save_doc_to_json()`.
- `logs/sem_description_*.log`：semantic-description logs written by
  `finalize()`.
- `archive/`：deprecated experiment lines kept for reference only — the
  Wikipedia redirect/synonym index (`wikipedia_redirects.py` etc.) and the
  WordNet/SemCor word-sense exploration (`wordnet_explorer.py`,
  `prepare_semcor.py` etc.). None are referenced by the runtime. See
  `archive/README.md`.
