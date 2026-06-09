"""Minimal chunk-vector RAG baseline (HotpotQA-oriented, independent of LiteSemRAG).

Index: fixed-size character chunks -> API or local embeddings -> FAISS (cosine via L2-normalized IP).
Query: embed question -> top-k chunk texts (no scores returned).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np

DEFAULT_INDEX_DIR = Path("simple_rag_index")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BACKEND = "api"
DEFAULT_LOCAL_MODEL_PATH = Path("/home/xiaoyue/e5-large-v2")
DEFAULT_LOCAL_DEVICE = "auto"
DEFAULT_LOCAL_MAX_LENGTH = 512
DEFAULT_CHUNK_CHARS = 800
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_API_KEY_FILE = "API_KEY"
DEFAULT_TOP_K = 5
EMBED_BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# HotpotQA (same document/sample shape as utils.build_hotpot_retrieval_dataset)
# ---------------------------------------------------------------------------


def load_hotpot_documents(
    file_path: str | Path,
    num_samples: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load HotpotQA distractor JSON into document and sample records."""
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]

    title_to_doc: dict[str, dict[str, Any]] = {}
    for item in data:
        for title, sentences in item["context"]:
            text = " ".join(sentences).strip()
            if title not in title_to_doc:
                title_to_doc[title] = {
                    "title": title,
                    "text": text,
                    "sentences": sentences,
                }
            elif len(text) > len(title_to_doc[title]["text"]):
                title_to_doc[title]["text"] = text
                title_to_doc[title]["sentences"] = sentences

    documents: list[dict[str, Any]] = []
    title_to_id: dict[str, int] = {}
    for idx, (title, doc) in enumerate(title_to_doc.items()):
        documents.append(
            {
                "doc_id": idx,
                "title": title,
                "text": doc["text"],
                "sentences": doc["sentences"],
            }
        )
        title_to_id[title] = idx

    samples: list[dict[str, Any]] = []
    for item in data:
        gold_titles = {title for title, _ in item["supporting_facts"]}
        gold_doc_ids = [
            title_to_id[title] for title in gold_titles if title in title_to_id
        ]
        samples.append(
            {
                "sample_id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "gold_doc_ids": gold_doc_ids,
            }
        )

    return documents, samples


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text_fixed_chars(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into fixed-width character chunks with optional overlap."""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# OpenAI API key (standalone; mirrors local_llm key-file parsing)
# ---------------------------------------------------------------------------


def read_openai_api_key(
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
) -> str:
    if api_key:
        return api_key.strip()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    key_path = Path(api_key_file or os.getenv("OPENAI_API_KEY_FILE") or DEFAULT_API_KEY_FILE)
    if not key_path.is_absolute():
        key_path = Path.cwd() / key_path
    if not key_path.exists():
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY or provide API_KEY file."
        )

    text = key_path.read_text(encoding="utf-8").strip()
    key_match = re.search(r"sk-[A-Za-z0-9_-]+", text)
    if key_match:
        return key_match.group(0)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            name = name.strip().strip("'\"")
            value = value.strip().strip(",").strip().strip("'\"")
            if name in {"OPENAI_API_KEY", "API_KEY", "chatgpt_api"}:
                return value
            if value.startswith("sk-"):
                return value
        if line.startswith("sk-"):
            return line
    raise ValueError(f"No API key found in {key_path}")


# ---------------------------------------------------------------------------
# Embeddings + FAISS index
# ---------------------------------------------------------------------------


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _mean_pool_vectors(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("mean pool expects a non-empty 2D array")
    return arr.mean(axis=0)


@dataclass
class ChunkRecord:
    text: str
    doc_id: int
    title: str
    chunk_idx: int


@dataclass
class SimpleVectorRAG:
    """Chunk-level vector retrieval using API/local embeddings and FAISS."""

    index_dir: Path = field(default_factory=lambda: DEFAULT_INDEX_DIR)
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    local_model_path: str | Path = field(default_factory=lambda: DEFAULT_LOCAL_MODEL_PATH)
    local_device: str = DEFAULT_LOCAL_DEVICE
    local_max_length: int = DEFAULT_LOCAL_MAX_LENGTH
    chunk_size: int = DEFAULT_CHUNK_CHARS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    api_key: str | None = None
    api_key_file: str | Path | None = None
    use_faiss_gpu: bool = True

    _client: Any = field(default=None, init=False, repr=False)
    _local_tokenizer: Any = field(default=None, init=False, repr=False)
    _local_model: Any = field(default=None, init=False, repr=False)
    _resolved_local_device: str | None = field(default=None, init=False, repr=False)
    _records: list[ChunkRecord] = field(default_factory=list, init=False)
    _index: faiss.Index | None = field(default=None, init=False, repr=False)
    _dim: int | None = field(default=None, init=False, repr=False)
    _gpu_resources: Any = field(default=None, init=False, repr=False)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "Install the openai package to use SimpleVectorRAG."
                ) from exc
            key = read_openai_api_key(self.api_key, self.api_key_file)
            base_url = os.getenv("OPENAI_BASE_URL") or None
            self._client = OpenAI(api_key=key, base_url=base_url)
        return self._client

    def _get_local_encoder(self):
        if self._local_tokenizer is None or self._local_model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Install torch and transformers to use local embeddings."
                ) from exc

            if self.local_device == "auto":
                self._resolved_local_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._resolved_local_device = self.local_device

            model_path = Path(self.local_model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                use_fast=True,
            )
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
            )
            model.to(self._resolved_local_device)
            model.eval()
            self._local_tokenizer = tokenizer
            self._local_model = model
        return self._local_tokenizer, self._local_model

    @staticmethod
    def _e5_average_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def _prefix_e5_texts(texts: list[str], input_type: str) -> list[str]:
        if input_type == "query":
            prefix = "query: "
        elif input_type == "passage":
            prefix = "passage: "
        else:
            raise ValueError("input_type must be 'query' or 'passage'")
        return [text if text.startswith(prefix) else f"{prefix}{text}" for text in texts]

    def _embed_api_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        client = self._get_client()
        response = client.embeddings.create(
            input=texts,
            model=self.embedding_model,
        )
        ordered = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        vectors = np.asarray(ordered, dtype=np.float32)
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        return vectors

    def _embed_local_batch(self, texts: list[str], input_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch to use local embeddings.") from exc

        tokenizer, model = self._get_local_encoder()
        encoded = tokenizer(
            self._prefix_e5_texts(texts, input_type),
            padding=True,
            truncation=True,
            max_length=self.local_max_length,
            return_tensors="pt",
        )
        device = self._resolved_local_device or "cpu"
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            embeddings = self._e5_average_pool(outputs.last_hidden_state, encoded["attention_mask"])

        vectors = embeddings.detach().to(torch.float32).cpu().numpy().astype(np.float32)
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        return vectors

    def _embed_batch(self, texts: list[str], input_type: str) -> np.ndarray:
        backend = self.embedding_backend.lower()
        if backend in {"api", "openai"}:
            return self._embed_api_batch(texts)
        if backend in {"local", "e5", "e5-large-v2"}:
            return self._embed_local_batch(texts, input_type)
        raise ValueError("embedding_backend must be one of: 'api', 'openai', 'local', 'e5'")

    def _embed_with_mean_pool(self, texts: list[str], input_type: str) -> np.ndarray:
        """One vector per text; long texts are sub-chunked and mean-pooled."""
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        max_chars = max(self.chunk_size, 256)
        pooled: list[np.ndarray | None] = [None] * len(texts)
        short_indices: list[int] = []
        short_texts: list[str] = []

        for idx, text in enumerate(texts):
            if len(text) <= max_chars:
                short_indices.append(idx)
                short_texts.append(text)
                continue

            sub_chunks = chunk_text_fixed_chars(
                text,
                chunk_size=max_chars,
                overlap=0,
            )
            sub_vectors: list[np.ndarray] = []
            for start in range(0, len(sub_chunks), EMBED_BATCH_SIZE):
                sub_vectors.append(
                    self._embed_batch(
                        sub_chunks[start : start + EMBED_BATCH_SIZE],
                        input_type=input_type,
                    )
                )
            if sub_vectors:
                sub_vecs = np.vstack(sub_vectors)
                pooled[idx] = _mean_pool_vectors(sub_vecs)
            else:
                pooled[idx] = np.zeros((self._dim or 0,), dtype=np.float32)

        if short_texts:
            short_vectors: list[np.ndarray] = []
            for start in range(0, len(short_texts), EMBED_BATCH_SIZE):
                short_vectors.append(
                    self._embed_batch(
                        short_texts[start : start + EMBED_BATCH_SIZE],
                        input_type=input_type,
                    )
                )
            short_vecs = np.vstack(short_vectors)
            for idx, vec in zip(short_indices, short_vecs, strict=True):
                pooled[idx] = vec

        if any(vec is None for vec in pooled):
            raise RuntimeError("Embedding failed for one or more texts.")
        return np.vstack(pooled).astype(np.float32)

    def _can_use_faiss_gpu(self) -> bool:
        return (
            self.use_faiss_gpu
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > 0
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "index_cpu_to_gpu")
        )

    def _move_index_to_gpu(self, cpu_index: faiss.Index) -> faiss.Index:
        if self._gpu_resources is None:
            self._gpu_resources = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)

    def _move_index_to_cpu(self) -> faiss.Index:
        if (
            self._index is not None
            and hasattr(faiss, "index_gpu_to_cpu")
            and hasattr(faiss, "GpuIndex")
            and isinstance(self._index, faiss.GpuIndex)
        ):
            return faiss.index_gpu_to_cpu(self._index)
        if self._index is None:
            raise RuntimeError("Index is empty.")
        return self._index

    def _build_faiss_index(self, vectors: np.ndarray) -> faiss.Index:
        vectors = _l2_normalize(vectors)
        dim = vectors.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)

        if self._can_use_faiss_gpu():
            return self._move_index_to_gpu(cpu_index)

        return cpu_index

    def _add_vectors_to_index(self, vectors: np.ndarray) -> None:
        vectors = _l2_normalize(vectors)
        if self._index is None:
            self._index = self._build_faiss_index(vectors)
            self._dim = vectors.shape[1]
        self._index.add(vectors)

    def index_documents(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        reset: bool = True,
    ) -> int:
        """Chunk and embed Hotpot-style documents. Returns number of chunks indexed."""
        if reset:
            self._records = []
            self._index = None
            self._dim = None
            self._gpu_resources = None

        doc_list = list(documents)
        new_records: list[ChunkRecord] = []
        texts: list[str] = []

        for doc in doc_list:
            doc_id = int(doc["doc_id"])
            title = str(doc.get("title", ""))
            body = str(doc.get("text", ""))
            for chunk_idx, chunk in enumerate(
                chunk_text_fixed_chars(
                    body,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )
            ):
                new_records.append(
                    ChunkRecord(
                        text=chunk,
                        doc_id=doc_id,
                        title=title,
                        chunk_idx=chunk_idx,
                    )
                )
                texts.append(chunk)

        if not texts:
            return 0

        all_vectors: list[np.ndarray] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            all_vectors.append(self._embed_with_mean_pool(batch, input_type="passage"))

        vectors = np.vstack(all_vectors)
        self._add_vectors_to_index(vectors)
        self._records.extend(new_records)
        return len(new_records)

    def index_hotpot_file(
        self,
        file_path: str | Path,
        num_samples: int | None = None,
        *,
        reset: bool = True,
    ) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
        """Load HotpotQA JSON, index all unique documents, return (n_chunks, documents, samples)."""
        documents, samples = load_hotpot_documents(file_path, num_samples=num_samples)
        n_chunks = self.index_documents(documents, reset=reset)
        return n_chunks, documents, samples

    def save(self, index_dir: str | Path | None = None) -> Path:
        """Persist FAISS index and chunk metadata to disk."""
        if self._index is None or not self._records:
            raise RuntimeError("Nothing to save; build the index first.")

        out_dir = Path(index_dir or self.index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cpu_index = self._move_index_to_cpu()
        faiss.write_index(cpu_index, str(out_dir / "index.faiss"))

        metadata = [
            {
                "text": rec.text,
                "doc_id": rec.doc_id,
                "title": rec.title,
                "chunk_idx": rec.chunk_idx,
            }
            for rec in self._records
        ]
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        config = {
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "local_model_path": str(self.local_model_path),
            "local_device": self.local_device,
            "local_max_length": self.local_max_length,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dim": self._dim,
            "num_vectors": len(self._records),
        }
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self.index_dir = out_dir
        return out_dir

    def load(self, index_dir: str | Path | None = None) -> None:
        """Load a previously saved index from disk."""
        in_dir = Path(index_dir or self.index_dir)
        config_path = in_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.embedding_backend = config.get("embedding_backend", self.embedding_backend)
        self.embedding_model = config.get("embedding_model", self.embedding_model)
        self.local_model_path = config.get("local_model_path", self.local_model_path)
        self.local_device = config.get("local_device", self.local_device)
        self.local_max_length = int(config.get("local_max_length", self.local_max_length))
        self.chunk_size = int(config.get("chunk_size", self.chunk_size))
        self.chunk_overlap = int(config.get("chunk_overlap", self.chunk_overlap))
        self._dim = int(config["dim"])
        self._client = None
        self._local_tokenizer = None
        self._local_model = None
        self._resolved_local_device = None

        cpu_index = faiss.read_index(str(in_dir / "index.faiss"))
        if self._can_use_faiss_gpu():
            self._index = self._move_index_to_gpu(cpu_index)
        else:
            self._index = cpu_index
            self._gpu_resources = None

        with open(in_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self._records = [
            ChunkRecord(
                text=item["text"],
                doc_id=int(item["doc_id"]),
                title=str(item.get("title", "")),
                chunk_idx=int(item.get("chunk_idx", 0)),
            )
            for item in metadata
        ]
        self.index_dir = in_dir

    def _search_indices(self, question: str, top_k: int) -> list[int]:
        if self._index is None or not self._records:
            raise RuntimeError("Index is empty; call index_documents() or load() first.")
        if top_k <= 0:
            return []

        query_vec = self._embed_with_mean_pool([question], input_type="query")
        query_vec = _l2_normalize(query_vec)
        k = min(top_k, len(self._records))
        _, indices = self._index.search(query_vec, k)
        return [int(i) for i in indices[0] if 0 <= i < len(self._records)]

    def query_records(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[ChunkRecord]:
        """Return top-k chunk records (text, doc_id, title) for evaluation."""
        return [self._records[i] for i in self._search_indices(question, top_k)]

    def query(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """Return top-k chunk texts for a question (no scores)."""
        return [rec.text for rec in self.query_records(question, top_k=top_k)]

    def query_sample(
        self,
        sample: dict[str, Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[str]:
        """Run retrieval for a Hotpot-style sample using its question field."""
        return self.query(str(sample["question"]), top_k=top_k)

    def query_sample_records(
        self,
        sample: dict[str, Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[ChunkRecord]:
        """Run retrieval for a Hotpot-style sample; return chunk records."""
        return self.query_records(str(sample["question"]), top_k=top_k)


__all__ = [
    "ChunkRecord",
    "SimpleVectorRAG",
    "chunk_text_fixed_chars",
    "load_hotpot_documents",
    "read_openai_api_key",
]
