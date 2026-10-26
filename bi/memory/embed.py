"""Local text embedders.

Backends (memory.embeddings.backend):
  onnx                   all-MiniLM-L6-v2 as ONNX (onnxruntime + tokenizers): ~0.3 s load, ~2 ms/query
  sentence-transformers  the same model via sentence-transformers (pulls in torch; ~6 s import)
  hash                   dependency-free hashed bag of words + char trigrams (lexical only)
  auto                   onnx -> sentence-transformers -> hash, first that loads
"""
from __future__ import annotations

import logging
import re
import tarfile
import time
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

MINILM_FAMILY = "minilm-l6-v2"      # ST and ONNX share weights -> same index


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashEmbedder:
    """Deterministic feature hashing. Captures word overlap, not meaning."""

    _word = re.compile(r"[a-z0-9']+")
    _stop = set("a an the is are was were be to of and or in on at for with my me i you it this that "
                "do does did what who how about your".split())

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    @staticmethod
    def _stem(w: str) -> str:
        for suf in ("ing", "ies", "es", "ed", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)] + ("y" if suf == "ies" else "")
        return w

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in self._word.findall(text.lower()):
            if w in self._stop:
                continue
            w = self._stem(w)
            feats = [(w, 1.0)] + [(f"#{w[i:i+3]}", 0.35) for i in range(max(1, len(w) - 2))]
            for f, weight in feats:
                h = zlib.crc32(f.encode())
                v[h % self.dim] += weight if (h >> 16) & 1 else -weight
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model: str, cache_dir: Path):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, cache_folder=str(cache_dir), device="cpu")
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.name = MINILM_FAMILY if "MiniLM-L6-v2" in model else model.split("/")[-1]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                            show_progress_bar=False), dtype=np.float32)


class OnnxMiniLMEmbedder:
    def __init__(self, url: str, cache_dir: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = ensure_onnx_minilm(url, cache_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.dim = 384
        self.name = MINILM_FAMILY

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return _normalize(pooled)


def ensure_onnx_minilm(url: str, cache_dir: Path) -> Path:
    d = cache_dir / "all-MiniLM-L6-v2-onnx"
    if (d / "model.onnx").exists() and (d / "tokenizer.json").exists():
        return d
    import httpx
    d.mkdir(parents=True, exist_ok=True)
    tgz = d / "onnx.tar.gz"
    log.info("downloading ONNX MiniLM from %s", url)
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        with open(tgz, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if m.isfile() and name in ("model.onnx", "tokenizer.json", "config.json", "vocab.txt"):
                src = tf.extractfile(m)
                if src:
                    (d / name).write_bytes(src.read())
    tgz.unlink(missing_ok=True)
    return d


def make_embedder(cfg) -> Embedder:
    e = cfg.section("memory.embeddings")
    backend = (e.get("backend") or "auto").lower()
    models_dir = cfg.path_of("paths.models_dir", "models")
    order = ["onnx", "sentence-transformers", "hash"] if backend == "auto" else [backend]
    errors = []
    for b in order:
        t0 = time.perf_counter()
        try:
            if b == "sentence-transformers":
                emb = SentenceTransformerEmbedder(e.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                                                  models_dir / "sentence-transformers")
            elif b == "onnx":
                emb = OnnxMiniLMEmbedder(e.get("onnx_url"), models_dir)
            elif b == "hash":
                emb = HashEmbedder()
            else:
                raise ValueError(f"unknown embeddings backend {b}")
            log.info("embedder %s (%s) loaded in %.0f ms", b, emb.name, (time.perf_counter() - t0) * 1000)
            return emb
        except Exception as ex:  # noqa: BLE001 - try the next backend
            errors.append(f"{b}: {ex}")
            log.warning("embedder %s unavailable: %s", b, ex)
    raise RuntimeError("No embedder could be loaded: " + "; ".join(errors))
