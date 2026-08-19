"""Pluggable embedding backends, selected by the EMBEDDING_BACKEND env var.

Why this exists: `.env.example` has always advertised
`EMBEDDING_BACKEND=fastembed | bge-m3 | openai`, setup-docker.sh flips it to
`bge-m3` and prints "bge-m3 embeddings", and the README sells bge-m3 as the
reason to take the Docker path -- but nothing ever read the variable. Every
path silently used BAAI/bge-small-en-v1.5, an ENGLISH model, which is exactly
why NB2 shows weak recall on Vietnamese paraphrases. Students were promised an
upgrade that never happened.

The default is unchanged (fastembed / bge-small / 384-dim), so the lite path
and every rubric threshold behave exactly as before. The other backends are
opt-in via the environment.

    EMBEDDING_BACKEND=fastembed     BAAI/bge-small-en-v1.5     384   (default, lite)
    EMBEDDING_BACKEND=multilingual  intfloat/multilingual-e5-large 1024 (fastembed)
    EMBEDDING_BACKEND=bge-m3        BAAI/bge-m3                1024  (sentence-transformers)
    EMBEDDING_BACKEND=openai        text-embedding-3-small     1536  (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from app.gpu import cuda_available, setup_cuda_dlls

DEFAULT_BACKEND = "fastembed"
DEFAULT_DEVICE = "auto"        # auto | cpu | cuda
# Registering the NVIDIA DLL dirs must happen before onnxruntime is first
# imported, and importing this module is the earliest hook every path shares.
setup_cuda_dlls()


@dataclass(frozen=True)
class BackendSpec:
    model: str
    dim: int
    provider: str       # fastembed | sentence-transformers | openai
    note: str = ""
    # E5-family models are trained ASYMMETRICALLY: documents are seen as
    # "passage: <text>" and questions as "query: <text>". Drop the prefixes and
    # you are querying the model off-distribution -- it still returns vectors,
    # nothing errors, the numbers are just worse. fastembed does NOT add them:
    # `TextEmbedding.query_embed` / `.passage_embed` fall through to plain
    # `embed()` for every ONNX model (only the Jina multitask class overrides
    # them), so the prefixes have to live here.
    query_prefix: str = ""
    passage_prefix: str = ""


BACKENDS: dict[str, BackendSpec] = {
    "fastembed": BackendSpec("BAAI/bge-small-en-v1.5", 384, "fastembed",
                             "English-focused; weak on Vietnamese paraphrase (that is the NB2 lesson)"),
    "multilingual": BackendSpec("intfloat/multilingual-e5-large", 1024, "fastembed",
                                "Multilingual, no extra dependency, ~2.2 GB download",
                                query_prefix="query: ", passage_prefix="passage: "),
    # bge-m3 is symmetric -- no prefixes, unlike the e5 family.
    "bge-m3": BackendSpec("BAAI/bge-m3", 1024, "sentence-transformers",
                          "Multilingual; needs sentence-transformers (requirements-full.txt)"),
    "openai": BackendSpec("text-embedding-3-small", 1536, "openai",
                          "Needs OPENAI_API_KEY; costs money"),
}


class Embedder:
    """Uniform `.embed(list[str]) -> Iterator[np.ndarray]`, matching fastembed."""

    def __init__(self, backend: str | None = None, device: str | None = None) -> None:
        name = (backend or os.getenv("EMBEDDING_BACKEND") or DEFAULT_BACKEND).strip().lower()
        if name not in BACKENDS:
            raise ValueError(
                f"Unknown EMBEDDING_BACKEND={name!r}. "
                f"Valid: {', '.join(sorted(BACKENDS))}"
            )
        self.backend = name
        self.spec = BACKENDS[name]
        self._impl = None

        dev = (device or os.getenv("EMBEDDING_DEVICE") or DEFAULT_DEVICE).strip().lower()
        if dev not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unknown EMBEDDING_DEVICE={dev!r}. Valid: auto, cpu, cuda")
        self.requested_device = dev
        # `auto` must never turn a working CPU run into a crash, so it only
        # upgrades when a CUDA session genuinely builds. `cuda` is explicit and
        # is allowed to fail loudly -- that is the point of asking for it.
        if self.spec.provider != "fastembed":
            self.device = "cpu"          # only the fastembed path is wired for CUDA
        elif dev == "cuda":
            self.device = "cuda"
        elif dev == "auto":
            self.device = "cuda" if cuda_available() else "cpu"
        else:
            self.device = "cpu"

    # dimension is a property of the chosen model, never a hard-coded constant
    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def model_name(self) -> str:
        return self.spec.model

    def _load(self):
        if self._impl is not None:
            return self._impl
        p = self.spec.provider
        if p == "fastembed":
            from fastembed import TextEmbedding
            if self.device == "cuda":
                # fastembed only WARNS when the CUDA provider fails and then
                # runs on CPU, so verify afterwards instead of assuming.
                self._impl = TextEmbedding(
                    model_name=self.spec.model, cuda=True, device_ids=[0]
                )
                if self.requested_device == "cuda" and not self._active_is_cuda(self._impl):
                    raise RuntimeError(
                        "EMBEDDING_DEVICE=cuda but the CUDA provider did not load "
                        "(fastembed fell back to CPU). Run `python -c "
                        "'from app.gpu import describe; print(describe())'` to see why."
                    )
            else:
                self._impl = TextEmbedding(model_name=self.spec.model)
        elif p == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:                       # pragma: no cover
                raise ImportError(
                    f"EMBEDDING_BACKEND={self.backend} needs sentence-transformers.\n"
                    "It ships with the Docker path:  pip install -r requirements-full.txt"
                ) from exc
            self._impl = SentenceTransformer(self.spec.model)
        elif p == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:                       # pragma: no cover
                raise ImportError(
                    "EMBEDDING_BACKEND=openai needs the openai package "
                    "(requirements-full.txt) and OPENAI_API_KEY."
                ) from exc
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("EMBEDDING_BACKEND=openai but OPENAI_API_KEY is unset.")
            self._impl = OpenAI()
        return self._impl

    def embed_documents(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        """Embed corpus documents (indexing side)."""
        pre = self.spec.passage_prefix
        return self.embed([pre + t for t in texts] if pre else texts)

    def embed_query(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        """Embed user questions (search side).

        Must pair with `embed_documents` -- mixing a prefixed index with an
        unprefixed query silently degrades every score.
        """
        pre = self.spec.query_prefix
        return self.embed([pre + t for t in texts] if pre else texts)

    @staticmethod
    def _active_is_cuda(impl) -> bool:
        """Ask the live ONNX session which provider it actually got."""
        model = getattr(impl, "model", None)
        sess = getattr(model, "model", None) if model is not None else None
        get = getattr(sess, "get_providers", None)
        return bool(get and "CUDAExecutionProvider" in get())

    def embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        texts = list(texts)
        impl = self._load()
        p = self.spec.provider
        if p == "fastembed":
            yield from impl.embed(texts)
        elif p == "sentence-transformers":
            for v in impl.encode(texts, normalize_embeddings=True):
                yield np.asarray(v, dtype=np.float32)
        else:  # openai
            resp = impl.embeddings.create(model=self.spec.model, input=texts)
            for item in resp.data:
                yield np.asarray(item.embedding, dtype=np.float32)


def describe() -> str:
    e = Embedder()
    return (f"{e.backend} -> {e.model_name} ({e.dim}d) on {e.device} "
            f"— {e.spec.note}")
