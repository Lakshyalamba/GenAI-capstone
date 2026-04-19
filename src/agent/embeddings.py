from __future__ import annotations

from functools import lru_cache

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings

from src.utils import VECTORSTORE_DIR


ONNXMiniLM_L6_V2.DOWNLOAD_PATH = VECTORSTORE_DIR / "embedding_cache" / ONNXMiniLM_L6_V2.MODEL_NAME


@lru_cache(maxsize=1)
def _get_default_embedding_function() -> DefaultEmbeddingFunction:
    """Reuse the local Chroma embedding backend across the app session."""
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    return DefaultEmbeddingFunction()


class LocalChromaEmbeddings(Embeddings):
    """LangChain-compatible wrapper around Chroma's local default embeddings."""

    def __init__(self) -> None:
        self._embedding_function = _get_default_embedding_function()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(value) for value in vector] for vector in self._embedding_function(texts)]

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        return [float(value) for value in self._embedding_function([text])[0]]
