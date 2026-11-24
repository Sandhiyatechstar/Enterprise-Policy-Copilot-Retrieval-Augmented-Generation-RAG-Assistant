"""
Embeddings Generation Module for Enterprise Policy Copilot.

Stage 5: Local Dense Embeddings with Sentence Transformers.

Features:
- Local execution only using the sentence-transformers library.
- Programmatic dimension resolution (no hardcoded dimensions).
- L2-normalized embeddings for direct cosine similarity evaluation.
- Robust validation against non-finite values (NaN / Inf).
- Separate chunk metadata preservation and tracking.
- Configurable batch size and execution timing telemetry.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> Any:
    """Load a local SentenceTransformer model.

    Args:
        model_name: Hugging Face model identifier or local directory path.
        device: Device to load model on ('cpu', 'cuda', etc.). If None, automatically detected.

    Returns:
        Loaded SentenceTransformer model instance.

    Raises:
        ValueError: If model_name is empty or invalid.
        RuntimeError: If model fails to load.
    """
    if not model_name or not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    # Deferred import to keep unit tests and lightweight scripts fast
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name.strip(), device=device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e


def get_embedding_dimension(model: Any) -> int:
    """Programmatically retrieve the embedding dimension from a model.

    Args:
        model: Loaded SentenceTransformer or mock model with get_sentence_embedding_dimension.

    Returns:
        Integer dimension of output embeddings.

    Raises:
        ValueError: If model is None or dimension cannot be determined.
    """
    if model is None:
        raise ValueError("Model cannot be None.")

    # 1. Standard sentence-transformers method
    if hasattr(model, "get_embedding_dimension") and callable(
        model.get_embedding_dimension
    ):
        dim = model.get_embedding_dimension()
        if isinstance(dim, int) and dim > 0:
            return dim

    if hasattr(model, "get_sentence_embedding_dimension") and callable(
        model.get_sentence_embedding_dimension
    ):
        dim = model.get_sentence_embedding_dimension()
        if isinstance(dim, int) and dim > 0:
            return dim

    # 2. Check modules or config attributes
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)

    # 3. Fallback: probe with a minimal synthetic string
    try:
        test_emb = model.encode(["test probe"], convert_to_numpy=True)
        return int(test_emb.shape[1])
    except Exception as err:
        raise ValueError(
            f"Unable to determine embedding dimension from model: {err}"
        ) from err


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding matrix along axis 1 for cosine similarity.

    With L2-normalized vectors (||v||_2 = 1.0), the dot product / inner product
    directly equals the cosine similarity:
        cosine_sim(u, v) = dot(u / ||u||, v / ||v||)

    Args:
        embeddings: 2D NumPy array of shape (num_samples, dim).

    Returns:
        L2-normalized 2D NumPy array of shape (num_samples, dim).

    Raises:
        ValueError: If array is not 2D, contains NaN/Inf, or has zero norm rows.
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding array (num_samples, dim), got {embeddings.ndim}D array."
        )

    if embeddings.shape[0] == 0:
        return embeddings.astype(np.float32)

    validate_embeddings(embeddings)

    # Compute Euclidean (L2) norms along each row
    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

    # Safeguard against zero vectors
    if np.any(norms == 0):
        # Epsilon smoothing for zero norm to prevent divide-by-zero
        norms = np.where(norms == 0, 1e-12, norms)

    normalized = embeddings / norms
    return normalized.astype(np.float32)


def validate_embeddings(
    embeddings: np.ndarray,
    expected_dim: Optional[int] = None,
) -> None:
    """Validate that an embedding array is finite, 2D, and optionally matches expected dimension.

    Args:
        embeddings: NumPy array to validate.
        expected_dim: Expected column dimension if known.

    Raises:
        ValueError: If embeddings contain NaN, Inf, non-2D shape, or dimension mismatch.
    """
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(f"Embeddings must be a NumPy array, got {type(embeddings)}.")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D array (n_samples, dim), got ndim={embeddings.ndim} with shape {embeddings.shape}."
        )

    if embeddings.shape[0] > 0:
        if not np.isfinite(embeddings).all():
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            raise ValueError(
                f"Embedding array contains non-finite values (NaN count: {nan_count}, Inf count: {inf_count})."
            )

        if expected_dim is not None and embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}."
            )


def embed_texts(
    texts: List[str],
    model: Any,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress_bar: bool = False,
) -> Tuple[np.ndarray, float]:
    """Generate dense embeddings for a list of raw text strings using a local model.

    Args:
        texts: List of input text strings.
        model: Loaded SentenceTransformer model or compatible mock.
        batch_size: Batch size for model inference. Must be > 0.
        normalize: If True, L2-normalizes vectors so inner products equal cosine similarities.
        show_progress_bar: Whether to show inference progress bar.

    Returns:
        Tuple of:
            - embeddings: 2D float32 NumPy array of shape (len(texts), embedding_dim).
            - elapsed_time: Floating point seconds spent during generation.

    Raises:
        ValueError: If inputs or batch_size are invalid or if model fails.
    """
    if not isinstance(texts, list):
        raise ValueError(f"texts must be a list of strings, got {type(texts)}.")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

    if model is None:
        raise ValueError("model cannot be None.")

    dim = get_embedding_dimension(model)

    if len(texts) == 0:
        empty_arr = np.empty((0, dim), dtype=np.float32)
        return empty_arr, 0.0

    # Ensure all elements are strings
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise ValueError(f"All texts must be strings. Element at index {i} is {type(t)}.")

    start_time = time.perf_counter()

    # Call model encode
    raw_emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We handle normalization explicitly for precision and testing
    )

    elapsed_time = time.perf_counter() - start_time

    if not isinstance(raw_emb, np.ndarray):
        raw_emb = np.array(raw_emb, dtype=np.float32)
    else:
        raw_emb = raw_emb.astype(np.float32)

    validate_embeddings(raw_emb, expected_dim=dim)

    if normalize:
        processed_emb = normalize_embeddings(raw_emb)
    else:
        processed_emb = raw_emb

    return processed_emb, elapsed_time


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model: Any,
    text_key: str = "chunk_text",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress_bar: bool = False,
) -> Dict[str, Any]:
    """Generate dense embeddings for a list of chunk dictionaries, keeping metadata intact.

    Args:
        chunks: List of chunk dictionaries produced by any chunking strategy.
        model: Loaded SentenceTransformer model.
        text_key: Dictionary key containing chunk text. Default 'chunk_text'.
        batch_size: Batch size for model inference.
        normalize: Whether to apply L2 normalization.
        show_progress_bar: Whether to display progress bar.

    Returns:
        Dictionary containing:
            - 'embeddings': 2D NumPy array of shape (num_chunks, dim).
            - 'metadata': List of chunk dictionaries (original metadata preserved separately).
            - 'chunk_ids': List of chunk ID strings.
            - 'model_dim': Integer dimension.
            - 'num_chunks': Integer count of chunks embedded.
            - 'elapsed_time': Seconds taken to generate embeddings.
            - 'normalized': Boolean flag indicating if vectors are L2-normalized.

    Raises:
        ValueError: If input is invalid or chunks lack the required text_key.
    """
    if not isinstance(chunks, list):
        raise ValueError(f"chunks must be a list of dictionaries, got {type(chunks)}.")

    if model is None:
        raise ValueError("model cannot be None.")

    dim = get_embedding_dimension(model)

    if len(chunks) == 0:
        return {
            "embeddings": np.empty((0, dim), dtype=np.float32),
            "metadata": [],
            "chunk_ids": [],
            "model_dim": dim,
            "num_chunks": 0,
            "elapsed_time": 0.0,
            "normalized": normalize,
        }

    texts: List[str] = []
    chunk_ids: List[str] = []
    preserved_metadata: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk at index {i} is not a dictionary.")
        if text_key not in chunk:
            raise ValueError(f"Chunk at index {i} does not contain required text key '{text_key}'.")

        text_val = chunk[text_key]
        if not isinstance(text_val, str):
            raise ValueError(f"Chunk text at index {i} must be a string, got {type(text_val)}.")

        texts.append(text_val)
        cid = str(chunk.get("chunk_id", f"chunk_{i}"))
        chunk_ids.append(cid)
        # Deep copy-like dict preservation
        preserved_metadata.append(dict(chunk))

    embeddings, elapsed = embed_texts(
        texts=texts,
        model=model,
        batch_size=batch_size,
        normalize=normalize,
        show_progress_bar=show_progress_bar,
    )

    return {
        "embeddings": embeddings,
        "metadata": preserved_metadata,
        "chunk_ids": chunk_ids,
        "model_dim": dim,
        "num_chunks": len(chunks),
        "elapsed_time": elapsed,
        "normalized": normalize,
    }
