"""
FAISS Vector Indexing & Search Module for Enterprise Policy Copilot.

Stage 5: Local Vector Store & Retrieval with FAISS.

Features:
- Local execution only with faiss-cpu.
- Uses IndexFlatIP (Inner Product) with L2-normalized embeddings for exact cosine similarity.
- Explicit and reliable mapping between FAISS vector row positions and chunk metadata.
- Top-k nearest neighbor semantic search with score and metadata extraction.
- Validation of dimensional consistency, non-empty index, and bounds.
- Local serialization (saving and loading) of FAISS index and metadata sidecars.
- Zero external network dependencies or external databases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

from src.retrieval.embeddings import normalize_embeddings, validate_embeddings


class FAISSVectorStore:
    """FAISS-backed local vector store maintaining explicit metadata mappings."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "FlatIP",
    ) -> None:
        """Initialize an empty FAISS vector store.

        Args:
            dimension: Dimensionality of vectors to store (must match model output).
            index_type: Index type ('FlatIP' for cosine similarity / inner product).

        Raises:
            ValueError: If dimension <= 0 or unsupported index_type.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}.")

        if index_type != "FlatIP":
            raise ValueError(
                f"Unsupported index_type '{index_type}'. Only 'FlatIP' is supported for cosine similarity."
            )

        self.dimension: int = dimension
        self.index_type: str = index_type

        # Initialize IndexFlatIP
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)

        # Explicit mapping: index row position -> metadata dict
        self.metadata_store: List[Dict[str, Any]] = []

        # Explicit mapping: index row position -> chunk_id string
        self.id_store: List[str] = []

    @property
    def total_vectors(self) -> int:
        """Return the current count of indexed vectors."""
        return self.index.ntotal

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk_ids: Optional[List[str]] = None,
        normalize: bool = False,
    ) -> int:
        """Add vectors to the FAISS index along with matching metadata and chunk IDs.

        Args:
            embeddings: 2D float32 NumPy array of shape (N, dimension).
            metadata: Optional list of N metadata dictionaries.
            chunk_ids: Optional list of N unique chunk IDs.
            normalize: If True, normalizes vectors before adding (if not already normalized).

        Returns:
            Number of vectors added in this call.

        Raises:
            ValueError: If dimensions mismatch, array is invalid, or counts do not match.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = embeddings.astype(np.float32)

        validate_embeddings(embeddings, expected_dim=self.dimension)

        n_vectors = embeddings.shape[0]
        if n_vectors == 0:
            return 0

        if metadata is not None and len(metadata) != n_vectors:
            raise ValueError(
                f"Metadata count ({len(metadata)}) does not match vector count ({n_vectors})."
            )

        if chunk_ids is not None and len(chunk_ids) != n_vectors:
            raise ValueError(
                f"Chunk ID count ({len(chunk_ids)}) does not match vector count ({n_vectors})."
            )

        if normalize:
            embeddings = normalize_embeddings(embeddings)

        # Ensure C-contiguous array for FAISS C++ backend
        if not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings)

        # Add vectors to FAISS index
        self.index.add(embeddings)

        # Store metadata mappings at the corresponding row positions
        for i in range(n_vectors):
            item_meta = dict(metadata[i]) if metadata is not None else {}
            cid = str(chunk_ids[i]) if chunk_ids is not None else str(item_meta.get("chunk_id", f"chunk_{len(self.id_store)}"))

            self.metadata_store.append(item_meta)
            self.id_store.append(cid)

        return n_vectors

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        normalize_query: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search the index for the top-k nearest neighbors to a query vector.

        Args:
            query_vector: 1D or 2D NumPy array representing the query embedding.
            k: Number of nearest neighbors to retrieve. Must be > 0.
            normalize_query: Whether to L2-normalize query vector for cosine similarity.

        Returns:
            List of result dictionaries sorted by similarity score (descending), each with:
                - 'chunk_id': String chunk identifier
                - 'score': Float similarity score (cosine similarity if normalized)
                - 'rank': 1-based rank integer
                - 'metadata': Preserved chunk metadata dictionary

        Raises:
            ValueError: If index is empty, k <= 0, query is invalid, or dimension mismatches.
        """
        if self.total_vectors == 0:
            return []

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        else:
            query_vector = query_vector.astype(np.float32)

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        validate_embeddings(query_vector, expected_dim=self.dimension)

        if normalize_query:
            query_vector = normalize_embeddings(query_vector)

        if not query_vector.flags["C_CONTIGUOUS"]:
            query_vector = np.ascontiguousarray(query_vector)

        # Clamp k to available vectors
        actual_k = min(k, self.total_vectors)

        # Execute FAISS search
        # D: distances / similarity scores (shape: 1, actual_k)
        # I: indices / vector row positions (shape: 1, actual_k)
        scores, indices = self.index.search(query_vector, actual_k)

        results: List[Dict[str, Any]] = []
        for rank_idx, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                # FAISS placeholder when fewer vectors are found than requested
                continue

            int_idx = int(idx)
            chunk_id = self.id_store[int_idx]
            meta = dict(self.metadata_store[int_idx])

            results.append({
                "rank": rank_idx,
                "chunk_id": chunk_id,
                "score": float(score),
                "metadata": meta,
            })

        return results

    def save(self, directory: Union[str, Path], prefix: str = "index") -> Tuple[str, str]:
        """Save FAISS index binary and metadata sidecar JSON to a local directory.

        Args:
            directory: Directory path where index and metadata will be stored.
            prefix: Filename prefix (e.g. 'fixed_size_minilm').

        Returns:
            Tuple of (index_file_path, metadata_file_path).
        """
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / f"{prefix}.faiss"
        meta_file = target_dir / f"{prefix}_metadata.json"

        # Write FAISS binary index
        faiss.write_index(self.index, str(index_file))

        # Write metadata sidecar
        sidecar_data = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": self.total_vectors,
            "id_store": self.id_store,
            "metadata_store": self.metadata_store,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(sidecar_data, f, indent=2, ensure_ascii=False)

        return str(index_file), str(meta_file)

    @classmethod
    def load(cls, directory: Union[str, Path], prefix: str = "index") -> "FAISSVectorStore":
        """Load a saved FAISS index binary and its metadata sidecar.

        Args:
            directory: Directory path where files are located.
            prefix: Filename prefix.

        Returns:
            Reconstituted FAISSVectorStore instance.

        Raises:
            FileNotFoundError: If index or metadata file does not exist.
            ValueError: If metadata sidecar is malformed.
        """
        target_dir = Path(directory)
        index_file = target_dir / f"{prefix}.faiss"
        meta_file = target_dir / f"{prefix}_metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata sidecar file not found: {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            sidecar_data = json.load(f)

        dimension = int(sidecar_data["dimension"])
        index_type = sidecar_data.get("index_type", "FlatIP")

        store = cls(dimension=dimension, index_type=index_type)
        store.index = faiss.read_index(str(index_file))
        store.id_store = list(sidecar_data["id_store"])
        store.metadata_store = list(sidecar_data["metadata_store"])

        if store.index.ntotal != len(store.metadata_store):
            raise ValueError(
                f"Mismatch in loaded store: index has {store.index.ntotal} vectors, "
                f"but metadata sidecar has {len(store.metadata_store)} records."
            )

        return store


def create_vector_store(
    embeddings: np.ndarray,
    metadata: Optional[List[Dict[str, Any]]] = None,
    chunk_ids: Optional[List[str]] = None,
    dimension: Optional[int] = None,
    normalize: bool = False,
) -> FAISSVectorStore:
    """Helper function to create and populate a FAISSVectorStore in one step.

    Args:
        embeddings: 2D NumPy array of shape (N, dim).
        metadata: Optional list of N chunk metadata dictionaries.
        chunk_ids: Optional list of N unique chunk IDs.
        dimension: Explicit dimension. If None, inferred from embeddings.shape[1].
        normalize: Whether to apply L2 normalization when adding vectors.

    Returns:
        Populated FAISSVectorStore instance.

    Raises:
        ValueError: If embeddings array is invalid or dimensions cannot be resolved.
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D.")

    if dimension is None:
        dimension = int(embeddings.shape[1])

    store = FAISSVectorStore(dimension=dimension, index_type="FlatIP")

    if embeddings.shape[0] > 0:
        store.add_vectors(
            embeddings=embeddings,
            metadata=metadata,
            chunk_ids=chunk_ids,
            normalize=normalize,
        )

    return store
