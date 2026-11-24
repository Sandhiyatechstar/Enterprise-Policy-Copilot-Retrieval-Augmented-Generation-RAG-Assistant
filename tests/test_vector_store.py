"""
Unit tests for the FAISS Vector Store module (src/retrieval/vector_store.py).

Tests:
- FAISS index creation (IndexFlatIP).
- Vector insertion and dimension validation.
- Top-k nearest neighbor search and score ordering.
- Explicit and accurate mapping between vector rows and chunk metadata.
- Dimension mismatch rejection on add and search.
- Empty index behavior (returns empty list without crashing).
- Invalid query and invalid k parameter handling.
- Local serialization (save) and deserialization (load) consistency.
- Cosine similarity verification with synthetic vectors.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.retrieval.embeddings import normalize_embeddings
from src.retrieval.vector_store import (
    FAISSVectorStore,
    create_vector_store,
)


class TestVectorStoreModule(unittest.TestCase):
    """Test suite for FAISS vector indexing, searching, and metadata persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dim = 4

        # Create simple 4D synthetic vectors
        # v0 = [1, 0, 0, 0], v1 = [0, 1, 0, 0], v2 = [0.707, 0.707, 0, 0]
        self.vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.70710678, 0.70710678, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.metadata = [
            {"chunk_id": "c0", "title": "Section A", "content": "Alpha"},
            {"chunk_id": "c1", "title": "Section B", "content": "Beta"},
            {"chunk_id": "c2", "title": "Section C", "content": "Gamma"},
        ]
        self.chunk_ids = ["c0", "c1", "c2"]

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_invalid_dimension_raises(self):
        """Non-positive dimension or invalid index type must raise ValueError."""
        with self.assertRaises(ValueError):
            FAISSVectorStore(dimension=0)
        with self.assertRaises(ValueError):
            FAISSVectorStore(dimension=-5)
        with self.assertRaises(ValueError):
            FAISSVectorStore(dimension=4, index_type="UnsupportedType")

    def test_add_vectors_and_count(self):
        """Adding vectors updates total_vectors property accurately."""
        store = FAISSVectorStore(dimension=self.dim)
        self.assertEqual(store.total_vectors, 0)

        added = store.add_vectors(self.vectors, metadata=self.metadata, chunk_ids=self.chunk_ids)
        self.assertEqual(added, 3)
        self.assertEqual(store.total_vectors, 3)

    def test_add_vectors_empty(self):
        """Adding empty array should return 0 and keep total_vectors unchanged."""
        store = FAISSVectorStore(dimension=self.dim)
        added = store.add_vectors(np.empty((0, self.dim), dtype=np.float32))
        self.assertEqual(added, 0)
        self.assertEqual(store.total_vectors, 0)

    def test_add_vectors_dimension_mismatch_raises(self):
        """Adding vectors with mismatched columns must raise ValueError."""
        store = FAISSVectorStore(dimension=self.dim)
        wrong_dim_vectors = np.ones((3, 10), dtype=np.float32)
        with self.assertRaises(ValueError):
            store.add_vectors(wrong_dim_vectors)

    def test_add_vectors_metadata_count_mismatch_raises(self):
        """Mismatch between vector count and metadata count must raise ValueError."""
        store = FAISSVectorStore(dimension=self.dim)
        with self.assertRaises(ValueError):
            store.add_vectors(self.vectors, metadata=self.metadata[:2])

    def test_search_empty_index_returns_empty(self):
        """Searching an empty index should safely return an empty list."""
        store = FAISSVectorStore(dimension=self.dim)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, k=3)
        self.assertEqual(results, [])

    def test_search_invalid_k_raises(self):
        """k <= 0 or non-integer must raise ValueError."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_vectors(self.vectors, self.metadata, self.chunk_ids)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            store.search(query, k=0)
        with self.assertRaises(ValueError):
            store.search(query, k=-2)

    def test_search_query_dimension_mismatch_raises(self):
        """Query vector with wrong dimension must raise ValueError."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_vectors(self.vectors, self.metadata, self.chunk_ids)
        bad_query = np.ones(8, dtype=np.float32)
        with self.assertRaises(ValueError):
            store.search(bad_query, k=2)

    def test_search_nearest_neighbor_and_explicit_metadata_mapping(self):
        """Verify top-1 retrieval matches the exact vector and extracts correct metadata."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_vectors(self.vectors, metadata=self.metadata, chunk_ids=self.chunk_ids)

        # Query aligned with v0: [1, 0, 0, 0]
        query0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results0 = store.search(query0, k=3)

        self.assertEqual(len(results0), 3)
        self.assertEqual(results0[0]["chunk_id"], "c0")
        self.assertAlmostEqual(results0[0]["score"], 1.0, places=4)
        self.assertEqual(results0[0]["metadata"]["title"], "Section A")
        self.assertEqual(results0[0]["metadata"]["content"], "Alpha")

        # Query aligned with v1: [0, 1, 0, 0]
        query1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        results1 = store.search(query1, k=1)
        self.assertEqual(results1[0]["chunk_id"], "c1")
        self.assertAlmostEqual(results1[0]["score"], 1.0, places=4)
        self.assertEqual(results1[0]["metadata"]["content"], "Beta")

    def test_search_k_greater_than_total_vectors(self):
        """Requesting k > total_vectors returns exactly total_vectors results."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_vectors(self.vectors, metadata=self.metadata, chunk_ids=self.chunk_ids)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, k=10)
        self.assertEqual(len(results), 3)

    def test_save_and_load_consistency(self):
        """Saving to disk and loading preserves vectors, scores, and metadata mappings."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_vectors(self.vectors, metadata=self.metadata, chunk_ids=self.chunk_ids)

        idx_path, meta_path = store.save(self.temp_dir, prefix="test_store")
        self.assertTrue(os.path.exists(idx_path))
        self.assertTrue(os.path.exists(meta_path))

        loaded_store = FAISSVectorStore.load(self.temp_dir, prefix="test_store")
        self.assertEqual(loaded_store.dimension, self.dim)
        self.assertEqual(loaded_store.total_vectors, 3)
        self.assertEqual(loaded_store.id_store, ["c0", "c1", "c2"])

        # Run identical query on loaded store
        query = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
        results = loaded_store.search(query, k=2)
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)
        self.assertEqual(results[0]["metadata"]["title"], "Section C")

    def test_create_vector_store_helper(self):
        """Helper create_vector_store creates and populates instance properly."""
        store = create_vector_store(
            embeddings=self.vectors,
            metadata=self.metadata,
            chunk_ids=self.chunk_ids,
        )
        self.assertEqual(store.dimension, self.dim)
        self.assertEqual(store.total_vectors, 3)


if __name__ == "__main__":
    unittest.main()
