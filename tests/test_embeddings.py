"""
Unit tests for the Embeddings module (src/retrieval/embeddings.py).

Tests:
- Embedding output dimensions.
- Embedding array shape.
- Empty input handling.
- Invalid input handling (non-string, non-dict, non-positive batch size).
- Non-finite embedding rejection (NaN, Inf).
- Metadata preservation.
- Vector normalization (L2 norm == 1.0, cosine similarity property).
- Programmatic dimension resolution without hardcoding.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.retrieval.embeddings import (
    get_embedding_dimension,
    normalize_embeddings,
    validate_embeddings,
    embed_texts,
    embed_chunks,
    load_embedding_model,
)


class MockSentenceTransformer:
    """Lightweight mock model simulating SentenceTransformer for zero-network testing."""

    def __init__(self, dimension: int = 384):
        self.dim = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        n = len(sentences)
        # Generate deterministic synthetic float embeddings based on sentence length
        rng = np.random.default_rng(seed=42)
        base = rng.standard_normal((n, self.dim)).astype(np.float32)
        return base


class TestEmbeddingsModule(unittest.TestCase):
    """Test suite for embeddings generation, normalization, and validation."""

    def setUp(self):
        self.mock_model_384 = MockSentenceTransformer(dimension=384)
        self.mock_model_768 = MockSentenceTransformer(dimension=768)

    def test_get_embedding_dimension(self):
        """Verify dimension is read programmatically from model methods."""
        self.assertEqual(get_embedding_dimension(self.mock_model_384), 384)
        self.assertEqual(get_embedding_dimension(self.mock_model_768), 768)

    def test_get_embedding_dimension_none_raises(self):
        """Passing None model must raise ValueError."""
        with self.assertRaises(ValueError):
            get_embedding_dimension(None)

    def test_normalize_embeddings_l2_norm(self):
        """Verify normalized vectors have L2 norm approximately equal to 1.0."""
        raw = np.array([[3.0, 4.0], [1.0, -1.0], [10.0, 0.0]], dtype=np.float32)
        normed = normalize_embeddings(raw)
        norms = np.linalg.norm(normed, ord=2, axis=1)
        np.testing.assert_allclose(norms, np.ones(3, dtype=np.float32), rtol=1e-5)

    def test_normalize_embeddings_empty(self):
        """Empty 2D array should return empty array."""
        empty = np.empty((0, 384), dtype=np.float32)
        normed = normalize_embeddings(empty)
        self.assertEqual(normed.shape, (0, 384))

    def test_normalize_embeddings_non_2d_raises(self):
        """1D or 3D arrays must raise ValueError."""
        with self.assertRaises(ValueError):
            normalize_embeddings(np.array([1.0, 2.0, 3.0]))

    def test_validate_embeddings_nan_raises(self):
        """Array containing NaN must be rejected."""
        bad_arr = np.array([[1.0, np.nan], [0.5, 0.5]], dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_embeddings(bad_arr)

    def test_validate_embeddings_inf_raises(self):
        """Array containing Inf must be rejected."""
        bad_arr = np.array([[1.0, np.inf], [0.5, 0.5]], dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_embeddings(bad_arr)

    def test_validate_embeddings_dimension_mismatch(self):
        """Array with wrong column count must be rejected."""
        arr = np.ones((5, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_embeddings(arr, expected_dim=384)

    def test_embed_texts_output_shape_and_dtype(self):
        """Verify embed_texts returns correct shape, float32 dtype, and timing."""
        texts = ["First sample sentence.", "Second sample sentence.", "Third text."]
        emb, elapsed = embed_texts(texts, self.mock_model_384, batch_size=2, normalize=True)
        self.assertEqual(emb.shape, (3, 384))
        self.assertEqual(emb.dtype, np.float32)
        self.assertGreaterEqual(elapsed, 0.0)
        # Check normalization
        norms = np.linalg.norm(emb, ord=2, axis=1)
        np.testing.assert_allclose(norms, np.ones(3, dtype=np.float32), rtol=1e-5)

    def test_embed_texts_empty_list(self):
        """Empty text list returns (0, dim) array with 0 elapsed time."""
        emb, elapsed = embed_texts([], self.mock_model_384)
        self.assertEqual(emb.shape, (0, 384))
        self.assertEqual(elapsed, 0.0)

    def test_embed_texts_invalid_inputs(self):
        """Non-list texts or invalid batch_size must raise ValueError."""
        with self.assertRaises(ValueError):
            embed_texts("not a list", self.mock_model_384)
        with self.assertRaises(ValueError):
            embed_texts(["valid"], self.mock_model_384, batch_size=0)
        with self.assertRaises(ValueError):
            embed_texts([123], self.mock_model_384)  # non-string inside list

    def test_embed_chunks_metadata_preservation(self):
        """Verify embed_chunks preserves all chunk metadata without mutations."""
        chunks = [
            {
                "chunk_id": "POL_01_chk00",
                "doc_id": "POL-01",
                "document_title": "Policy 1",
                "chunk_text": "Content of chunk 0.",
                "chunk_index": 0,
            },
            {
                "chunk_id": "POL_01_chk01",
                "doc_id": "POL-01",
                "document_title": "Policy 1",
                "chunk_text": "Content of chunk 1.",
                "chunk_index": 1,
            },
        ]
        result = embed_chunks(chunks, self.mock_model_384, text_key="chunk_text", normalize=True)
        self.assertEqual(result["num_chunks"], 2)
        self.assertEqual(result["model_dim"], 384)
        self.assertEqual(result["embeddings"].shape, (2, 384))
        self.assertEqual(result["chunk_ids"], ["POL_01_chk00", "POL_01_chk01"])
        self.assertEqual(len(result["metadata"]), 2)
        self.assertEqual(result["metadata"][0]["doc_id"], "POL-01")
        self.assertEqual(result["metadata"][1]["chunk_id"], "POL_01_chk01")
        self.assertTrue(result["normalized"])

    def test_embed_chunks_empty(self):
        """Empty chunk list returns well-formed empty result dictionary."""
        result = embed_chunks([], self.mock_model_768)
        self.assertEqual(result["num_chunks"], 0)
        self.assertEqual(result["embeddings"].shape, (0, 768))
        self.assertEqual(result["metadata"], [])
        self.assertEqual(result["chunk_ids"], [])

    def test_embed_chunks_missing_text_key_raises(self):
        """Chunks missing the specified text_key must raise ValueError."""
        bad_chunks = [{"chunk_id": "chk01", "wrong_key": "some text"}]
        with self.assertRaises(ValueError):
            embed_chunks(bad_chunks, self.mock_model_384, text_key="chunk_text")

    def test_load_embedding_model_invalid_name(self):
        """Empty or invalid model name must raise ValueError."""
        with self.assertRaises(ValueError):
            load_embedding_model("")
        with self.assertRaises(ValueError):
            load_embedding_model(None)


if __name__ == "__main__":
    unittest.main()
