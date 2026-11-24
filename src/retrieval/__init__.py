"""Retrieval module for Enterprise Policy Copilot."""
from src.retrieval.chunking import (
    fixed_size_chunk,
    chunk_policy_section,
    chunk_policy_dataset,
    section_aware_chunk,
    chunk_policy_section_section_aware,
    chunk_policy_dataset_section_aware,
    split_into_sentences,
    sentence_window_chunk,
    chunk_policy_section_sentence_window,
    chunk_policy_dataset_sentence_window,
)
from src.retrieval.embeddings import (
    load_embedding_model,
    get_embedding_dimension,
    normalize_embeddings,
    validate_embeddings,
    embed_texts,
    embed_chunks,
)
from src.retrieval.vector_store import (
    FAISSVectorStore,
    create_vector_store,
)

__all__ = [
    # Chunking
    "fixed_size_chunk",
    "chunk_policy_section",
    "chunk_policy_dataset",
    "section_aware_chunk",
    "chunk_policy_section_section_aware",
    "chunk_policy_dataset_section_aware",
    "split_into_sentences",
    "sentence_window_chunk",
    "chunk_policy_section_sentence_window",
    "chunk_policy_dataset_sentence_window",
    # Embeddings
    "load_embedding_model",
    "get_embedding_dimension",
    "normalize_embeddings",
    "validate_embeddings",
    "embed_texts",
    "embed_chunks",
    # Vector Store
    "FAISSVectorStore",
    "create_vector_store",
]
