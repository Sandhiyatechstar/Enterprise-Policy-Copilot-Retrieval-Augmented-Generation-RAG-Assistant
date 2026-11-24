"""
Enterprise Policy Copilot - Text Chunking Module
------------------------------------------------
This module provides reusable text segmentation functions designed for
Retrieval-Augmented Generation (RAG) pipelines.

Supported Chunking Strategies:
1. Fixed-Size Character Chunking (`fixed_size_chunk` - Stage 2):
   Slices text using a fixed-width character sliding window with configurable overlap.
2. Section-Aware Chunking (`section_aware_chunk` - Stage 3):
   Hierarchical segmentation respecting structural boundaries (paragraphs -> sentences -> words)
   to ensure that natural policy rules, sentences, and paragraphs are never arbitrarily severed.
3. Sentence-Window Chunking (`sentence_window_chunk` - Stage 4):
   Splits text into sentences, then groups consecutive sentences into sliding windows with
   configurable window size and step size. Ideal for embedding retrieval + generation context.

Key Features:
- Process each policy section independently (never mixes different documents or sections).
- Complete metadata lineage inheritance (`doc_id`, `document_title`, `section_heading`, etc.).
- Deterministic chunk identifiers.
- Character-level boundary tracking (`start_char`, `end_char`) directly indexing original text.
- Full input validation and infinite loop prevention.
"""

from typing import Any, Dict, List, Optional, Tuple
import re


def _create_deterministic_chunk_id(
    doc_id: str,
    section_heading: str,
    chunk_index: int,
    prefix: str = "chk",
) -> str:
    """
    Generate a deterministic, clean chunk identifier.

    Example:
        doc_id="POL-AL-001", section_heading="Leave Entitlement", chunk_index=0, prefix="winchk"
        -> "POL-AL-001_leave_entitlement_winchk00"
    """
    # Normalize section heading: lowercased, non-alphanumeric replaced with underscore
    clean_heading = re.sub(r"[^a-zA-Z0-9]+", "_", section_heading.strip().lower()).strip("_")
    if not clean_heading:
        clean_heading = "sec"
    return f"{doc_id}_{clean_heading}_{prefix}{chunk_index:02d}"


# =============================================================================
# STRATEGY 1: FIXED-SIZE CHARACTER CHUNKING (Stage 2)
# =============================================================================

def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size character chunks with a configurable sliding window overlap.

    Parameters:
        text (str):
            The raw text content to split into chunks.
        chunk_size (int, default=500):
            Maximum number of characters in each chunk. Must be a positive integer (> 0).
        chunk_overlap (int, default=100):
            Number of characters shared between consecutive chunks.
            Must be non-negative (>= 0) and strictly less than chunk_size.
        metadata (dict, optional):
            Parent document metadata dictionary containing fields such as:
            'doc_id', 'document_title', 'section_heading', 'version', 'effective_date'.

    Returns:
        List[Dict[str, Any]]:
            A list of chunk dictionaries in their original reading order.
    """
    # 1. Parameter Validation
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer (> 0), got {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative (>= 0), got {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}). "
            "Otherwise, the sliding window cannot advance and would cause an infinite loop."
        )

    # 2. Handle Empty or Whitespace-Only Text
    if not isinstance(text, str) or not text.strip():
        return []

    # 3. Setup Metadata Defaults
    meta = metadata.copy() if metadata else {}
    doc_id = str(meta.get("doc_id", "DOC-000"))
    doc_title = str(meta.get("document_title", ""))
    sec_heading = str(meta.get("section_heading", ""))
    version = str(meta.get("version", ""))
    effective_date = str(meta.get("effective_date", ""))

    # 4. Sliding Window Chunking Logic
    step_size = chunk_size - chunk_overlap
    text_length = len(text)
    chunks = []
    chunk_index = 0
    start_char = 0

    while start_char < text_length:
        end_char = min(start_char + chunk_size, text_length)
        chunk_text = text[start_char:end_char]

        chunk_id = _create_deterministic_chunk_id(doc_id, sec_heading, chunk_index, prefix="chk")

        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "document_title": doc_title,
            "section_heading": sec_heading,
            "version": version,
            "effective_date": effective_date,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "start_char": start_char,
            "end_char": end_char,
        })

        chunk_index += 1

        if end_char == text_length:
            break

        start_char += step_size

    return chunks


def chunk_policy_section(
    section_record: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Convenience wrapper to chunk a single policy section dictionary using fixed-size chunking."""
    text = section_record.get("section_content", "")
    metadata = {
        "doc_id": section_record.get("doc_id", ""),
        "document_title": section_record.get("document_title", ""),
        "section_heading": section_record.get("section_heading", ""),
        "version": section_record.get("version", ""),
        "effective_date": section_record.get("effective_date", ""),
    }
    return fixed_size_chunk(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )


def chunk_policy_dataset(
    dataset: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Chunk an entire list of policy section records independently using fixed-size chunking."""
    all_chunks = []
    for section in dataset:
        chunks = chunk_policy_section(
            section_record=section,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


# =============================================================================
# STRATEGY 2: SECTION-AWARE (HIERARCHICAL) CHUNKING (Stage 3)
# =============================================================================

def _decompose_to_atomic_units(text: str, chunk_size: int) -> List[Tuple[int, int]]:
    """
    Decompose text hierarchically into atomic units (start_char, end_char) such that
    every unit satisfies: (end_char - start_char) <= chunk_size.
    """
    units: List[Tuple[int, int]] = []

    # Level 1: Paragraph decomposition
    if "\n\n" in text:
        para_pattern = re.compile(r"\S.*?(?=\n\s*\n|\Z)", re.DOTALL)
    elif "\n" in text:
        para_pattern = re.compile(r"\S.*?(?=\n|\Z)", re.DOTALL)
    else:
        para_pattern = re.compile(r"\S.*", re.DOTALL)

    for p_match in para_pattern.finditer(text):
        p_start, p_end = p_match.start(), p_match.end()
        p_text = p_match.group()
        if not p_text.strip():
            continue

        if len(p_text) <= chunk_size:
            units.append((p_start, p_end))
            continue

        # Level 2: Sentence decomposition
        sent_pattern = re.compile(r"\S.*?(?:(?<=[.!?])(?=\s+|\Z)|\Z)", re.DOTALL)
        for s_match in sent_pattern.finditer(p_text):
            s_start = p_start + s_match.start()
            s_end = p_start + s_match.end()
            s_text = s_match.group()
            if not s_text.strip():
                continue

            if len(s_text) <= chunk_size:
                units.append((s_start, s_end))
                continue

            # Level 3: Word boundary fallback
            word_pattern = re.compile(r"\S+")
            words = list(word_pattern.finditer(s_text))
            if not words:
                continue

            w_idx = 0
            while w_idx < len(words):
                w_start = s_start + words[w_idx].start()
                w_end = s_start + words[w_idx].end()

                # Level 4: Character slice fallback
                if (w_end - w_start) > chunk_size:
                    for char_offset in range(w_start, w_end, chunk_size):
                        char_end = min(char_offset + chunk_size, w_end)
                        units.append((char_offset, char_end))
                    w_idx += 1
                    continue

                packed_end = w_end
                next_w = w_idx + 1
                while next_w < len(words):
                    cand_end = s_start + words[next_w].end()
                    if (cand_end - w_start) <= chunk_size:
                        packed_end = cand_end
                        next_w += 1
                    else:
                        break
                units.append((w_start, packed_end))
                w_idx = next_w

    return units


def section_aware_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Split a policy section into chunks respecting natural paragraph and sentence boundaries.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer (> 0), got {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative (>= 0), got {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}). "
            "Otherwise, the window cannot advance and would cause an infinite loop."
        )

    if not isinstance(text, str) or not text.strip():
        return []

    meta = metadata.copy() if metadata else {}
    doc_id = str(meta.get("doc_id", "DOC-000"))
    doc_title = str(meta.get("document_title", ""))
    sec_heading = str(meta.get("section_heading", ""))
    version = str(meta.get("version", ""))
    effective_date = str(meta.get("effective_date", ""))

    units = _decompose_to_atomic_units(text=text, chunk_size=chunk_size)
    if not units:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_index = 0
    i = 0
    n = len(units)

    while i < n:
        chunk_start = units[i][0]
        j = i
        while j < n:
            cand_end = units[j][1]
            if (cand_end - chunk_start) <= chunk_size:
                j += 1
            else:
                break

        end_unit_idx = max(i, j - 1)
        chunk_end = units[end_unit_idx][1]
        chunk_text = text[chunk_start:chunk_end]

        chunk_id = _create_deterministic_chunk_id(doc_id, sec_heading, chunk_index, prefix="secchk")

        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "document_title": doc_title,
            "section_heading": sec_heading,
            "version": version,
            "effective_date": effective_date,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "start_char": chunk_start,
            "end_char": chunk_end,
        })
        chunk_index += 1

        if j >= n:
            break

        if chunk_overlap > 0:
            k = end_unit_idx
            overlap_start = i + 1
            while k > i:
                cand_overlap = chunk_end - units[k][0]
                if cand_overlap <= chunk_overlap:
                    overlap_start = k
                    k -= 1
                else:
                    break
            i = max(i + 1, overlap_start)
        else:
            i = j

    return chunks


def chunk_policy_section_section_aware(
    section_record: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Convenience wrapper to chunk a single policy section using section-aware chunking."""
    text = section_record.get("section_content", "")
    metadata = {
        "doc_id": section_record.get("doc_id", ""),
        "document_title": section_record.get("document_title", ""),
        "section_heading": section_record.get("section_heading", ""),
        "version": section_record.get("version", ""),
        "effective_date": section_record.get("effective_date", ""),
    }
    return section_aware_chunk(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )


def chunk_policy_dataset_section_aware(
    dataset: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Chunk an entire list of policy section records independently using section-aware chunking."""
    all_chunks = []
    for section in dataset:
        chunks = chunk_policy_section_section_aware(
            section_record=section,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


# =============================================================================
# STRATEGY 3: SENTENCE-WINDOW CHUNKING (Stage 4)
# =============================================================================

# Common English abbreviations ending in '.' that should not trigger false sentence breaks
KNOWN_ABBREVIATIONS = {
    "e.g.", "i.e.", "vs.", "etc.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "dept.", "inc.", "ltd.", "co.", "approx.", "no.", "fig."
}


def split_into_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Split text into individual sentences while preserving exact character offsets
    and punctuation. Avoids splitting on common abbreviations and decimal numbers.

    Limitations:
    - Highly unconventional abbreviations or ambiguous dot usages (e.g., initialisms
      not in the known list followed by a capitalized word) may occasionally cause
      a split.
    - Ellipses (...) immediately preceding another sentence are treated as terminators.

    Parameters:
        text (str): Input text string.

    Returns:
        List of (start_char, end_char, sentence_text) tuples referencing `text`.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Find candidate punctuation marks followed by whitespace or end of string
    matches = list(re.finditer(r"([.!?]+)(?:\s+|\Z)", text))
    boundaries: List[int] = []

    for m in matches:
        punc = m.group(1)
        punc_end = m.end(1)

        # Question marks and exclamation points unambiguously terminate sentences
        if "?" in punc or "!" in punc:
            boundaries.append(punc_end)
            continue

        # For period terminators, check if the preceding token is a known abbreviation
        preceding = text[:punc_end].lower()
        word_m = re.search(r"\b([a-z]+(?:\.[a-z]+)*\.)\Z", preceding)
        if word_m and word_m.group(1) in KNOWN_ABBREVIATIONS:
            # Skip false boundary on abbreviation
            continue

        boundaries.append(punc_end)

    sentences: List[Tuple[int, int, str]] = []

    # Find the start of the first non-whitespace character
    lead_m = re.search(r"\S", text)
    if not lead_m:
        return []

    curr_start = lead_m.start()

    for b in boundaries:
        if b > curr_start:
            sent_text = text[curr_start:b]
            if sent_text.strip():
                sentences.append((curr_start, b, sent_text))
            # Advance to the start of the next sentence (skip intervening whitespace)
            next_m = re.search(r"\S", text[b:])
            if next_m:
                curr_start = b + next_m.start()
            else:
                curr_start = len(text)
                break

    # Handle remaining text after the final punctuation boundary (e.g. trailing fragment)
    if curr_start < len(text) and text[curr_start:].strip():
        # Find end of last non-whitespace character
        end_pos = len(text.rstrip())
        sentences.append((curr_start, end_pos, text[curr_start:end_pos]))

    return sentences


def sentence_window_chunk(
    text: str,
    window_size: int = 3,
    step_size: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Split a policy section into sliding sentence windows.

    Algorithm:
    1. Parse text into individual sentence spans using punctuation and abbreviation detection.
    2. Form windows of `window_size` consecutive sentences, stepping forward by `step_size`.
    3. Handles short sections (< window_size sentences) by returning 1 window containing all sentences.
    4. Exact offsets: `start_char` is the start of the first sentence in the window,
       `end_char` is the end of the last sentence in the window.
       Guarantee: `text[start_char:end_char] == chunk_text` including exact original spacing.
    5. Fully deterministic chunk IDs: `{doc_id}_{clean_heading}_winchk{chunk_index:02d}`.

    Parameters:
        text (str):
            The policy section text to chunk.
        window_size (int, default=3):
            Number of sentences in each window. Must be a positive integer (> 0).
        step_size (int, default=1):
            Number of sentences to advance between consecutive windows.
            Must be a positive integer (> 0).
        metadata (dict, optional):
            Parent document metadata dictionary containing fields such as:
            'doc_id', 'document_title', 'section_heading', 'version', 'effective_date'.

    Returns:
        List[Dict[str, Any]]:
            A list of chunk dictionaries in reading order.

    Raises:
        TypeError: If window_size or step_size are not integers.
        ValueError: If window_size <= 0 or step_size <= 0.
    """
    # -------------------------------------------------------------
    # 1. Parameter Validation
    # -------------------------------------------------------------
    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise TypeError(f"window_size must be an integer, got {type(window_size).__name__}")

    if not isinstance(step_size, int) or isinstance(step_size, bool):
        raise TypeError(f"step_size must be an integer, got {type(step_size).__name__}")

    if window_size <= 0:
        raise ValueError(f"window_size must be a positive integer (> 0), got {window_size}")

    if step_size <= 0:
        raise ValueError(f"step_size must be a positive integer (> 0), got {step_size}")

    # -------------------------------------------------------------
    # 2. Handle Empty or Whitespace-Only Text
    # -------------------------------------------------------------
    if not isinstance(text, str) or not text.strip():
        return []

    # -------------------------------------------------------------
    # 3. Setup Metadata Defaults
    # -------------------------------------------------------------
    meta = metadata.copy() if metadata else {}
    doc_id = str(meta.get("doc_id", "DOC-000"))
    doc_title = str(meta.get("document_title", ""))
    sec_heading = str(meta.get("section_heading", ""))
    version = str(meta.get("version", ""))
    effective_date = str(meta.get("effective_date", ""))

    # -------------------------------------------------------------
    # 4. Sentence Segmentation
    # -------------------------------------------------------------
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    num_sents = len(sentences)

    # -------------------------------------------------------------
    # 5. Sliding Window Generation
    # -------------------------------------------------------------
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0
    i = 0

    while i < num_sents:
        end_idx = min(i + window_size, num_sents)

        start_char = sentences[i][0]
        end_char = sentences[end_idx - 1][1]
        chunk_text = text[start_char:end_char]

        chunk_id = _create_deterministic_chunk_id(doc_id, sec_heading, chunk_index, prefix="winchk")

        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "document_title": doc_title,
            "section_heading": sec_heading,
            "version": version,
            "effective_date": effective_date,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "start_char": start_char,
            "end_char": end_char,
            "sentence_count": end_idx - i,
        })
        chunk_index += 1

        # If this window covers up to the final sentence, we have finished
        if end_idx == num_sents:
            break

        i += step_size

    return chunks


def chunk_policy_section_sentence_window(
    section_record: Dict[str, Any],
    window_size: int = 3,
    step_size: int = 1,
) -> List[Dict[str, Any]]:
    """Convenience wrapper to chunk a single policy section using sentence-window chunking."""
    text = section_record.get("section_content", "")
    metadata = {
        "doc_id": section_record.get("doc_id", ""),
        "document_title": section_record.get("document_title", ""),
        "section_heading": section_record.get("section_heading", ""),
        "version": section_record.get("version", ""),
        "effective_date": section_record.get("effective_date", ""),
    }
    return sentence_window_chunk(
        text=text,
        window_size=window_size,
        step_size=step_size,
        metadata=metadata,
    )


def chunk_policy_dataset_sentence_window(
    dataset: List[Dict[str, Any]],
    window_size: int = 3,
    step_size: int = 1,
) -> List[Dict[str, Any]]:
    """
    Chunk an entire list of policy section records independently using sentence-window chunking.

    Guarantees that policy sections are NEVER merged or combined across sections.
    """
    all_chunks = []
    for section in dataset:
        chunks = chunk_policy_section_sentence_window(
            section_record=section,
            window_size=window_size,
            step_size=step_size,
        )
        all_chunks.extend(chunks)
    return all_chunks
