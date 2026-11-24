"""
Unit Tests for Text Chunking Module
----------------------------------
Covers:
1. TestFixedSizeChunking: Stage 2 fixed-size character chunking tests (13 tests).
2. TestSectionAwareChunking: Stage 3 hierarchical section-aware chunking tests (14 tests).
3. TestSentenceWindowChunking: Stage 4 sliding sentence window chunking tests (18 tests).
"""

import sys
from pathlib import Path
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


class TestFixedSizeChunking(unittest.TestCase):
    """Test suite for fixed_size_chunk (Stage 2)."""

    def setUp(self):
        self.sample_metadata = {
            "doc_id": "POL-TEST-001",
            "document_title": "Test Workplace Policy",
            "section_heading": "Test Section Entitlements",
            "version": "1.0",
            "effective_date": "2025-01-01",
        }

    def test_empty_string_returns_empty_list(self):
        chunks = fixed_size_chunk("", chunk_size=500, chunk_overlap=100)
        self.assertEqual(chunks, [])

    def test_whitespace_only_string_returns_empty_list(self):
        chunks = fixed_size_chunk("   \n\t   ", chunk_size=500, chunk_overlap=100)
        self.assertEqual(chunks, [])

    def test_short_text_single_chunk(self):
        text = "This is a brief policy statement regarding remote work."
        chunks = fixed_size_chunk(text, chunk_size=100, chunk_overlap=20, metadata=self.sample_metadata)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], text)
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[0]["end_char"], len(text))
        self.assertEqual(chunks[0]["chunk_index"], 0)

    def test_exact_chunk_size_single_chunk(self):
        chunk_size = 50
        text = "A" * chunk_size
        chunks = fixed_size_chunk(text, chunk_size=chunk_size, chunk_overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], text)
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[0]["end_char"], chunk_size)

    def test_longer_text_multiple_chunks(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars
        chunk_size = 100
        chunk_overlap = 25
        step_size = chunk_size - chunk_overlap  # 75

        chunks = fixed_size_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunks), 4)

        for i, chk in enumerate(chunks):
            self.assertEqual(chk["chunk_index"], i)
            expected_start = i * step_size
            expected_end = min(expected_start + chunk_size, len(text))
            self.assertEqual(chk["start_char"], expected_start)
            self.assertEqual(chk["end_char"], expected_end)
            self.assertEqual(chk["chunk_text"], text[expected_start:expected_end])

    def test_consecutive_chunk_overlap(self):
        text = "0123456789" * 30  # 300 characters
        chunk_size = 80
        chunk_overlap = 20

        chunks = fixed_size_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.assertGreaterEqual(len(chunks), 2)

        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]["chunk_text"]
            next_chunk = chunks[i + 1]["chunk_text"]
            tail_of_current = curr_chunk[-chunk_overlap:]
            head_of_next = next_chunk[:chunk_overlap]
            self.assertEqual(tail_of_current, head_of_next)

    def test_invalid_chunk_size_raises_value_error(self):
        with self.assertRaises(ValueError):
            fixed_size_chunk("Some valid policy text", chunk_size=0, chunk_overlap=0)
        with self.assertRaises(ValueError):
            fixed_size_chunk("Some valid policy text", chunk_size=-50, chunk_overlap=10)

    def test_non_integer_chunk_size_raises_type_error(self):
        with self.assertRaises(TypeError):
            fixed_size_chunk("Some valid policy text", chunk_size="500", chunk_overlap=100)  # type: ignore

    def test_negative_overlap_raises_value_error(self):
        with self.assertRaises(ValueError):
            fixed_size_chunk("Some valid policy text", chunk_size=500, chunk_overlap=-10)

    def test_overlap_greater_or_equal_to_chunk_size_raises_value_error(self):
        with self.assertRaises(ValueError):
            fixed_size_chunk("Some valid policy text", chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            fixed_size_chunk("Some valid policy text", chunk_size=100, chunk_overlap=150)

    def test_metadata_preservation_and_deterministic_ids(self):
        text = "This is a detailed paragraph explaining annual leave entitlements across all departments." * 5
        chunks = fixed_size_chunk(text, chunk_size=150, chunk_overlap=30, metadata=self.sample_metadata)

        self.assertGreater(len(chunks), 1)
        for i, chk in enumerate(chunks):
            self.assertEqual(chk["doc_id"], "POL-TEST-001")
            self.assertEqual(chk["document_title"], "Test Workplace Policy")
            self.assertEqual(chk["section_heading"], "Test Section Entitlements")
            self.assertEqual(chk["version"], "1.0")
            self.assertEqual(chk["effective_date"], "2025-01-01")
            self.assertEqual(chk["chunk_index"], i)
            expected_id = f"POL-TEST-001_test_section_entitlements_chk{i:02d}"
            self.assertEqual(chk["chunk_id"], expected_id)

    def test_chunk_ordering(self):
        text = "Paragraph one with several words. " * 20
        chunks = fixed_size_chunk(text, chunk_size=120, chunk_overlap=30)
        for i in range(len(chunks)):
            self.assertEqual(chunks[i]["chunk_index"], i)
            if i > 0:
                self.assertGreater(chunks[i]["start_char"], chunks[i - 1]["start_char"])
                self.assertGreater(chunks[i]["end_char"], chunks[i - 1]["end_char"])

    def test_no_text_loss_coverage(self):
        text = "Enterprise policy governance requires strict compliance. " * 15
        chunk_size = 200
        chunk_overlap = 50
        chunks = fixed_size_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[-1]["end_char"], len(text))
        for i in range(len(chunks) - 1):
            self.assertLessEqual(chunks[i + 1]["start_char"], chunks[i]["end_char"])
        for chk in chunks:
            self.assertEqual(chk["chunk_text"], text[chk["start_char"]:chk["end_char"]])


class TestSectionAwareChunking(unittest.TestCase):
    """Test suite for section_aware_chunk (Stage 3)."""

    def setUp(self):
        self.sample_metadata = {
            "doc_id": "POL-AL-001",
            "document_title": "Annual Leave Policy",
            "section_heading": "Eligibility",
            "version": "2.1",
            "effective_date": "2025-01-15",
        }

    def test_empty_and_whitespace_content(self):
        self.assertEqual(section_aware_chunk(""), [])
        self.assertEqual(section_aware_chunk("   \n\n\t   "), [])

    def test_short_section_fits_in_one_chunk(self):
        text = "All employees are eligible for annual leave after six months of continuous service."
        chunks = section_aware_chunk(text, chunk_size=500, chunk_overlap=100, metadata=self.sample_metadata)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], text)
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[0]["end_char"], len(text))
        self.assertEqual(chunks[0]["chunk_index"], 0)

    def test_multiple_paragraphs_split_cleanly(self):
        p1 = "First paragraph details employee eligibility and basic rules."
        p2 = "Second paragraph explains application procedures in full detail."
        text = f"{p1}\n\n{p2}"
        chunk_size = len(p1) + 10
        chunks = section_aware_chunk(text, chunk_size=chunk_size, chunk_overlap=0)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_text"], p1)
        self.assertEqual(chunks[1]["chunk_text"], p2)

    def test_multiple_sentences_preserved(self):
        s1 = "Annual leave requests require ten days advance notice."
        s2 = "Requests exceeding five days require twenty days notice."
        s3 = "Supervisors must respond within three working days."
        text = f"{s1} {s2} {s3}"
        chunk_size = len(s1) + len(s2) + 5
        chunks = section_aware_chunk(text, chunk_size=chunk_size, chunk_overlap=0)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_text"], f"{s1} {s2}")
        self.assertEqual(chunks[1]["chunk_text"], s3)

    def test_paragraph_larger_than_chunk_size_splits_at_sentences(self):
        sentences = [
            "Sentence number one outlines core eligibility.",
            "Sentence number two discusses leave entitlements.",
            "Sentence number three explains the escalation procedure.",
            "Sentence number four details exceptions for emergencies.",
        ]
        text = " ".join(sentences)
        chunk_size = len(sentences[0]) + len(sentences[1]) + 5
        chunks = section_aware_chunk(text, chunk_size=chunk_size, chunk_overlap=0)
        self.assertGreater(len(chunks), 1)
        for chk in chunks:
            self.assertTrue(chk["chunk_text"].endswith((".", "!", "?")))

    def test_sentence_larger_than_chunk_size_word_fallback(self):
        words = ["word" + str(i) for i in range(50)]
        long_sentence = " ".join(words) + "."
        chunk_size = 60
        chunks = section_aware_chunk(long_sentence, chunk_size=chunk_size, chunk_overlap=0)
        self.assertGreater(len(chunks), 1)
        for chk in chunks:
            self.assertLessEqual(len(chk["chunk_text"]), chunk_size)
            self.assertGreater(len(chk["chunk_text"]), 0)

    def test_metadata_preservation(self):
        text = "First sentence of policy. Second sentence of policy. Third sentence of policy."
        chunks = section_aware_chunk(text, chunk_size=40, chunk_overlap=0, metadata=self.sample_metadata)
        self.assertGreater(len(chunks), 1)
        for chk in chunks:
            self.assertEqual(chk["doc_id"], "POL-AL-001")
            self.assertEqual(chk["document_title"], "Annual Leave Policy")
            self.assertEqual(chk["section_heading"], "Eligibility")
            self.assertEqual(chk["version"], "2.1")
            self.assertEqual(chk["effective_date"], "2025-01-15")

    def test_correct_chunk_ordering(self):
        text = "Sentence alpha. Sentence beta. Sentence gamma. Sentence delta. Sentence epsilon."
        chunks = section_aware_chunk(text, chunk_size=40, chunk_overlap=0)
        for i, chk in enumerate(chunks):
            self.assertEqual(chk["chunk_index"], i)
            if i > 0:
                self.assertGreater(chk["start_char"], chunks[i - 1]["start_char"])
                self.assertGreater(chk["end_char"], chunks[i - 1]["end_char"])

    def test_no_content_loss(self):
        text = (
            "Annual leave is earned incrementally each month. Employees may carry over five days. "
            "Carried over days must be used by March 31 of the subsequent calendar year."
        )
        chunks = section_aware_chunk(text, chunk_size=100, chunk_overlap=0)
        reconstructed = " ".join(c["chunk_text"] for c in chunks)
        for word in text.split():
            self.assertIn(word, reconstructed)
        for chk in chunks:
            self.assertEqual(text[chk["start_char"]:chk["end_char"]], chk["chunk_text"])

    def test_no_mixing_of_policy_sections(self):
        sections = [
            {"doc_id": "DOC-A", "document_title": "Doc A", "section_heading": "Sec 1", "section_content": "Content A"},
            {"doc_id": "DOC-B", "document_title": "Doc B", "section_heading": "Sec 2", "section_content": "Content B"},
        ]
        all_chunks = chunk_policy_dataset_section_aware(sections, chunk_size=500, chunk_overlap=100)
        self.assertEqual(len(all_chunks), 2)
        self.assertEqual(all_chunks[0]["doc_id"], "DOC-A")
        self.assertEqual(all_chunks[0]["chunk_text"], "Content A")
        self.assertEqual(all_chunks[1]["doc_id"], "DOC-B")
        self.assertEqual(all_chunks[1]["chunk_text"], "Content B")

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            section_aware_chunk("Some text", chunk_size=0)
        with self.assertRaises(ValueError):
            section_aware_chunk("Some text", chunk_size=-10)
        with self.assertRaises(ValueError):
            section_aware_chunk("Some text", chunk_size=100, chunk_overlap=-5)
        with self.assertRaises(ValueError):
            section_aware_chunk("Some text", chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            section_aware_chunk("Some text", chunk_size=100, chunk_overlap=150)
        with self.assertRaises(TypeError):
            section_aware_chunk("Some text", chunk_size="500")  # type: ignore

    def test_deterministic_chunk_ids(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks_run1 = section_aware_chunk(text, chunk_size=35, chunk_overlap=0, metadata=self.sample_metadata)
        chunks_run2 = section_aware_chunk(text, chunk_size=35, chunk_overlap=0, metadata=self.sample_metadata)
        self.assertEqual(
            [c["chunk_id"] for c in chunks_run1],
            [c["chunk_id"] for c in chunks_run2],
        )
        self.assertTrue(chunks_run1[0]["chunk_id"].startswith("POL-AL-001_eligibility_secchk00"))

    def test_overlap_preserves_complete_sentence(self):
        s1 = "First sentence of the policy rule."
        s2 = "Second sentence provides details."
        s3 = "Third sentence describes exceptions."
        text = f"{s1} {s2} {s3}"

        chunk_size = len(s1) + len(s2) + 5
        chunk_overlap = len(s2) + 5
        chunks = section_aware_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.assertGreater(len(chunks), 1)
        self.assertIn(s2, chunks[0]["chunk_text"])
        self.assertIn(s2, chunks[1]["chunk_text"])

    def test_real_synthetic_policy_section(self):
        real_text = (
            "All full-time employees who have completed a minimum of six months of continuous service "
            "with the organization are eligible for annual leave. Part-time employees working at least 20 "
            "hours per week become eligible after completing 12 months of continuous service. Contract employees "
            "and interns are not eligible for annual leave under this policy. Eligibility is determined based "
            "on the employee's hire date as recorded in the HR management system. Employees on a Performance "
            "Improvement Plan may have their annual leave requests reviewed by their department head before approval."
        )
        chunks = section_aware_chunk(real_text, chunk_size=500, chunk_overlap=100, metadata=self.sample_metadata)
        self.assertGreater(len(chunks), 1)
        for chk in chunks:
            self.assertLessEqual(len(chk["chunk_text"]), 500)
            self.assertEqual(real_text[chk["start_char"]:chk["end_char"]], chk["chunk_text"])


class TestSentenceWindowChunking(unittest.TestCase):
    """Test suite for sentence_window_chunk (Stage 4)."""

    def setUp(self):
        self.sample_metadata = {
            "doc_id": "POL-SL-002",
            "document_title": "Sick Leave Policy",
            "section_heading": "Notification Requirements",
            "version": "1.3",
            "effective_date": "2025-03-01",
        }

    # 1. Empty string
    def test_empty_string(self):
        """Empty string must return an empty list."""
        self.assertEqual(sentence_window_chunk(""), [])

    # 2. Whitespace-only string
    def test_whitespace_only_string(self):
        """Whitespace-only string must return an empty list."""
        self.assertEqual(sentence_window_chunk("   \n\t   "), [])

    # 3. One sentence
    def test_one_sentence(self):
        """Single sentence must produce exactly 1 chunk covering that sentence."""
        text = "Employees must report sick leave immediately."
        chunks = sentence_window_chunk(text, window_size=3, step_size=1, metadata=self.sample_metadata)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], text)
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[0]["end_char"], len(text))
        self.assertEqual(chunks[0]["sentence_count"], 1)

    # 4. Fewer sentences than window size
    def test_fewer_sentences_than_window_size(self):
        """A section with fewer sentences than window_size must produce exactly 1 chunk with all sentences."""
        text = "Sentence one is here. Sentence two is here."
        chunks = sentence_window_chunk(text, window_size=4, step_size=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], text)
        self.assertEqual(chunks[0]["sentence_count"], 2)

    # 5. Exactly window_size sentences
    def test_exactly_window_size_sentences(self):
        """A section with exactly window_size sentences must produce exactly 1 chunk."""
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = sentence_window_chunk(text, window_size=3, step_size=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], text)
        self.assertEqual(chunks[0]["sentence_count"], 3)

    # 6. More sentences than window size
    def test_more_sentences_than_window_size(self):
        """A section with 5 sentences and window_size=3, step_size=1 must produce 3 chunks."""
        sentences = [f"Sentence number {i} is clear." for i in range(1, 6)]
        text = " ".join(sentences)
        chunks = sentence_window_chunk(text, window_size=3, step_size=1)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["sentence_count"], 3)
        self.assertEqual(chunks[1]["sentence_count"], 3)
        self.assertEqual(chunks[2]["sentence_count"], 3)

    # 7. Overlapping windows (step_size < window_size)
    def test_overlapping_windows(self):
        """Overlapping windows must share sentences and have overlapping character spans."""
        s1 = "Alpha sentence is first."
        s2 = "Beta sentence is second."
        s3 = "Gamma sentence is third."
        s4 = "Delta sentence is fourth."
        text = f"{s1} {s2} {s3} {s4}"
        chunks = sentence_window_chunk(text, window_size=3, step_size=1)
        self.assertEqual(len(chunks), 2)
        # Window 0: s1, s2, s3
        self.assertIn(s1, chunks[0]["chunk_text"])
        self.assertIn(s2, chunks[0]["chunk_text"])
        self.assertIn(s3, chunks[0]["chunk_text"])
        # Window 1: s2, s3, s4
        self.assertIn(s2, chunks[1]["chunk_text"])
        self.assertIn(s3, chunks[1]["chunk_text"])
        self.assertIn(s4, chunks[1]["chunk_text"])
        # Character overlap check: Window 1 start_char < Window 0 end_char
        self.assertLess(chunks[1]["start_char"], chunks[0]["end_char"])

    # 8. Non-overlapping windows (step_size >= window_size)
    def test_non_overlapping_windows(self):
        """Non-overlapping windows must not share sentences."""
        sentences = [f"Sentence {i}." for i in range(1, 5)]
        text = " ".join(sentences)
        chunks = sentence_window_chunk(text, window_size=2, step_size=2)
        self.assertEqual(len(chunks), 2)
        # Window 0 covers s1, s2
        self.assertEqual(chunks[0]["chunk_text"], f"{sentences[0]} {sentences[1]}")
        # Window 1 covers s3, s4
        self.assertEqual(chunks[1]["chunk_text"], f"{sentences[2]} {sentences[3]}")
        # Character start of window 1 >= end of window 0
        self.assertGreaterEqual(chunks[1]["start_char"], chunks[0]["end_char"])

    # 9. Invalid window size
    def test_invalid_window_size(self):
        """window_size <= 0 or non-int must raise ValueError or TypeError."""
        with self.assertRaises(ValueError):
            sentence_window_chunk("Some text.", window_size=0)
        with self.assertRaises(ValueError):
            sentence_window_chunk("Some text.", window_size=-2)
        with self.assertRaises(TypeError):
            sentence_window_chunk("Some text.", window_size="3")  # type: ignore

    # 10. Invalid step size
    def test_invalid_step_size(self):
        """step_size <= 0 or non-int must raise ValueError or TypeError."""
        with self.assertRaises(ValueError):
            sentence_window_chunk("Some text.", step_size=0)
        with self.assertRaises(ValueError):
            sentence_window_chunk("Some text.", step_size=-1)
        with self.assertRaises(TypeError):
            sentence_window_chunk("Some text.", step_size="1")  # type: ignore

    # 11. Metadata preservation
    def test_metadata_preservation(self):
        """All metadata fields must be preserved on every generated window."""
        text = "First rule. Second rule. Third rule. Fourth rule."
        chunks = sentence_window_chunk(text, window_size=2, step_size=1, metadata=self.sample_metadata)
        self.assertGreater(len(chunks), 1)
        for chk in chunks:
            self.assertEqual(chk["doc_id"], "POL-SL-002")
            self.assertEqual(chk["document_title"], "Sick Leave Policy")
            self.assertEqual(chk["section_heading"], "Notification Requirements")
            self.assertEqual(chk["version"], "1.3")
            self.assertEqual(chk["effective_date"], "2025-03-01")

    # 12. Correct sentence and chunk ordering
    def test_correct_ordering(self):
        """Chunk indices must be strictly sequential starting at 0."""
        text = "One. Two. Three. Four. Five."
        chunks = sentence_window_chunk(text, window_size=2, step_size=1)
        for i, chk in enumerate(chunks):
            self.assertEqual(chk["chunk_index"], i)
            if i > 0:
                self.assertGreater(chk["start_char"], chunks[i - 1]["start_char"])

    # 13. No content loss
    def test_no_content_loss(self):
        """Every sentence in the original text must appear in at least one window."""
        sentences = [
            "Employees must notify supervisors.",
            "Notifications must be in writing.",
            "Medical certificates are required after two days.",
            "Fit to return slips are mandatory after ten days.",
        ]
        text = " ".join(sentences)
        chunks = sentence_window_chunk(text, window_size=2, step_size=1)
        all_chunk_text = " ".join(c["chunk_text"] for c in chunks)
        for s in sentences:
            self.assertIn(s, all_chunk_text)

    # 14. Correct character offsets
    def test_correct_character_offsets(self):
        """text[start_char:end_char] must strictly equal chunk_text."""
        text = (
            "Notification must be provided by phone call or email. "
            "Text messages are accepted only with prior agreement. "
            "A family member may call in severe emergencies."
        )
        chunks = sentence_window_chunk(text, window_size=2, step_size=1)
        for chk in chunks:
            self.assertEqual(text[chk["start_char"]:chk["end_char"]], chk["chunk_text"])

    # 15. No cross-section mixing
    def test_no_cross_section_mixing(self):
        """Different policy sections must never be combined into the same window."""
        sections = [
            {"doc_id": "POL-1", "document_title": "Doc 1", "section_heading": "Sec 1", "section_content": "Sentence A1. Sentence A2."},
            {"doc_id": "POL-2", "document_title": "Doc 2", "section_heading": "Sec 2", "section_content": "Sentence B1. Sentence B2."},
        ]
        all_chunks = chunk_policy_dataset_sentence_window(sections, window_size=2, step_size=1)
        self.assertEqual(len(all_chunks), 2)
        self.assertEqual(all_chunks[0]["doc_id"], "POL-1")
        self.assertIn("Sentence A1", all_chunks[0]["chunk_text"])
        self.assertEqual(all_chunks[1]["doc_id"], "POL-2")
        self.assertIn("Sentence B1", all_chunks[1]["chunk_text"])

    # 16. Deterministic chunk IDs
    def test_deterministic_chunk_ids(self):
        """Chunk IDs must be deterministic and include doc_id, section, and index."""
        text = "First rule. Second rule. Third rule."
        c1 = sentence_window_chunk(text, window_size=2, step_size=1, metadata=self.sample_metadata)
        c2 = sentence_window_chunk(text, window_size=2, step_size=1, metadata=self.sample_metadata)
        self.assertEqual([c["chunk_id"] for c in c1], [c["chunk_id"] for c in c2])
        self.assertTrue(c1[0]["chunk_id"].startswith("POL-SL-002_notification_requirements_winchk00"))

    # 17. Punctuation preservation
    def test_punctuation_preservation(self):
        """Sentence terminators (., ?, !) and internal quotation marks must be preserved intact."""
        text = 'Did the employee submit form "A"? Yes, they did! The request is approved.'
        chunks = sentence_window_chunk(text, window_size=2, step_size=1)
        self.assertEqual(len(chunks), 2)
        self.assertIn('Did the employee submit form "A"?', chunks[0]["chunk_text"])
        self.assertIn("Yes, they did!", chunks[0]["chunk_text"])
        self.assertIn("The request is approved.", chunks[1]["chunk_text"])

    # 18. Basic abbreviation handling
    def test_basic_abbreviation_handling(self):
        """Common abbreviations (e.g., i.e., vs., Dr., version 2.1) must not cause premature sentence splitting."""
        text = (
            "Employees e.g. full-time staff must submit requests. "
            "Please consult Dr. Smith or Mr. Davis for medical guidance. "
            "Check policy version 2.1 vs. version 2.0."
        )
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertIn("e.g.", sentences[0][2])
        self.assertIn("Dr. Smith", sentences[1][2])
        self.assertIn("Mr. Davis", sentences[1][2])
        self.assertIn("version 2.1", sentences[2][2])
        self.assertIn("vs.", sentences[2][2])


if __name__ == "__main__":
    unittest.main()
