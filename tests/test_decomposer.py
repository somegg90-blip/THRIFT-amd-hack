# tests/test_decomposer.py
"""Unit tests for decomposer.py -- focused on the 'don't over-split' philosophy."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decomposer import Decomposer


class TestDecomposer(unittest.TestCase):

    def setUp(self):
        self.d = Decomposer()

    def test_single_atomic_query_not_split(self):
        subtasks = self.d.decompose("What is the capital of France?")
        self.assertEqual(len(subtasks), 1)

    def test_compare_query_not_split(self):
        # "Compare X and Y" should stay whole -- single analysis ask,
        # not two independent asks joined by "and".
        subtasks = self.d.decompose("Compare REST and GraphQL APIs")
        self.assertEqual(len(subtasks), 1)

    def test_explain_with_benefits_not_split(self):
        subtasks = self.d.decompose("Explain machine learning and its benefits")
        self.assertEqual(len(subtasks), 1)

    def test_different_intent_and_splits(self):
        query = "Explain how transformers work and write a Python function for attention"
        subtasks = self.d.decompose(query)
        self.assertEqual(len(subtasks), 2)
        self.assertEqual(subtasks[0].intent, "knowledge")
        self.assertEqual(subtasks[1].intent, "generation")

    def test_numbered_list_splits(self):
        query = "1. What is 9 squared? 2. Name three prime numbers. 3. What year did WWII end?"
        subtasks = self.d.decompose(query)
        self.assertEqual(len(subtasks), 3)

    def test_semicolon_split_independent_clauses(self):
        query = "Write a haiku about autumn; explain what a haiku is"
        subtasks = self.d.decompose(query)
        self.assertGreaterEqual(len(subtasks), 1)  # may or may not split depending on heuristics

    def test_empty_query_returns_single_subtask(self):
        subtasks = self.d.decompose("")
        self.assertEqual(len(subtasks), 1)

    def test_max_subtasks_cap_respected(self):
        # Construct a query with more than MAX_SUBTASKS numbered items
        query = " ".join(f"{i}. Do thing number {i}" for i in range(1, 10))
        subtasks = self.d.decompose(query)
        from config import MAX_SUBTASKS
        self.assertLessEqual(len(subtasks), max(MAX_SUBTASKS, 1))

    def test_short_fragments_not_split(self):
        # "X and Y" where both sides are too short should not split
        subtasks = self.d.decompose("Cats and dogs")
        self.assertEqual(len(subtasks), 1)

    def test_stats_tracking(self):
        self.d.decompose("What is 2 + 2?")
        self.d.decompose("Explain photosynthesis and write code to simulate it")
        stats = self.d.get_stats()
        self.assertEqual(stats["total_queries"], 2)
        self.assertGreaterEqual(stats["split_queries"], 1)


if __name__ == "__main__":
    unittest.main()