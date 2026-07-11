# tests/test_integration.py
"""End-to-end tests for the full THRIFT pipeline (decomposer -> cascade -> reassembler)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thrift import Thrift


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.agent = Thrift()

    def test_simple_math_end_to_end(self):
        run = self.agent.answer("what is 12 * 12")
        self.assertEqual(run.final_answer.text, "144")
        self.assertEqual(run.tokens_used, 0)

    def test_empty_query_does_not_crash(self):
        run = self.agent.answer("")
        self.assertIsNotNone(run.final_answer.text)

    def test_compound_query_decomposes_and_runs(self):
        run = self.agent.answer(
            "Explain how recursion works and write a Python function for factorial"
        )
        self.assertEqual(len(run.subtasks), 2)
        self.assertEqual(len(run.cascade_results), 2)
        self.assertIsNotNone(run.final_answer.text)

    def test_never_raises_on_arbitrary_input(self):
        weird_inputs = [
            "??? !!! ###",
            "a" * 5000,
            "1. 2. 3.",
            None,
        ]
        for inp in weird_inputs:
            try:
                run = self.agent.answer(inp if inp is not None else "")
            except Exception as e:
                self.fail(f"agent.answer() raised on input {inp!r}: {e}")
            self.assertIsNotNone(run)

    def test_stats_available_after_runs(self):
        self.agent.answer("what is 5 + 5")
        stats = self.agent.get_stats()
        self.assertIn("total_queries", stats)
        self.assertIn("estimated_token_savings_pct", stats)

    def test_answer_text_convenience_method(self):
        text = self.agent.answer_text("what is 3 * 3")
        self.assertEqual(text, "9")


if __name__ == "__main__":
    unittest.main()