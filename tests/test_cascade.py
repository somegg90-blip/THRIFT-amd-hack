# tests/test_cascade.py
"""Unit tests for cascade.py -- escalation order and graceful degradation."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade import Cascade
from decomposer import SubTask


class TestCascade(unittest.TestCase):

    def setUp(self):
        self.cascade = Cascade(confidence_threshold=0.65)

    def test_tier0_handles_math_with_zero_tokens(self):
        result = self.cascade.run(SubTask("what is 6 * 7"))
        self.assertEqual(result.tier_used, 0)
        self.assertEqual(result.answer, "42")
        self.assertEqual(result.tokens_used, 0)
        self.assertFalse(result.degraded)

    def test_tier0_skipped_for_non_math(self):
        result = self.cascade.run(SubTask("explain gravity"))
        # Tier 0 can't handle it -- attempted but tier_used should not be 0
        self.assertIn(0, result.tiers_attempted)
        self.assertNotEqual(result.tier_used, 0)

    def test_never_raises_on_unreachable_remote(self):
        # In environments without network/API key, this should degrade
        # gracefully rather than throwing.
        try:
            result = self.cascade.run(SubTask("explain quantum entanglement"))
        except Exception as e:
            self.fail(f"cascade.run() raised an exception instead of degrading: {e}")
        self.assertIsNotNone(result)

    def test_empty_subtask_does_not_crash(self):
        try:
            result = self.cascade.run(SubTask(""))
        except Exception as e:
            self.fail(f"cascade.run() raised on empty subtask: {e}")
        self.assertIsNotNone(result)

    def test_stats_tracking(self):
        self.cascade.run(SubTask("what is 2 + 2"))
        self.cascade.run(SubTask("what is 3 + 3"))
        stats = self.cascade.get_stats()
        self.assertEqual(stats["total_subtasks"], 2)

    def test_run_all_processes_every_subtask(self):
        subtasks = [SubTask("what is 1 + 1"), SubTask("what is 2 + 2")]
        results = self.cascade.run_all(subtasks)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()