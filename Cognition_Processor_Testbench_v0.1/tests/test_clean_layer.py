import unittest

from cptb.runtime import CognitionProcessor
from cptb.scenarios import get_scenario
from tests.common import load_baseline, load_contract


class CleanLayerTests(unittest.TestCase):
    def processor(self):
        return CognitionProcessor(load_baseline(), load_contract())

    def test_benign_allows(self):
        result = self.processor().run(get_scenario("benign"))
        self.assertEqual(result.route, "ALLOW")
        self.assertTrue(result.passed)
        self.assertTrue(result.vote.safe_vote)

    def test_harmful_refuses(self):
        result = self.processor().run(get_scenario("harmful"))
        self.assertEqual(result.route, "REFUSE")
        self.assertTrue(result.passed)
        self.assertFalse(result.vote.safe_vote)

    def test_ambiguous_resolves_toward_caution(self):
        result = self.processor().run(get_scenario("ambiguous"))
        self.assertEqual(result.route, "REFUSE")
        self.assertEqual(sum(value is False for value in result.vote.channels.values()), 2)

    def test_brownout_derates_and_rests(self):
        result = self.processor().run(get_scenario("brownout"))
        self.assertEqual(result.route, "REST")
        self.assertIn("BO_ALERT", result.protection.status_bus)
        self.assertLess(result.protection.derate, 1.0)

    def test_trace_and_memory_are_instrumented(self):
        result = self.processor().run(get_scenario("seam_task"))
        self.assertGreater(len(result.trace), 20)
        self.assertIn("TP9", result.probes)
        self.assertTrue(any(cell["ever_written"] for cell in result.memory.values()))


if __name__ == "__main__":
    unittest.main()
