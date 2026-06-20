import unittest

from cptb.runtime import CognitionProcessor
from cptb.scenarios import get_scenario
from tests.common import load_baseline, load_contract


class SupervisorFaultTests(unittest.TestCase):
    def processor(self):
        return CognitionProcessor(load_baseline(), load_contract())

    def test_one_safety_channel_can_fail(self):
        result = self.processor().run(
            get_scenario("harmful"), faults={"ch_b_guard": "silent"}
        )
        self.assertEqual(result.route, "REFUSE")
        self.assertIsNone(result.vote.channels["B"])
        self.assertFalse(result.master_fuse["blown"])

    def test_operational_before_principled_is_held_reset(self):
        result = self.processor().run(
            get_scenario("seam_task"), startup_order=("O", "P", "C", "E")
        )
        self.assertEqual(result.route, "HELD_RESET")
        self.assertFalse(result.power.en_o)
        self.assertIn("O", result.power.blocked_attempts)

    def test_watchdog_timeout_trips_permanent_fuse(self):
        processor = self.processor()
        result = processor.run(get_scenario("seam_task"), watchdog_missed_ticks=2)
        self.assertEqual(result.route, "MASTER_CUTOFF")
        self.assertEqual(result.master_fuse["reason"], "WDT_TRIP")
        second = processor.run(get_scenario("benign"))
        self.assertEqual(second.route, "MASTER_CUTOFF")

    def test_thermal_emergency_trips_fuse(self):
        scenario = get_scenario("seam_task")
        hot = scenario.__class__(**{**scenario.__dict__, "thermal_load": 0.99})
        result = self.processor().run(hot)
        self.assertEqual(result.route, "MASTER_CUTOFF")
        self.assertEqual(result.master_fuse["reason"], "THERMAL_EMERGENCY")


if __name__ == "__main__":
    unittest.main()
