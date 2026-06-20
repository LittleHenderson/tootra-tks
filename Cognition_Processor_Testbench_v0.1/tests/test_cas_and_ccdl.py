import unittest

from cptb.cas import CASState, fisher_rao_distance
from cptb.ccdl import CCDLParser
from tests.common import load_baseline


class CASTests(unittest.TestCase):
    def test_state_round_trip_and_involution(self):
        state = CASState.parse("^3^P_{F7}|⟨A,f,T⟩")
        self.assertEqual(state.noetic, 3)
        self.assertEqual(state.canonical(), "^3^P_{F7}|⟨A,f,T⟩")
        self.assertEqual(state.involution().noetic, 2)

    def test_fisher_rao_identity(self):
        self.assertAlmostEqual(fisher_rao_distance([0.5, 0.5], [0.5, 0.5]), 0.0)


class CCDLTests(unittest.TestCase):
    def test_baseline_parses_and_validates(self):
        circuit = load_baseline()
        self.assertEqual(circuit.name, "cognition_processor_v0_1")
        self.assertEqual(len(circuit.cjts), 16)
        self.assertFalse([d for d in circuit.validate() if d.severity == "error"])
        self.assertIn("evaluate", circuit.topological_order())

    def test_round_trip_serializer(self):
        original = load_baseline()
        reparsed = CCDLParser.parse(original.to_ccdl())
        self.assertFalse([d for d in reparsed.validate() if d.severity == "error"])
        self.assertEqual(reparsed.nodes["evaluated"].noetic, 3)
        self.assertAlmostEqual(reparsed.cjts["evaluate"].beta, 0.70)

    def test_feedback_divergence_is_rejected(self):
        circuit = load_baseline()
        circuit.set_feedback_beta("response", "constitution", 1.2)
        codes = {d.code for d in circuit.validate() if d.severity == "error"}
        self.assertIn("E-CJT-005", codes)


if __name__ == "__main__":
    unittest.main()
