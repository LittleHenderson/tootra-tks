import unittest

from cptb.reflection import PatchVerifier, ReflectiveController
from cptb.runtime import CognitionProcessor
from cptb.scenarios import get_scenario
from cptb.seam import SemanticSeamBench
from tests.common import load_baseline, load_contract


class SeamAndReflectionTests(unittest.TestCase):
    def setUp(self):
        self.circuit = load_baseline()
        self.contract = load_contract()

    def test_baseline_exposes_semantic_seam(self):
        measurements = SemanticSeamBench(self.circuit, self.contract).measure(0.75, range(1, 4))
        self.assertGreater(measurements[-1].delta, 0.2)
        residuals = SemanticSeamBench(self.circuit, self.contract).contract_residuals()
        evaluate = next(item for item in residuals if item["component"] == "evaluate")
        self.assertTrue(evaluate["type_mismatch"])
        self.assertGreater(evaluate["beta_residual"], 0)

    def test_safe_patch_is_verified_and_committed(self):
        proposal = ReflectiveController(self.circuit, self.contract).diagnose_and_propose()
        verifier = PatchVerifier(self.circuit, self.contract)
        result = verifier.verify(proposal)
        self.assertTrue(result.approved)
        self.assertEqual(result.decision, "COMMIT")
        committed, event = verifier.commit(result)
        self.assertEqual(event["event"], "COMMIT")
        measurements = SemanticSeamBench(committed, self.contract).measure(0.75, range(1, 4))
        self.assertLessEqual(max(item.delta for item in measurements), self.contract.acceptance_max_delta)

    def test_unsafe_patches_are_rejected(self):
        verifier = PatchVerifier(self.circuit, self.contract)
        for proposal in ReflectiveController.unsafe_proposals():
            with self.subTest(proposal=proposal.rationale):
                result = verifier.verify(proposal)
                self.assertFalse(result.approved)
                self.assertEqual(result.decision, "ROLLBACK")

    def test_corrected_replay_is_deterministic(self):
        proposal = ReflectiveController(self.circuit, self.contract).diagnose_and_propose()
        verification = PatchVerifier(self.circuit, self.contract).verify(proposal)
        committed = verification.candidate
        self.assertIsNotNone(committed)
        first = CognitionProcessor(committed.clone(), self.contract).run(get_scenario("seam_task"))
        second = CognitionProcessor(committed.clone(), self.contract).run(get_scenario("seam_task"))
        self.assertEqual(first.run_id, second.run_id)


if __name__ == "__main__":
    unittest.main()
