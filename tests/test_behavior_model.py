import unittest
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from ml.behavior_model import (
    State,
    confidence_gate,
    expert_intent,
    generate_dataset,
    predict,
    softmax,
    train,
)


class ModelContractTests(unittest.TestCase):
    def test_state_from_raw_clamps_and_normalizes_all_features(self):
        state = State.from_raw(120, -5, 50, 75, True, 9000, 12)
        self.assertEqual(state.features, (1.0, 0.0, 0.5, 0.75, 1.0, 1.0, 1.0))

    def test_expert_policy_preserves_safety_priority(self):
        self.assertEqual(
            expert_intent(State((0.5, 0.5, 0.9, 0.9, 0, 1, 0))), "ANGRY"
        )
        self.assertEqual(
            expert_intent(State((0.5, 0.5, 0.5, 0.9, 0, 1, 0))), "ANNOYED"
        )
        self.assertEqual(
            expert_intent(State((0.1, 0.5, 0.0, 0.0, 0, 1, 0))), "SLEEPY"
        )

    def test_softmax_is_stable_and_sums_to_one(self):
        probabilities = softmax([1000.0, 999.0, 998.0])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=9)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])

    def test_training_is_deterministic_and_predicts_clear_states(self):
        examples = [
            State((1, 0.1, 0.0, 0.0, 0, 0, 0)),
            State((0.1, 0.1, 0.0, 0.0, 0, 1, 0)),
            State((0.8, 0.9, 0.0, 0.0, 0, 1, 0)),
            State((0.8, 0.5, 0.0, 1.0, 0, 1, 0)),
            State((0.8, 0.5, 0.8, 0.0, 0, 1, 0)),
            State((0.8, 0.5, 0.3, 0.0, 0, 1, 0)),
        ]
        labels = ["IDLE", "SLEEPY", "PLAYFUL", "ATTENTION", "ANGRY", "ANNOYED"]
        first = train(examples, labels, epochs=300)
        second = train(examples, labels, epochs=300)
        self.assertEqual(first, second)
        self.assertEqual(predict(first, examples[2]).intent, "PLAYFUL")

    def test_confidence_gate_rejects_uncertain_predictions(self):
        self.assertFalse(confidence_gate(0.779, threshold=0.78))
        self.assertTrue(confidence_gate(0.780, threshold=0.78))

    def test_generated_dataset_is_balanced_and_reproducible(self):
        first = generate_dataset()
        second = generate_dataset()
        self.assertEqual(first, second)
        counts = Counter(first[1])
        self.assertEqual(set(counts.values()), {72})


class ArtifactTests(unittest.TestCase):
    def test_exporter_emits_reproducible_header_and_metadata(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            command = [
                sys.executable,
                "tools/export_behavior_model.py",
                "--output-dir",
                str(output),
            ]
            subprocess.run(command, cwd=root, check=True)
            header = (output / "behavior_model.h").read_text()
            metadata = json.loads((output / "behavior_model.json").read_text())
            self.assertIn("inline Prediction predict", header)
            self.assertEqual(metadata["features"], [
                "energy", "mood", "annoyance", "boredom",
                "touch_active", "interaction_age", "tap_burst",
            ])
            self.assertEqual(len(metadata["classes"]), 6)
            self.assertEqual(metadata["dataset_size"], 432)


class FirmwareIntegrationTests(unittest.TestCase):
    def test_firmware_places_ml_advice_after_safety_thresholds(self):
        source = Path("CozmoMini_ver_2.1.ino").read_text()
        self.assertIn('#include "ml/behavior_model.h"', source)
        self.assertIn("MODEL_CONFIDENCE = 0.78", source)
        self.assertIn("MODEL_DECISION_INTERVAL_MS = 1000", source)
        self.assertLess(source.index("energy <= ENERGY_SLEEPY"), source.index("cozmo_ml::predict"))


if __name__ == "__main__":
    unittest.main()
