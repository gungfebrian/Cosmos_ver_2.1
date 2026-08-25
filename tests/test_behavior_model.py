import unittest
from collections import Counter

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


if __name__ == "__main__":
    unittest.main()
