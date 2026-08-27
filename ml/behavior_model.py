"""Train and run the small behavior model used by CozmoMini.

The model intentionally uses only the Python standard library so that the
training and export path remains reproducible in a clean checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


FEATURE_NAMES = (
    "energy",
    "mood",
    "annoyance",
    "boredom",
    "touch_active",
    "interaction_age",
    "tap_burst",
)
CLASS_NAMES = ("IDLE", "ANGRY", "ANNOYED", "SLEEPY", "ATTENTION", "PLAYFUL")


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Return a finite value inside the model's supported range."""

    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class State:
    """The seven normalized inputs shared by Python and embedded inference."""

    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features")
        object.__setattr__(self, "features", tuple(clamp(value) for value in self.features))

    @classmethod
    def from_raw(
        cls,
        energy: float,
        mood: float,
        annoyance: float,
        boredom: float,
        touch_active: bool,
        interaction_age_ms: float,
        tap_burst: float,
    ) -> "State":
        return cls(
            (
                clamp(energy / 100.0),
                clamp(mood / 100.0),
                clamp(annoyance / 100.0),
                clamp(boredom / 100.0),
                1.0 if touch_active else 0.0,
                clamp(interaction_age_ms / 5000.0),
                clamp(tap_burst / 8.0),
            )
        )


@dataclass(frozen=True)
class Model:
    """Weights and biases for a multiclass linear softmax model."""

    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    epochs: int = 0
    learning_rate: float = 0.0
    l2: float = 0.0


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    probabilities: tuple[float, ...]


def expert_intent(state: State) -> str:
    """Label a state using CozmoMini's existing safety-first policy."""

    energy, mood, annoyance, boredom, touch_active, _, tap_burst = state.features
    if annoyance >= 0.75 or tap_burst >= 1.0:
        return "ANGRY"
    if annoyance >= 0.40 or tap_burst >= 0.375:
        return "ANNOYED"
    if energy <= 0.18:
        return "SLEEPY"
    if boredom >= 0.70 and not touch_active:
        return "ATTENTION"
    if mood >= 0.82 and energy >= 0.55:
        return "PLAYFUL"
    return "IDLE"


def generate_dataset() -> tuple[tuple[State, ...], tuple[str, ...]]:
    """Generate deterministic representative states from the expert policy."""

    energy_values = (0.05, 0.18, 0.30, 0.55, 0.80, 1.0)
    mood_values = (0.0, 0.50, 0.82, 0.95)
    annoyance_values = (0.0, 0.40, 0.75, 1.0)
    boredom_values = (0.0, 0.70, 1.0)
    touch_values = (0.0, 1.0)
    age_values = (0.0, 0.5, 1.0)
    tap_values = (0.0, 0.375, 1.0)

    buckets: dict[str, list[State]] = {class_name: [] for class_name in CLASS_NAMES}
    for energy in energy_values:
        for mood in mood_values:
            for annoyance in annoyance_values:
                for boredom in boredom_values:
                    for touch_active in touch_values:
                        for interaction_age in age_values:
                            for tap_burst in tap_values:
                                state = State(
                                    (
                                        energy,
                                        mood,
                                        annoyance,
                                        boredom,
                                        touch_active,
                                        interaction_age,
                                        tap_burst,
                                    )
                                )
                                buckets[expert_intent(state)].append(state)

    # The raw Cartesian grid over-represents high-annoyance states. Keep the
    # same deterministic number per class so the classifier learns every
    # intent instead of learning the class prior.
    examples_per_class = min(len(examples) for examples in buckets.values()) if buckets else 0
    states = [
        state
        for class_name in CLASS_NAMES
        for state in buckets[class_name][:examples_per_class]
    ]
    labels = [
        class_name
        for class_name in CLASS_NAMES
        for _ in buckets[class_name][:examples_per_class]
    ]
    return tuple(states), tuple(labels)


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Compute numerically stable softmax probabilities."""

    if not logits:
        raise ValueError("softmax requires at least one logit")
    if not all(math.isfinite(value) for value in logits):
        raise ValueError("softmax requires finite logits")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def train(
    states: Sequence[State],
    labels: Sequence[str],
    *,
    epochs: int = 1800,
    learning_rate: float = 0.25,
    l2: float = 0.001,
) -> Model:
    """Fit softmax regression with deterministic full-batch gradient descent."""

    if not states or len(states) != len(labels):
        raise ValueError("states and labels must be non-empty and have equal length")
    if epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("training parameters are out of range")
    if any(label not in CLASS_NAMES for label in labels):
        raise ValueError("unknown behavior label")

    class_count = len(CLASS_NAMES)
    feature_count = len(FEATURE_NAMES)
    weights = [[0.0] * feature_count for _ in range(class_count)]
    bias = [0.0] * class_count
    label_indices = [CLASS_NAMES.index(label) for label in labels]

    for _ in range(epochs):
        weight_gradients = [[0.0] * feature_count for _ in range(class_count)]
        bias_gradients = [0.0] * class_count

        for state, target_index in zip(states, label_indices):
            logits = [
                bias[class_index]
                + sum(weight * feature for weight, feature in zip(weights[class_index], state.features))
                for class_index in range(class_count)
            ]
            probabilities = softmax(logits)
            for class_index, probability in enumerate(probabilities):
                error = probability - (1.0 if class_index == target_index else 0.0)
                bias_gradients[class_index] += error
                for feature_index, feature in enumerate(state.features):
                    weight_gradients[class_index][feature_index] += error * feature

        scale = 1.0 / len(states)
        for class_index in range(class_count):
            bias[class_index] -= learning_rate * bias_gradients[class_index] * scale
            for feature_index in range(feature_count):
                gradient = weight_gradients[class_index][feature_index] * scale
                gradient += l2 * weights[class_index][feature_index]
                weights[class_index][feature_index] -= learning_rate * gradient

    return Model(
        weights=tuple(tuple(row) for row in weights),
        bias=tuple(bias),
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )


def predict(model: Model, state: State) -> Prediction:
    """Return the most likely intent and its confidence for a state."""

    if len(model.weights) != len(CLASS_NAMES) or len(model.bias) != len(CLASS_NAMES):
        raise ValueError("model class dimensions do not match")
    if any(len(row) != len(FEATURE_NAMES) for row in model.weights):
        raise ValueError("model feature dimensions do not match")
    logits = tuple(
        model.bias[class_index]
        + sum(weight * feature for weight, feature in zip(model.weights[class_index], state.features))
        for class_index in range(len(CLASS_NAMES))
    )
    probabilities = softmax(logits)
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return Prediction(CLASS_NAMES[best_index], probabilities[best_index], probabilities)


def confidence_gate(confidence: float, *, threshold: float = 0.78) -> bool:
    """Return whether a prediction is confident enough to advise firmware."""

    return math.isfinite(confidence) and confidence >= threshold


def save_model(model: Model, path: str | Path) -> None:
    """Write a stable JSON representation of a model."""

    payload = {
        "weights": model.weights,
        "bias": model.bias,
        "epochs": model.epochs,
        "learning_rate": model.learning_rate,
        "l2": model.l2,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_model(path: str | Path) -> Model:
    """Load and validate a model produced by save_model."""

    payload = json.loads(Path(path).read_text())
    weights = tuple(tuple(float(value) for value in row) for row in payload["weights"])
    bias = tuple(float(value) for value in payload["bias"])
    model = Model(
        weights=weights,
        bias=bias,
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        l2=float(payload["l2"]),
    )
    if len(weights) != len(CLASS_NAMES) or len(bias) != len(CLASS_NAMES):
        raise ValueError("model class dimensions do not match")
    if any(len(row) != len(FEATURE_NAMES) for row in weights):
        raise ValueError("model feature dimensions do not match")
    if not all(math.isfinite(value) for row in weights for value in row):
        raise ValueError("model weights must be finite")
    if not all(math.isfinite(value) for value in bias):
        raise ValueError("model bias must be finite")
    return model
