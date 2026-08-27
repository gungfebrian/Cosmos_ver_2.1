# TinyML Behavior Inference Design

**Date:** 2026-08-25  
**Status:** Approved for implementation  
**Branch:** `feat/tinyml-behavior-inference`

## Goal

Add a small, reproducible machine-learning layer to CozmoMini that predicts the robot's next autonomous behavior from its existing internal drives, while preserving the current hard safety and personality thresholds.

## Why this fits CozmoMini

CozmoMini already exposes a compact state vector: energy, mood, annoyance, boredom, touch activity, interaction recency, and tap-burst intensity. That makes it a good TinyML teaching example: the model has meaningful inputs, the target behaviors are explainable, and inference can run locally without Wi-Fi, a cloud API, or a large runtime.

The contribution will not claim that the model learned from real user telemetry. Training data is generated deterministically from the existing expert policy, so the README will describe it accurately as policy distillation and make the limitation explicit.

## Design

### Model boundary

The model is an advisory classifier used only when the firmware is in an interruptible autonomous mode (`IDLE`, `ROAM`, or `ATTENTION`). Existing threshold checks remain authoritative for rage, annoyance, and low-energy sleep transitions. Touch gestures and timed transient modes remain hand-coded because they are event-driven rather than state-classification problems.

The classifier predicts one of six intents:

| Intent | Firmware action |
| --- | --- |
| `IDLE` | Keep the current autonomous behavior |
| `ANGRY` | Enter the existing tantrum mode |
| `ANNOYED` | Enter the existing annoyed mode |
| `SLEEPY` | Enter the existing sleepy mode |
| `ATTENTION` | Enter the existing attention-seeking mode |
| `PLAYFUL` | Enter the existing playful mode |

The firmware will accept a prediction only when confidence is at least `0.78`, and it will evaluate the model no more often than once per second. This prevents low-confidence jitter and keeps the model from repeatedly re-entering the same mode.

### Features

All seven features are normalized to `[0, 1]` before training and inference:

1. `energy`: current energy divided by 100.
2. `mood`: current mood divided by 100.
3. `annoyance`: current annoyance divided by 100.
4. `boredom`: current boredom divided by 100.
5. `touch_active`: whether the capacitive pad is currently active.
6. `interaction_age`: time since the last interaction, capped at five seconds.
7. `tap_burst`: recent taps in the five-second window, capped at eight taps.

### Training and export

The Python training module will:

1. Generate a deterministic grid of representative drive and interaction states.
2. Label each state with the current priority policy.
3. Fit multinomial softmax regression with L2 regularization using only the Python standard library.
4. Export the learned coefficients to a checked-in C++ header.
5. Record model metadata (feature order, class order, training parameters, and dataset size) in JSON.

The C++ header will implement the same linear logits and numerically stable softmax calculation. A host-side C++ smoke test will compile and exercise the header so the generated artifact cannot silently drift away from the embedded interface.

### Data flow

```text
existing drives + gesture context
              |
              v
       normalized features
              |
              v
      softmax linear model
              |
       confidence gate >= .78
              |
              v
 existing firmware mode transition
```

### Failure handling

- Invalid feature values are clamped to the supported normalized range.
- A non-finite or malformed model is rejected by the Python loader/export tests.
- The firmware uses the existing rule-based behavior if the confidence gate is not met.
- Model inference must not allocate memory, access hardware, or block the main loop.

## Testing strategy

- Python unit tests cover feature normalization, policy-label precedence, stable probabilities, confidence gating, deterministic training, and export metadata.
- A C++ host smoke test verifies that the generated header compiles without Arduino dependencies and returns a valid class/confidence pair.
- The Arduino sketch is checked for the expected include, feature construction, confidence gate, and safety-priority ordering. Hardware flashing remains a manual step because the repository does not contain a board test fixture.
- GitHub Actions will run the Python test suite and C++ smoke test on every push and pull request.

## Documentation

The README will add an ML/TinyML section covering the model's purpose, exact feature list, training command, export command, how the firmware uses confidence, and the policy-distillation limitation. It will also add the new repository layout and a roadmap item for replacing synthetic labels with opt-in real telemetry.

## Out of scope

- Cloud inference, remote telemetry, or collection of personal data.
- A large framework such as TensorFlow Lite Micro.
- Replacing the existing hand-authored gesture state machine.
- Claiming benchmark accuracy against real-world user behavior.
- Changing motor safety limits or hardware pin assignments.
