# TinyML Behavior Inference Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a deterministic, dependency-free softmax behavior classifier that runs as an advisory TinyML layer on the ESP32-C3 and is reproducibly trained, exported, tested, and documented.

**Architecture:** ml/behavior_model.py owns normalized feature construction, expert-label generation, softmax training, inference, and model serialization. tools/export_behavior_model.py trains the model and emits the checked-in C++ header plus metadata. The firmware includes that header and applies only high-confidence predictions after its existing safety thresholds; Python tests and a host C++ smoke test protect both sides of the interface.

**Tech Stack:** Python 3 standard library, C++17 host smoke test, Arduino C++/ESP32-C3, GitHub Actions.

**Spec:** docs/superpowers/specs/2026-08-25-tinyml-behavior-inference-design.md

## Global Constraints

- Keep inference dependency-free, deterministic, non-blocking, and allocation-free on the ESP32-C3.
- Normalize exactly seven features to [0, 1] in this order: energy, mood, annoyance, boredom, touch_active, interaction_age, tap_burst.
- Preserve the firmware's existing rage, annoyance, and low-energy threshold checks before model advice.
- Apply a model prediction only at confidence >= 0.78 and at most once per second.
- Use only synthetic labels generated from the existing expert policy; do not imply real user telemetry or benchmark accuracy.
- Every production function added in Python must have a behavior-focused unit test written before its implementation.

---

### Task 1: Add failing tests for the model contract

**Files:**

- Create: tests/test_behavior_model.py

**Interfaces:**

- Consumes: the public API that will be added in ml.behavior_model.
- Produces: executable requirements for State.from_raw, expert_intent, softmax, train, predict, and confidence_gate.

- [ ] **Step 1: Write the failing tests**

~~~python
from ml.behavior_model import State, confidence_gate, expert_intent, predict, softmax, train


def test_state_from_raw_clamps_and_normalizes_all_features():
    state = State.from_raw(120, -5, 50, 75, True, 9000, 12)
    assert state.features == (1.0, 0.0, 0.5, 0.75, 1.0, 1.0, 1.0)


def test_expert_policy_preserves_safety_priority():
    assert expert_intent(State((0.5, 0.5, 0.9, 0.9, 0, 1, 0))) == "ANGRY"
    assert expert_intent(State((0.5, 0.5, 0.5, 0.9, 0, 1, 0))) == "ANNOYED"
    assert expert_intent(State((0.1, 0.5, 0.0, 0.0, 0, 1, 0))) == "SLEEPY"


def test_softmax_is_stable_and_sums_to_one():
    probabilities = softmax([1000.0, 999.0, 998.0])
    assert abs(sum(probabilities) - 1.0) < 1e-9
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_training_is_deterministic_and_predicts_clear_states():
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
    assert first == second
    assert predict(first, examples[2]).intent == "PLAYFUL"


def test_confidence_gate_rejects_uncertain_predictions():
    assert confidence_gate(0.779, threshold=0.78) is False
    assert confidence_gate(0.780, threshold=0.78) is True
~~~

- [ ] **Step 2: Run the tests to verify they fail for the missing module**

Run: python3 -m unittest discover -s tests -v

Expected: collection fails with ModuleNotFoundError: No module named ml.

- [ ] **Step 3: Commit the red tests**

~~~bash
git add tests/test_behavior_model.py
git commit -m "test: define TinyML behavior model contract"
~~~

### Task 2: Implement deterministic Python training and inference

**Files:**

- Create: ml/__init__.py
- Create: ml/behavior_model.py

**Interfaces:**

- Consumes: the tests from Task 1.
- Produces: State, Prediction, softmax, expert_intent, generate_dataset, train, predict, confidence_gate, save_model, and load_model.

- [ ] **Step 1: Implement normalization and the public data types**

~~~python
FEATURE_NAMES = (
    "energy", "mood", "annoyance", "boredom",
    "touch_active", "interaction_age", "tap_burst",
)
CLASS_NAMES = ("IDLE", "ANGRY", "ANNOYED", "SLEEPY", "ATTENTION", "PLAYFUL")


@dataclass(frozen=True)
class State:
    features: tuple[float, ...]

    @classmethod
    def from_raw(cls, energy, mood, annoyance, boredom, touch_active,
                 interaction_age_ms, tap_burst):
        return cls((
            clamp(energy / 100.0),
            clamp(mood / 100.0),
            clamp(annoyance / 100.0),
            clamp(boredom / 100.0),
            1.0 if touch_active else 0.0,
            clamp(interaction_age_ms / 5000.0),
            clamp(tap_burst / 8.0),
        ))
~~~

- [ ] **Step 2: Run the focused normalization test and finish only that implementation**

Run: python3 -m unittest tests.test_behavior_model.test_state_from_raw_clamps_and_normalizes_all_features -v

Expected: after adding the module skeleton, the test fails only on an unimplemented normalization method; finish that method before moving on.

- [ ] **Step 3: Implement policy labeling, stable softmax, deterministic gradient descent, inference, and persistence**

~~~python
def expert_intent(state):
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
~~~

The trainer will prepend a bias term, iterate examples in fixed input order, use stable softmax with logit minus max(logit), and apply L2 regularization to weights but not biases. No random initialization or external package is permitted.

- [ ] **Step 4: Run all Python tests and refactor only while green**

Run: python3 -m unittest discover -s tests -v

Expected: all Task 1 tests pass with zero failures.

- [ ] **Step 5: Commit the implementation**

~~~bash
git add ml tests/test_behavior_model.py
git commit -m "feat: add deterministic TinyML behavior model"
~~~

### Task 3: Generate and verify the embedded model artifact

**Files:**

- Create: tools/export_behavior_model.py
- Create: ml/behavior_model.h
- Create: ml/behavior_model.json
- Create: tests/cpp_smoke.cpp

**Interfaces:**

- Consumes: ml.behavior_model.generate_dataset, train, and CLASS_NAMES.
- Produces: cozmo_ml::Features, cozmo_ml::Prediction, and cozmo_ml::predict in the generated header; --check verifies committed artifacts are reproducible.

- [ ] **Step 1: Add failing artifact and C++ smoke tests**

The Python tests will invoke the exporter into a temporary directory and assert that it emits all seven feature names, all six class names, finite coefficients, and deterministic bytes. The C++ smoke test will compile this program:

~~~cpp
#include "ml/behavior_model.h"
#include <cassert>

int main() {
  const cozmo_ml::Features features = {{1.0f, 0.9f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
  const cozmo_ml::Prediction prediction = cozmo_ml::predict(features);
  assert(prediction.confidence >= 0.0f && prediction.confidence <= 1.0f);
  assert(static_cast<int>(prediction.intent) >= 0);
}
~~~

- [ ] **Step 2: Run the new tests and verify they fail because artifacts do not exist**

Run: python3 -m unittest discover -s tests -v

Expected: artifact tests fail with missing exporter/header; g++ -std=c++17 -I. tests/cpp_smoke.cpp -o /tmp/cozmo-model-smoke fails because ml/behavior_model.h is absent.

- [ ] **Step 3: Implement the exporter and generated files**

tools/export_behavior_model.py will accept --output-dir and --check. The generated header will contain constexpr feature/class counts, a fixed float weight matrix and bias vector, and an allocation-free predict function using expf. JSON metadata will include exact class/feature order, training hyperparameters, and dataset size.

- [ ] **Step 4: Generate artifacts and run both test layers**

Run:

~~~bash
python3 tools/export_behavior_model.py
python3 -m unittest discover -s tests -v
g++ -std=c++17 -Wall -Wextra -Werror -I. tests/cpp_smoke.cpp -o /tmp/cozmo-model-smoke
/tmp/cozmo-model-smoke
python3 tools/export_behavior_model.py --check
~~~

Expected: all Python tests pass, the C++ process exits 0, and --check reports that committed artifacts match deterministic regeneration.

- [ ] **Step 5: Commit the artifacts and exporter**

~~~bash
git add ml tools tests/cpp_smoke.cpp
git commit -m "feat: export TinyML model for embedded inference"
~~~

### Task 4: Integrate confidence-gated inference into the firmware

**Files:**

- Modify: CozmoMini_ver_2.1.ino near library includes, global timers, and updateBrain().
- Modify: tests/test_behavior_model.py with source-level integration assertions.

**Interfaces:**

- Consumes: ml/behavior_model.h from Task 3.
- Produces: one-second, 0.78 confidence-gated advisory decision flow that never precedes existing safety checks.

- [ ] **Step 1: Add failing source assertions**

~~~python
def test_firmware_places_ml_advice_after_safety_thresholds():
    source = Path("CozmoMini_ver_2.1.ino").read_text()
    assert '#include "ml/behavior_model.h"' in source
    assert "MODEL_CONFIDENCE = 0.78" in source
    assert "MODEL_DECISION_INTERVAL_MS = 1000" in source
    assert source.index("energy <= ENERGY_SLEEPY") < source.index("cozmo_ml::predict")
~~~

- [ ] **Step 2: Run the focused test and verify it fails because firmware is not integrated**

Run: python3 -m unittest tests.test_behavior_model.FirmwareIntegrationTests -v

Expected: assertions fail on the missing header include and model call.

- [ ] **Step 3: Add the include, timer/configuration, feature construction, and guarded mode mapping**

The firmware will construct features from current drives and timers, call cozmo_ml::predict, and map only ANGRY, ANNOYED, SLEEPY, ATTENTION, and PLAYFUL. It will ignore IDLE, avoid re-entering the current mode, and leave the pre-existing threshold chain first in updateBrain().

- [ ] **Step 4: Run Python tests and inspect the firmware diff**

Run: python3 -m unittest discover -s tests -v and git diff --check.

Expected: all tests pass, no whitespace errors, and the diff shows no changed pins, servo limits, or gesture thresholds.

- [ ] **Step 5: Commit the firmware integration**

~~~bash
git add CozmoMini_ver_2.1.ino tests/test_behavior_model.py
git commit -m "feat: use confidence-gated behavior inference"
~~~

### Task 5: Add recruiter-facing documentation and CI

**Files:**

- Modify: README.md with TinyML architecture, setup, commands, limitations, and repository layout.
- Create: .github/workflows/test.yml
- Modify: tests/test_behavior_model.py with documentation assertions.

**Interfaces:**

- Consumes: stable commands and files from Tasks 1–4.
- Produces: a public explanation a recruiter can follow from training data to on-device inference, plus a reproducible GitHub check.

- [ ] **Step 1: Add documentation checks**

Extend tests to assert README includes python3 tools/export_behavior_model.py, --check, 0.78, policy distillation, and the ml/ directory. Run the new assertions and verify they fail before the documentation exists.

- [ ] **Step 2: Write the README section and workflow**

The workflow will use actions/checkout@v4, set up Python 3.11, run python3 tools/export_behavior_model.py --check, run python3 -m unittest discover -s tests -v, install no third-party package, and compile/run the C++ smoke test with g++.

- [ ] **Step 3: Run the complete local verification suite**

Run:

~~~bash
git diff --check
python3 tools/export_behavior_model.py --check
python3 -m unittest discover -s tests -v
g++ -std=c++17 -Wall -Wextra -Werror -I. tests/cpp_smoke.cpp -o /tmp/cozmo-model-smoke
/tmp/cozmo-model-smoke
arduino-cli compile --fqbn esp32:esp32:esp32c3 .
~~~

Expected: all commands exit 0. If Arduino CLI cannot resolve existing third-party libraries or board package, report that exact limitation separately; do not weaken the Python/C++ gates.

- [ ] **Step 4: Review the final diff and commit documentation/CI**

~~~bash
git diff origin/main...HEAD --stat
git diff origin/main...HEAD --check
git status --short --branch
git add README.md .github/workflows/test.yml tests/test_behavior_model.py
git commit -m "docs: document TinyML workflow and add CI"
~~~

### Task 6: Push the branch and verify GitHub status

**Files:**

- No source files; remote branch and CI state only.

**Interfaces:**

- Consumes: the verified local branch from Tasks 1–5.
- Produces: pushed branch feat/tinyml-behavior-inference and a GitHub Actions run that is green, if the account has push permission.

- [ ] **Step 1: Verify identity and branch before publishing**

Run: git config user.name, git config user.email, git branch --show-current, and git status --short.

Expected: Gung, guefef17@gmail.com, feat/tinyml-behavior-inference, and a clean worktree.

- [ ] **Step 2: Push only the feature branch**

~~~bash
git push -u origin feat/tinyml-behavior-inference
~~~

- [ ] **Step 3: Poll GitHub Actions until completion**

Use gh run list --branch feat/tinyml-behavior-inference and gh run watch <run-id> --exit-status when the GitHub CLI is authenticated. If it is unavailable, report the pushed branch and the local verification evidence instead of claiming remote CI is green.

- [ ] **Step 4: Hand off the contribution**

Report the branch, commit SHAs, changed files, local test results, remote CI result, and exact limitation if a pull request cannot be opened automatically. Do not add an assistant identity, co-author trailer, or contributor entry.

