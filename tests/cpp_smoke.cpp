#include "ml/behavior_model.h"

#include <cassert>

int main() {
  const cozmo_ml::Features features = {{1.0f, 0.9f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
  const cozmo_ml::Prediction prediction = cozmo_ml::predict(features);
  assert(prediction.confidence >= 0.0f && prediction.confidence <= 1.0f);
  assert(static_cast<int>(prediction.intent) < cozmo_ml::kClassCount);
  return 0;
}
