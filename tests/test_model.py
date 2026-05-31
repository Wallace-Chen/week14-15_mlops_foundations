import json
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from financial_mlops.model import load_metadata, load_model, predict


class ModelTests(unittest.TestCase):
    def test_artifacts_exist_and_metadata_is_complete(self):
        self.assertTrue((PROJECT_ROOT / "models" / "model.pkl").exists())
        metadata = load_metadata(PROJECT_ROOT / "models" / "metadata.json")
        self.assertEqual(metadata["model_name"], "spy_direction_baseline")
        self.assertEqual(metadata["target"], "next_day_direction")
        self.assertGreater(len(metadata["features"]), 10)
        self.assertIn("training_data", metadata)

    def test_model_loads_and_predicts_sample_request(self):
        load_model(PROJECT_ROOT / "models" / "model.pkl")
        payload = json.loads((PROJECT_ROOT / "data" / "sample_request.json").read_text())
        result = predict(
            payload["features"],
            PROJECT_ROOT / "models" / "model.pkl",
            PROJECT_ROOT / "models" / "metadata.json",
        )
        self.assertIn(result["predicted_direction"], [0, 1])
        self.assertGreaterEqual(result["probability_up"], 0.0)
        self.assertLessEqual(result["probability_up"], 1.0)


if __name__ == "__main__":
    unittest.main()
