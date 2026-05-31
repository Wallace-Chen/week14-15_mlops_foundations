import json
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from financial_mlops.features import FeatureValidationError, request_to_feature_vector
from financial_mlops.model import load_metadata


class FeatureTests(unittest.TestCase):
    def test_sample_request_matches_metadata_order(self):
        metadata = load_metadata(PROJECT_ROOT / "models" / "metadata.json")
        payload = json.loads((PROJECT_ROOT / "data" / "sample_request.json").read_text())
        vector = request_to_feature_vector(payload, metadata["features"])
        self.assertEqual(vector.shape, (1, len(metadata["features"])))

    def test_missing_feature_is_rejected(self):
        metadata = load_metadata(PROJECT_ROOT / "models" / "metadata.json")
        payload = json.loads((PROJECT_ROOT / "data" / "sample_request.json").read_text())
        payload.pop(metadata["features"][0])
        with self.assertRaises(FeatureValidationError):
            request_to_feature_vector(payload, metadata["features"])


if __name__ == "__main__":
    unittest.main()
