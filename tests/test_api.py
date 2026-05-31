import json
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient
from financial_mlops.service import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample = json.loads((PROJECT_ROOT / "data" / "sample_request.json").read_text())

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["model_loaded"])

    def test_metadata(self):
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_name"], "spy_direction_baseline")
        self.assertGreater(len(payload["features"]), 10)

    def test_predict(self):
        response = self.client.post("/predict", json=self.sample)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in ["ticker", "prediction", "probability", "model_version", "latency_ms", "request_id"]:
            self.assertIn(key, payload)
        self.assertEqual(payload["ticker"], "SPY")
        self.assertIn(payload["prediction"], [0, 1])

    def test_predict_rejects_missing_feature(self):
        broken = dict(self.sample)
        broken["features"] = dict(self.sample["features"])
        broken["features"].pop(next(iter(broken["features"])))
        response = self.client.post("/predict", json=broken)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
