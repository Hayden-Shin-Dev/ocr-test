import unittest

from fastapi.testclient import TestClient

from ocr_test.app import app


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["engine"], "paddleocr")

    def test_samples_are_discovered_without_document_specific_rules(self) -> None:
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(payload["count"], 0)
        if payload["items"]:
            self.assertIn("path", payload["items"][0])
            self.assertIn("document_type", payload["items"][0])

    def test_sample_path_traversal_is_rejected(self) -> None:
        response = self.client.post("/api/ocr/sample", json={"path": "../../secret.png"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
