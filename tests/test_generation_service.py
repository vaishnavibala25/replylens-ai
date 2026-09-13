import unittest
from pathlib import Path

import pandas as pd

from services.generation_service import generate_response
from services.retrieval_service import RetrievalService


class OfflineLLM:
    is_configured = False


class DemoGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = Path(__file__).parent.parent / "data" / "emails.csv"
        cls.dataframe = pd.read_csv(data_path)
        cls.retrieval = RetrievalService(cls.dataframe)
        cls.llm = OfflineLLM()

    def test_demo_response_is_not_top_reply(self):
        email = "Hi, I need to update the shipping address for order ORD-40551 before it ships."
        retrieved = self.retrieval.retrieve(email, top_k=1)
        generated = generate_response(email, retrieved, self.llm)

        self.assertTrue(generated["is_demo"])
        self.assertNotEqual(generated["response_text"], retrieved[0]["historical_reply"])
        self.assertNotIn(retrieved[0]["historical_reply"], generated["response_text"])

    def test_demo_response_addresses_email_and_uses_grounding(self):
        email = "Hi, I need to update the shipping address for order ORD-40551 before it ships."
        retrieved = self.retrieval.retrieve(email, top_k=1)
        response = generate_response(email, retrieved, self.llm)["response_text"].lower()

        self.assertIn("ord-40551", response)
        self.assertIn("shipping address", response)
        self.assertIn(retrieved[0]["category"], {"address_change", "order_status"})

    def test_demo_outputs_do_not_contain_evaluation_replies(self):
        evaluation = self.dataframe[self.dataframe["split"] == "evaluation"]
        evaluation_replies = set(evaluation["historical_reply"])

        for _, row in evaluation.iterrows():
            retrieved = self.retrieval.retrieve(row["incoming_email"], top_k=3)
            response = generate_response(row["incoming_email"], retrieved, self.llm)["response_text"]
            self.assertFalse(any(reply in response for reply in evaluation_replies))


if __name__ == "__main__":
    unittest.main()