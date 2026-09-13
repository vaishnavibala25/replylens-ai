import unittest

from services.evaluation_service import evaluate_response
from services.llm_service import LLMService


class DeterministicEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = LLMService()

    def evaluate(self, email, response, reply=None):
        examples = []
        if reply:
            examples = [{
                "similarity": 0.95,
                "incoming_email": email,
                "historical_reply": reply,
            }]
        return evaluate_response(email, examples, response, self.llm)

    def test_good_beats_useless_wrong_intent_and_hallucination(self):
        email = "Hi, I was charged twice for order ORD-88221. Can you refund the duplicate charge?"
        good = self.evaluate(
            email,
            "Hi there, I'm sorry for the duplicate charge on order ORD-88221. "
            "I've refunded the extra charge; it should appear in 5-7 business days. Thanks.",
        )
        useless = self.evaluate(email, "Hi, thanks for your message. We are sorry to hear that.")
        wrong_intent = self.evaluate(email, "Hi, please use the Forgot Password link to reset your password.")
        hallucination = self.evaluate(email, "Hi, I refunded $15,999 and the money will arrive tomorrow.")

        self.assertGreater(good["overall_score"], useless["overall_score"])
        self.assertGreater(good["overall_score"], wrong_intent["overall_score"])
        self.assertGreater(good["overall_score"], hallucination["overall_score"])
        self.assertLess(hallucination["dimensions"]["factual_consistency"]["score"], 100)

    def test_incomplete_multi_request_is_lower_than_complete_response(self):
        email = "Hi, please cancel my subscription, refund the latest payment, and confirm I will not be charged again."
        incomplete = self.evaluate(email, "Hi, your subscription has been cancelled.")
        complete = self.evaluate(
            email,
            "Hi, your subscription has been cancelled and the payment has been refunded. "
            "You will not be charged again. Thanks for reaching out.",
        )
        self.assertGreater(complete["overall_score"], incomplete["overall_score"])

    def test_semantic_equivalence_scores_close_to_reference(self):
        email = "Hi, I need a refund for order ORD-48213."
        reference = "Your refund has been processed and should arrive within 5-7 business days."
        equivalent = "We've initiated your refund. It should be credited within approximately 5-7 working days."
        reference_score = self.evaluate(email, reference, reference)["overall_score"]
        equivalent_score = self.evaluate(email, equivalent, reference)["overall_score"]
        self.assertLessEqual(abs(reference_score - equivalent_score), 12)


if __name__ == "__main__":
    unittest.main()