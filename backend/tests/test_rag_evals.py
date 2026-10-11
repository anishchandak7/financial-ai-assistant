"""Automated Unit Tests for Reference-Free RAG Evaluator and Faithfulness Guardrail."""

import unittest
from evals.ragas_evaluator import ragas_evaluator
from agents.guardrails.faithfulness_guard import faithfulness_guard


class TestRAGEvaluation(unittest.TestCase):
    """Test reference-free evaluation metrics and citation guardrail."""

    def test_faithfulness_guardrail_valid_quotes(self):
        """Verify faithfulness guardrail accepts quotes present in retrieved context."""
        retrieved_contexts = [
            "Apple reported quarterly revenue of $85.8 billion, up 5 percent year over year.",
            "Services revenue reached an all-time record of $24.2 billion during the quarter."
        ]
        citations = [
            {
                "source_index": 0,
                "quote": "Apple reported quarterly revenue of $85.8 billion",
                "file_name": "AAPL_Q3.pdf"
            },
            {
                "source_index": 1,
                "quote": "Services revenue reached an all-time record of $24.2 billion",
                "file_name": "AAPL_Q3.pdf"
            }
        ]

        verified, score = faithfulness_guard.verify_citations(citations, retrieved_contexts)
        self.assertEqual(len(verified), 2)
        self.assertEqual(score, 1.0)

    def test_faithfulness_guardrail_drops_hallucinated_quotes(self):
        """Verify faithfulness guardrail drops quotes not present in retrieved context."""
        retrieved_contexts = [
            "Apple reported quarterly revenue of $85.8 billion."
        ]
        citations = [
            {
                "source_index": 0,
                "quote": "Apple reported quarterly revenue of $85.8 billion",
                "file_name": "AAPL_Q3.pdf"
            },
            {
                "source_index": 0,
                "quote": "Tesla acquired Apple for 500 billion dollars yesterday",
                "file_name": "AAPL_Q3.pdf"
            }
        ]

        verified, score = faithfulness_guard.verify_citations(citations, retrieved_contexts)
        self.assertEqual(len(verified), 1)
        self.assertEqual(score, 0.5)

    def test_ragas_evaluator_scoring_logic(self):
        """Verify ragas_evaluator structure and fallback handling."""
        contexts = ["The 401(k) contribution limit for 2024 is $23,000 for individuals under 50."]
        query = "What is the 2024 401(k) contribution limit?"
        response = "The 2024 contribution limit for a 401(k) is $23,000 for employees under age 50 [0]."

        scores = ragas_evaluator.evaluate_single_turn(query, contexts, response)
        self.assertIn("faithfulness", scores)
        self.assertIn("answer_relevance", scores)
        self.assertIn("context_relevance", scores)
        self.assertIn("overall_rag_score", scores)
        self.assertGreaterEqual(scores["overall_rag_score"], 0.70)


if __name__ == "__main__":
    unittest.main()
