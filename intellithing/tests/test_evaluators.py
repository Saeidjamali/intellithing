# tests/test_evaluators.py
import unittest
from intellithing import Intellithing

class TestIntellithing(unittest.TestCase):
    def setUp(self):
        self.evaluator = Intellithing()

    def test_correctness(self):
        result = self.evaluator.evaluate(
            query="What is the capital of France?",
            generated_answer="Paris",
            reference_answers=["Paris", "paris"],
            context=["The capital of France is Paris."],
            source_nodes=["Paris", "London", "Berlin"]
        )
        self.assertTrue(result["correctness"]["is_correct"])

    def test_faithfulness(self):
        result = self.evaluator.evaluate(
            query="What is the capital of France?",
            generated_answer="Paris",
            reference_answers=["Paris"],
            context=["The capital of France is Paris."],
            source_nodes=["Paris", "London", "Berlin"]
        )
        self.assertTrue(result["faithfulness"]["is_faithful"])

    def test_relevancy(self):
        result = self.evaluator.evaluate(
            query="What is the capital of France?",
            generated_answer="Paris",
            reference_answers=["Paris"],
            context=["The capital of France is Paris."],
            source_nodes=["Paris", "London", "Berlin"]
        )
        self.assertTrue(result["relevancy"]["relevancy_scores"][0] > 0.5)

if __name__ == '__main__':
    unittest.main()
