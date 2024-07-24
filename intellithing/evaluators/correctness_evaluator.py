# evaluators/correctness_evaluator.py
from fuzzywuzzy import fuzz

class CorrectnessEvaluator:
    def __init__(self, threshold=80):
        self.threshold = threshold

    def evaluate(self, query, generated_answer, reference_answers):
        scores = [fuzz.ratio(generated_answer, ref_answer) for ref_answer in reference_answers]
        max_score = max(scores)
        is_correct = max_score >= self.threshold

        return {
            "query": query,
            "generated_answer": generated_answer,
            "reference_answers": reference_answers,
            "is_correct": is_correct,
            "max_score": max_score
        }
