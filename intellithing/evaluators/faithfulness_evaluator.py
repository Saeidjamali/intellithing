# evaluators/faithfulness_evaluator.py
from transformers import pipeline

class FaithfulnessEvaluator:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering")

    def evaluate(self, query, generated_answer, context):
        answers = []
        for passage in context:
            result = self.qa_pipeline(question=query, context=passage)
            answers.append(result['answer'])

        is_faithful = generated_answer in answers

        return {
            "query": query,
            "generated_answer": generated_answer,
            "context": context,
            "is_faithful": is_faithful,
            "supporting_contexts": answers if is_faithful else []
        }
