from .layer_freezer import LayerFreezer
from .lr_finder import LearningRateFinderCallback
from .hyper_tuner import HyperparameterTuner
from .autotrainer import AutoTrainer
from .t5customdataset import (
    T5RegressionDataset, T5ClassificationDataset, T5QADataset, 
    T5TextGenerationDataset, T5SummarizationDataset, T5NERDataset, 
    T5TranslationDataset
)
from .evaluators.correctness_evaluator import CorrectnessEvaluator
from .evaluators.faithfulness_evaluator import FaithfulnessEvaluator
from .evaluators.relevancy_evaluator import RelevancyEvaluator
from .utils.config import load_config

class Intellithing:
    def __init__(self):
        self.correctness_evaluator = CorrectnessEvaluator()
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.relevancy_evaluator = RelevancyEvaluator()

    def evaluate(self, query, generated_answer, reference_answers, context, source_nodes):
        correctness_result = self.correctness_evaluator.evaluate(query, generated_answer, reference_answers)
        faithfulness_result = self.faithfulness_evaluator.evaluate(query, generated_answer, context)
        relevancy_result = self.relevancy_evaluator.evaluate(query, generated_answer, source_nodes)
        
        return {
            "correctness": correctness_result,
            "faithfulness": faithfulness_result,
            "relevancy": relevancy_result
        }

__all__ = [
    'LayerFreezer',
    'LearningRateFinderCallback',
    'HyperparameterTuner',
    'AutoTrainer',
    'T5RegressionDataset', 'T5ClassificationDataset', 'T5QADataset',
    'T5TextGenerationDataset', 'T5SummarizationDataset', 'T5NERDataset', 
    'T5TranslationDataset',
    'CorrectnessEvaluator',
    'FaithfulnessEvaluator',
    'RelevancyEvaluator',
    'load_config',
    'Intellithing'
]
