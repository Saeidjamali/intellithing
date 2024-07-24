# evaluators/relevancy_evaluator.py
from sentence_transformers import SentenceTransformer, util

class RelevancyEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate(self, query, generated_answer, source_nodes):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        answer_embedding = self.model.encode(generated_answer, convert_to_tensor=True)
        source_embeddings = self.model.encode(source_nodes, convert_to_tensor=True)

        relevancy_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)[0]
        relevant_sources = [source_nodes[i] for i in relevancy_scores.topk(5).indices]

        return {
            "query": query,
            "generated_answer": generated_answer,
            "relevancy_scores": relevancy_scores,
            "relevant_sources": relevant_sources
        }
