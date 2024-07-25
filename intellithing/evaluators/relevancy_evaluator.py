from sentence_transformers import SentenceTransformer, util
import requests
from ..utils.config import load_config

class RelevancyEvaluator:
    def __init__(self, model_name="all-MiniLM-L6-v2", config_file='config.yaml'):
        config = load_config(config_file)
        self.model = SentenceTransformer(model_name)
        self.inference_api_url = config['inference_api']['url']

    def evaluate(self, context=None, pdf_path=None, database_query=None):
        if pdf_path:
            context = self.extract_text_from_pdf(pdf_path)
        if database_query:
            context = self.fetch_data_from_database(database_query)

        evaluation_results = []

        for passage in context:
            generated_answer = self.perform_remote_inference(passage)
            query_embedding = self.model.encode(passage, convert_to_tensor=True)
            answer_embedding = self.model.encode(generated_answer, convert_to_tensor=True)
            source_embeddings = self.model.encode(context, convert_to_tensor=True)
            relevancy_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)[0]
            relevant_sources = [context[i] for i in relevancy_scores.topk(5).indices]
            evaluation_results.append({
                "context": passage,
                "generated_answer": generated_answer,
                "relevancy_scores": relevancy_scores.tolist(),
                "relevant_sources": relevant_sources
            })

        return evaluation_results

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PdfFileReader(file)
            text = ""
            for page_num in range(reader.getNumPages()):
                text += reader.getPage(page_num).extractText()
        return [text]

    def fetch_data_from_database(self, query):
        session = self.Session()
        result = session.execute(query).fetchall()
        context = [row[0] for row in result]
        session.close()
        return context

    def perform_remote_inference(self, context):
        response = requests.post(self.inference_api_url, json={'context': context})
        if response.status_code != 200:
            raise requests.exceptions.RequestException("Failed to get response from inference API")
        return response.json().get('generated_answer', '').strip()
