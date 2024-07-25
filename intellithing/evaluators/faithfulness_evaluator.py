from transformers import pipeline
import requests
from PyPDF2 import PdfFileReader
from io import BytesIO
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..utils.config import load_config

class FaithfulnessEvaluator:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", config_file='config.yaml'):
        config = load_config(config_file)
        self.qa_pipeline = pipeline("question-answering", model=model_name)
        self.inference_api_url = config['inference_api']['url']
        db_config = config['database']
        db_url = self.build_db_url(db_config)
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def build_db_url(self, db_config):
        db_type = db_config.get('type', 'sqlite')
        username = db_config.get('username', '')
        password = db_config.get('password', '')
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', '')
        database = db_config.get('database', '')
        if db_type == 'sqlite':
            return f"sqlite:///{database}"
        return f"{db_type}://{username}:{password}@{host}:{port}/{database}"

    def evaluate(self, context=None, pdf_path=None, database_query=None):
        if pdf_path:
            context = self.extract_text_from_pdf(pdf_path)
        if database_query:
            context = self.fetch_data_from_database(database_query)

        evaluation_results = []

        for passage in context:
            generated_answer = self.perform_remote_inference(passage)
            result = self.qa_pipeline(question=passage, context=passage)
            is_faithful = generated_answer == result['answer']
            evaluation_results.append({
                "context": passage,
                "generated_answer": generated_answer,
                "expected_answer": result['answer'],
                "is_faithful": is_faithful
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
