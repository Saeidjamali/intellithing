from fuzzywuzzy import fuzz
from PyPDF2 import PdfFileReader
from io import BytesIO
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import spacy
import requests
import openai
from llama_index.evaluation import DatasetGenerator
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext
from ..utils.config import load_config

class CorrectnessEvaluator:
    def __init__(self, config_file='config.yaml', threshold=80):
        config = load_config(config_file)
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_sm")
        self.inference_api_url = config['inference_api']['url']
        self.openai_api_key = config['openai']['api_key']
        self.openai_model = config['openai']['model']
        openai.api_key = self.openai_api_key
        db_config = config['database']
        db_url = self.build_db_url(db_config)
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.data_generator = DatasetGenerator.from_documents(self.load_documents())

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

    def load_documents(self):
        reader = SimpleDirectoryReader("./data")
        return reader.load_data()

    def generate_questions_from_documents(self):
        questions_and_answers = []
        for doc in self.load_documents():
            text = doc.get('text')
            if text:
                questions = self.generate_questions(text)
                questions_and_answers.extend(questions)
        return questions_and_answers

    def generate_questions(self, text):
        response = openai.Completion.create(
            engine=self.openai_model,
            prompt=f"Generate a list of questions based on the following text:\n{text}",
            max_tokens=150
        )
        questions = response.choices[0].text.strip().split('\n')
        return [(q.strip(), self.answer_question(q.strip(), text)) for q in questions if q.strip()]

    def answer_question(self, question, context):
        response = openai.Completion.create(
            engine=self.openai_model,
            prompt=f"Answer the following question based on the context:\nContext: {context}\nQuestion: {question}",
            max_tokens=100
        )
        answer = response.choices[0].text.strip()
        return answer

    def evaluate(self, context=None, pdf_path=None, database_query=None):
        if pdf_path:
            context = self.extract_text_from_pdf(pdf_path)
        if database_query:
            context = self.fetch_data_from_database(database_query)
        
        qa_pairs = self.generate_questions_from_documents()
        
        evaluation_results = []
        for question, correct_answer in qa_pairs:
            generated_answer = self.perform_remote_inference(question)
            score = fuzz.ratio(generated_answer, correct_answer)
            is_correct = score >= self.threshold
            evaluation_results.append({
                "question": question,
                "generated_answer": generated_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "score": score
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

    def perform_remote_inference(self, query):
        response = requests.post(self.inference_api_url, json={'query': query})
        if response.status_code != 200:
            raise requests.exceptions.RequestException("Failed to get response from inference API")
        return response.json().get('generated_answer', '').strip()
