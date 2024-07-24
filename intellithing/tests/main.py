from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from PyPDF2 import PdfFileReader
from io import BytesIO
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from intellithing import Intellithing

app = FastAPI()
evaluator = Intellithing()

# Replace with your model's inference API endpoint
INFERENCE_API_URL = 'http://your-inference-api-endpoint.com/inference'

# Database setup
DATABASE_URL = 'sqlite:///example.db'  # Replace with your database URL
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

class EvaluationRequest(BaseModel):
    query: str
    reference_answers: list[str] = None
    context: list[str] = None
    source_nodes: list[str] = None
    database_query: str = None

@app.post("/evaluate")
async def evaluate(
    request: EvaluationRequest,
    pdf_file: UploadFile = File(None)
):
    query = request.query
    reference_answers = request.reference_answers
    context = request.context
    source_nodes = request.source_nodes

    if pdf_file:
        context, source_nodes = extract_text_from_pdf(await pdf_file.read())

    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    if not reference_answers and not context and not source_nodes and not request.database_query:
        raise HTTPException(status_code=400, detail="Missing evaluation data")

    if request.database_query:
        context, source_nodes = fetch_data_from_database(request.database_query)

    # Perform inference using the remote API
    generated_answer = perform_remote_inference(query)

    # Evaluate the generated answer
    results = evaluator.evaluate(
        query=query,
        generated_answer=generated_answer,
        reference_answers=reference_answers or [],
        context=context or [],
        source_nodes=source_nodes or []
    )

    return {
        'generated_answer': generated_answer,
        'evaluation': results
    }

def perform_remote_inference(query):
    # Simulating a call to a remote inference API
    response = requests.post(INFERENCE_API_URL, json={'query': query})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to get response from inference API")
    return response.json().get('generated_answer', '').strip()

def extract_text_from_pdf(pdf_bytes):
    reader = PdfFileReader(BytesIO(pdf_bytes))
    text = ""
    for page_num in range(reader.getNumPages()):
        text += reader.getPage(page_num).extractText()
    return [text], [text]

def fetch_data_from_database(query):
    # Example function to fetch data from a database
    result = session.execute(query).fetchall()
    context = [row[0] for row in result]
    source_nodes = context  # Modify as needed
    return context, source_nodes

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
