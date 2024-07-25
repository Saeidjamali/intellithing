# Intellithing Documentation

`intellithing` is a versatile library designed to evaluate model responses based on correctness, faithfulness, and relevancy. It provides tools to evaluate generated answers against predefined answers, context, or source nodes using various methodologies. The library is designed to be used both programmatically and via the command line.

## Installation

To install the `intellithing` library, run:

```bash
pip install -e .
```

## Configuration

Ensure you have a `config.yaml` file with the following structure:

```yaml
database:
  type: sqlite
  username: ""
  password: ""
  host: "localhost"
  port: ""
  database: "path/to/your/database.db"

inference_api:
  url: "http://your_inference_api_url"

openai:
  api_key: "your_openai_api_key"
  model: "gpt-4"
```

## Methodology

### Correctness Evaluation

The `CorrectnessEvaluator` assesses the correctness of a generated answer by generating questions from context (e.g., PDF files or database queries) and comparing the generated answers with the correct answers.

1. **Context Extraction**:
   - **PDF Extraction**: Extracts text from PDF files using `PyPDF2`.
   - **Database Queries**: Executes queries to fetch context.

2. **Question Generation**:
   - Uses OpenAI's GPT-4 to generate questions from the extracted context.

3. **Answer Comparison**:
   - The generated answers are compared with the correct answers using fuzzy matching (`fuzzywuzzy`).

#### Technical Steps:
- **Text Extraction**: Extract text from PDF using `PyPDF2`.
- **Database Query Execution**: Execute SQL queries to fetch data.
- **Question Generation**: Generate questions using OpenAI GPT-4.
- **Answer Comparison**: Compare answers using fuzzy matching.

### Faithfulness Evaluation

The `FaithfulnessEvaluator` checks if the generated answer is supported by the provided context using a question-answering model.

1. **Context Extraction**:
   - **PDF Extraction**: Extracts context from PDF files.
   - **Database Queries**: Extracts context from databases.

2. **Answer Generation**:
   - Uses a pre-trained QA model (like `distilbert-base-uncased-distilled-squad`) to generate answers from the context.

3. **Faithfulness Check**:
   - Compares the model's generated answers with the context-derived answers to check for consistency.

#### Technical Steps:
- **Text Extraction**: Extract text from PDF using `PyPDF2`.
- **Database Query Execution**: Execute SQL queries to fetch data.
- **Answer Generation**: Generate answers using a pre-trained QA model.
- **Faithfulness Check**: Compare generated answers with context-derived answers.

### Relevancy Evaluation

The `RelevancyEvaluator` evaluates the relevancy of the generated answer to a set of source nodes using sentence embeddings.

1. **Context Extraction**:
   - **PDF Extraction**: Extracts context from PDF files.
   - **Database Queries**: Extracts context from databases.

2. **Embedding Calculation**:
   - Uses `SentenceTransformer` to convert context and generated answers into embeddings.

3. **Similarity Measurement**:
   - Calculates cosine similarity between embeddings to assess relevancy.

#### Technical Steps:
- **Text Extraction**: Extract text from PDF using `PyPDF2`.
- **Database Query Execution**: Execute SQL queries to fetch data.
- **Embedding Calculation**: Calculate embeddings using `SentenceTransformer`.
- **Similarity Measurement**: Measure similarity using cosine similarity.

## Usage Scenarios

### Command Line Usage

The `intellithing` CLI tool allows you to evaluate model responses directly from the command line.

#### Correctness Evaluation

**Using a PDF File**:

```bash
intellithing evaluate_correctness --pdf_path document.pdf
```

**Using a Database Query**:

```bash
intellithing evaluate_correctness --database_query "SELECT content FROM documents WHERE id=1"
```

#### Faithfulness Evaluation

**Using a PDF File**:

```bash
intellithing evaluate_faithfulness --pdf_path document.pdf
```

**Using a Database Query**:

```bash
intellithing evaluate_faithfulness --database_query "SELECT content FROM documents WHERE id=1"
```

#### Relevancy Evaluation

**Using a PDF File**:

```bash
intellithing evaluate_relevancy --pdf_path document.pdf
```

**Using a Database Query**:

```bash
intellithing evaluate_relevancy --database_query "SELECT content FROM documents WHERE id=1"
```

### Full Evaluation

You can also perform a full evaluation (correctness, faithfulness, and relevancy) in one command:

**Using a PDF File (Command Line)**:

```bash
intellithing evaluate_all --pdf_path document.pdf
```

**Using a Database Query (Command Line)**:

```bash
intellithing evaluate_all --database_query "SELECT content FROM documents WHERE id=1"
```

### Programmatic Usage

#### Correctness Evaluation

**Using a PDF File**:

```python
from intellithing import CorrectnessEvaluator

correctness_evaluator = CorrectnessEvaluator(threshold=80)
results = correctness_evaluator.evaluate(
    pdf_path='document.pdf'
)
print(results)
```

**Using a Database Query**:

```python
from intellithing import CorrectnessEvaluator

correctness_evaluator = CorrectnessEvaluator(threshold=80)
results = correctness_evaluator.evaluate(
    database_query="SELECT content FROM documents WHERE id=1"
)
print(results)
```

#### Faithfulness Evaluation

**Using a PDF File**:

```python
from intellithing import FaithfulnessEvaluator

faithfulness_evaluator = FaithfulnessEvaluator()
results = faithfulness_evaluator.evaluate(
    pdf_path='document.pdf'
)
print(results)
```

**Using a Database Query**:

```python
from intellithing import FaithfulnessEvaluator

faithfulness_evaluator = FaithfulnessEvaluator()
results = faithfulness_evaluator.evaluate(
    database_query="SELECT content FROM documents WHERE id=1"
)
print(results)
```

#### Relevancy Evaluation

**Using a PDF File**:

```python
from intellithing import RelevancyEvaluator

relevancy_evaluator = RelevancyEvaluator()
results = relevancy_evaluator.evaluate(
    pdf_path='document.pdf'
)
print(results)
```

**Using a Database Query**:

```python
from intellithing import RelevancyEvaluator

relevancy_evaluator = RelevancyEvaluator()
results = relevancy_evaluator.evaluate(
    database_query="SELECT content FROM documents WHERE id=1"
)
print(results)
```

### Full Evaluation

You can also perform a full evaluation (correctness, faithfulness, and relevancy) programmatically:

**Using a PDF File**:

```python
from intellithing import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator

# Correctness Evaluation
correctness_evaluator = CorrectnessEvaluator(threshold=80)
correctness_results = correctness_evaluator.evaluate(
    pdf_path='document.pdf'
)
print("Correctness Results:", correctness_results)

# Faithfulness Evaluation
faithfulness_evaluator = FaithfulnessEvaluator()
faithfulness_results = faithfulness_evaluator.evaluate(
    pdf_path='document.pdf'
)
print("Faithfulness Results:", faithfulness_results)

# Relevancy Evaluation
relevancy_evaluator = RelevancyEvaluator()
relevancy_results = relevancy_evaluator.evaluate(
    pdf_path='document.pdf'
)
print("Relevancy Results:", relevancy_results)
```

**Using a Database Query**:

```python
from intellithing import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator

# Correctness Evaluation
correctness_evaluator = CorrectnessEvaluator(threshold=80)
correctness_results = correctness_evaluator.evaluate(
    database_query="SELECT content FROM documents WHERE id=1"
)
print("Correctness Results:", correctness_results)

# Faithfulness Evaluation
faithfulness_evaluator = FaithfulnessEvaluator()
faithfulness_results = faithfulness_evaluator.evaluate(
    database_query="SELECT content FROM documents WHERE id=1"
)
print("Faithfulness Results:", faithfulness_results)

# Relevancy Evaluation
relevancy_evaluator = RelevancyEvaluator()
relevancy_results = relevancy_evaluator.evaluate(
    database_query="SELECT content FROM documents WHERE id=1"
)
print("Relevancy Results:", relevancy_results)
```

## Conclusion

The `intellithing` library provides a comprehensive suite of tools to evaluate the correctness, faithfulness, and relevancy of responses generated by language models. By leveraging advanced techniques and models such as OpenAI's GPT-4, users can perform detailed and accurate evaluations to ensure the reliability and accuracy of their generative AI applications.