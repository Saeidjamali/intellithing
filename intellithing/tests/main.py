import argparse
from utils.config import load_config
from evaluators.correctness_evaluator import CorrectnessEvaluator
from evaluators.faithfulness_evaluator import FaithfulnessEvaluator
from evaluators.relevancy_evaluator import RelevancyEvaluator

def evaluate_correctness(query, reference_answers=None, context=None, source_nodes=None, pdf_path=None, database_query=None, threshold=80, config_file='config.yaml'):
    config = load_config(config_file)
    evaluator = CorrectnessEvaluator(
        threshold=threshold,
        db_config=config['database'],
        inference_api_url=config['inference_api']['url']
    )
    pdf_bytes = None
    if pdf_path:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
    generated_answer = evaluator.perform_remote_inference(query)
    results = evaluator.evaluate(
        query=query,
        generated_answer=generated_answer,
        reference_answers=reference_answers,
        context=context,
        source_nodes=source_nodes,
        pdf_bytes=pdf_bytes,
        database_query=database_query
    )
    return results

def evaluate_faithfulness(query, context=None, model_name="distilbert-base-uncased-distilled-squad", config_file='config.yaml'):
    config = load_config(config_file)
    evaluator = FaithfulnessEvaluator(model_name=model_name)
    generated_answer = CorrectnessEvaluator(inference_api_url=config['inference_api']['url']).perform_remote_inference(query)
    results = evaluator.evaluate(
        query=query,
        generated_answer=generated_answer,
        context=context or []
    )
    return results

def evaluate_relevancy(query, source_nodes=None, model_name="all-MiniLM-L6-v2", config_file='config.yaml'):
    config = load_config(config_file)
    evaluator = RelevancyEvaluator(model_name=model_name)
    generated_answer = CorrectnessEvaluator(inference_api_url=config['inference_api']['url']).perform_remote_inference(query)
    results = evaluator.evaluate(
        query=query,
        generated_answer=generated_answer,
        source_nodes=source_nodes or []
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Intellithing Library Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Correctness subcommand
    parser_correctness = subparsers.add_parser('evaluate_correctness', help='Evaluate correctness of a model\'s response')
    parser_correctness.add_argument('--query', required=True, help='The query to be evaluated.')
    parser_correctness.add_argument('--reference_answers', nargs='*', help='The reference answers for correctness evaluation.')
    parser_correctness.add_argument('--context', nargs='*', help='The context for faithfulness evaluation.')
    parser_correctness.add_argument('--source_nodes', nargs='*', help='The source nodes for relevancy evaluation.')
    parser_correctness.add_argument('--pdf_path', help='Path to the PDF file for extracting context and source nodes.')
    parser_correctness.add_argument('--database_query', help='The database query for extracting context and source nodes.')
    parser_correctness.add_argument('--threshold', type=int, default=80, help='Threshold for correctness evaluation.')
    parser_correctness.add_argument('--config_file', default='config.yaml', help='Path to the config file.')
    parser_correctness.set_defaults(func=lambda args: print(evaluate_correctness(
        query=args.query,
        reference_answers=args.reference_answers,
        context=args.context,
        source_nodes=args.source_nodes,
        pdf_path=args.pdf_path,
        database_query=args.database_query,
        threshold=args.threshold,
        config_file=args.config_file
    )))

    # Faithfulness subcommand
    parser_faithfulness = subparsers.add_parser('evaluate_faithfulness', help='Evaluate faithfulness of a model\'s response')
    parser_faithfulness.add_argument('--query', required=True, help='The query to be evaluated.')
    parser_faithfulness.add_argument('--context', nargs='*', help='The context for faithfulness evaluation.')
    parser_faithfulness.add_argument('--model_name', default="distilbert-base-uncased-distilled-squad", help='Model name for faithfulness evaluation.')
    parser_faithfulness.add_argument('--config_file', default='config.yaml', help='Path to the config file.')
    parser_faithfulness.set_defaults(func=lambda args: print(evaluate_faithfulness(
        query=args.query,
        context=args.context,
        model_name=args.model_name,
        config_file=args.config_file
    )))

    # Relevancy subcommand
    parser_relevancy = subparsers.add_parser('evaluate_relevancy', help='Evaluate relevancy of a model\'s response')
    parser_relevancy.add_argument('--query', required=True, help='The query to be evaluated.')
    parser_relevancy.add_argument('--source_nodes', nargs='*', help='The source nodes for relevancy evaluation.')
    parser_relevancy.add_argument('--model_name', default="all-MiniLM-L6-v2", help='Model name for relevancy evaluation.')
    parser_relevancy.add_argument('--config_file', default='config.yaml', help='Path to the config file.')
    parser_relevancy.set_defaults(func=lambda args: print(evaluate_relevancy(
        query=args.query,
        source_nodes=args.source_nodes,
        model_name=args.model_name,
        config_file=args.config_file
    )))

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
