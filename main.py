import argparse
from evaluators.correctness_evaluator import CorrectnessEvaluator
from evaluators.faithfulness_evaluator import FaithfulnessEvaluator
from evaluators.relevancy_evaluator import RelevancyEvaluator

def evaluate_correctness(context=None, pdf_path=None, database_query=None, threshold=80, config_file='config.yaml'):
    evaluator = CorrectnessEvaluator(
        config_file=config_file,
        threshold=threshold
    )
    results = evaluator.evaluate(
        context=context,
        pdf_path=pdf_path,
        database_query=database_query
    )
    return results

def evaluate_faithfulness(context=None, pdf_path=None, database_query=None, model_name="distilbert-base-uncased-distilled-squad", config_file='config.yaml'):
    evaluator = FaithfulnessEvaluator(model_name=model_name, config_file=config_file)
    results = evaluator.evaluate(
        context=context,
        pdf_path=pdf_path,
        database_query=database_query
    )
    return results

def evaluate_relevancy(context=None, pdf_path=None, database_query=None, model_name="all-MiniLM-L6-v2", config_file='config.yaml'):
    evaluator = RelevancyEvaluator(model_name=model_name, config_file=config_file)
    results = evaluator.evaluate(
        context=context,
        pdf_path=pdf_path,
        database_query=database_query
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Intellithing Library Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Correctness subcommand
    parser_correctness = subparsers.add_parser('evaluate_correctness', help='Evaluate correctness of a model\'s response')
    parser_correctness.add_argument('--context', nargs='*', help='The context for generating questions.')
    parser_correctness.add_argument('--pdf_path', help='Path to the PDF file for extracting context and source nodes.')
    parser_correctness.add_argument('--database_query', help='The database query for extracting context and source nodes.')
    parser_correctness.add_argument('--threshold', type=int, default=80, help='Threshold for correctness evaluation.')
    parser_correctness.add_argument('--config_file', default='config.yaml', help='Path to the config file.')
    parser_correctness.set_defaults(func=lambda args: print(evaluate_correctness(
        context=args.context,
        pdf_path=args.pdf_path,
        database_query=args.database_query,
        threshold=args.threshold,
        config_file=args.config_file
    )))

    # Faithfulness subcommand
    parser_faithfulness = subparsers.add_parser('evaluate_faithfulness', help='Evaluate faithfulness of a model\'s response')
    parser_faithfulness.add_argument('--context', nargs='*', help='The context for faithfulness evaluation.')
    parser_faithfulness.add_argument('--pdf_path', help='Path to the PDF file for extracting context.')
    parser_faithfulness.add_argument('--database_query', help='The database query for extracting context.')
    parser_faithfulness.add_argument('--model_name', default="distilbert-base-uncased-distilled-squad", help='Model name for faithfulness evaluation.')
    parser_faithfulness.add_argument('--config_file', default='config.yaml', help='Path to the config file.')
    parser_faithfulness.set_defaults(func=lambda args: print(evaluate_faithfulness(
        context=args.context,
        pdf_path=args.pdf_path,
        database_query=args.database_query,
        model_name=args.model_name,
        config_file=args.config_file
    )))

    # Relevancy subcommand
    parser_relevancy = subparsers.add_parser('evaluate_relevancy', help='Evaluate relevancy of a model\'s response')
    parser_relevancy.add_argument('--context', nargs='*', help='The context for relevancy evaluation.')
    parser_relevancy.add_argument('--pdf_path', help='Path to the PDF file for extracting context.')
    parser_relevancy.add_argument('--database_query', help='The database query for extracting context.')
    parser_relevancy.add_argument('--model_name', default="all-MiniLM-L6-v2", help='Model name for relevancy evaluation.')
    parser_relevancy.add_argument('--config_file', default='config.yaml', help='Path to the config file.')
    parser_relevancy.set_defaults(func=lambda args: print(evaluate_relevancy(
        context=args.context,
        pdf_path=args.pdf_path,
        database_query=args.database_query,
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
