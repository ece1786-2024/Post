import os

from dotenv import load_dotenv

from gpt_moderator_eval import GPTModeratorEval
from main import DatasetManager


def evaluate():
    load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    dataset_manager = DatasetManager()
    dataset, image_dir, annotations_file = dataset_manager.load_dataset()

    print("\n\nEvaluation start:")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # Import GPTModeratorEval and evaluate the moderator
    evaluator = GPTModeratorEval(OPENAI_API_KEY)
    test_dataset = evaluator.get_test_data(annotations_file)
    metrics = evaluator.evaluate_moderator(test_dataset)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}%")


if __name__ == "__main__":
    evaluate()