import os

from dotenv import load_dotenv

from gpt_editor_eval import GPTEditorEval
from gpt_moderator_eval import GPTModeratorEval
from main import DatasetManager


def evaluate():
    load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    dataset_manager = DatasetManager()
    dataset, image_dir, annotations_file = dataset_manager.load_dataset()

    print("\n\nEvaluation start:")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # Evaluate the moderator
    evaluator = GPTModeratorEval(OPENAI_API_KEY)
    test_dataset = evaluator.get_test_data(annotations_file)
    metrics = evaluator.evaluate_moderator(test_dataset)
    print("Moderator Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}%")

    # Evaluate the editor
    editor_eval = GPTEditorEval(OPENAI_API_KEY)
    texts, compliants, explanations, _ = editor_eval.get_data('output/evaluation_results.json')
    editor_eval_metrics = editor_eval.evaluate_editor(texts, compliants, explanations)
    print("Editor Evaluation Metrics:")
    for metric, value in editor_eval_metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}%")

if __name__ == "__main__":
    evaluate()