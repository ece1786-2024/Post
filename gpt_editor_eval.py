import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from gpt_moderator import GPTModerator
from gpt_editor import GPTEditor

class GPTEditorEval:
    def __init__(self, OPENAI_API_KEY):
        self.OPENAI_API_KEY = OPENAI_API_KEY

    def get_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        texts = [entry['text'] for entry in data]
        compliants = [entry['compliant'] for entry in data]
        explanations = [entry['explanations'] for entry in data]
        true_labels = [entry['true_label'] for entry in data]
        return texts, compliants, explanations, true_labels


    def evaluate_editor(self, texts, compliants, explanations):
        editor = GPTEditor(self.OPENAI_API_KEY)
        edited_texts = []
        all_edits = []  # Collect all edits for saving

        for t, c, e in zip(texts, compliants, explanations):
            if c == False:
                try:
                    edit = editor.edit_text(t, e)
                    print(edit)
                    print("")
                    edited_texts.append(edit['revised_text'])
                    all_edits.append(edit)  # Save the entire edit dictionary
                except Exception as ex:
                    print(f"Error editing text: {t}\nException: {ex}")
                    continue

        moderator = GPTModerator(self.OPENAI_API_KEY)
        predictions = []
        true_labels = [0] * len(edited_texts)  # Assuming all edited texts are true labels of Non-Hate (0)

        for text in edited_texts:
            try:
                pred = moderator.moderate_text(text)
                predicted_label = 0 if pred["compliant"] else 1
                predictions.append(predicted_label)
            except Exception as ex:
                print(f"Error moderating text: {text}\nException: {ex}")
                predictions.append(1)  # Default to Hate if moderation fails

        metrics = {
            "accuracy": accuracy_score(true_labels, predictions) * 100,
        }

        # Save metrics to a JSON file
        with open('output/editor_evaluation_metrics.json', 'w') as metrics_file:
            json.dump(metrics, metrics_file, indent=4)

        # Save evaluation results (print content) to a JSON file
        with open('output/editor_evaluation_results.json', 'w') as results_file:
            json.dump(all_edits, results_file, indent=4)

        return metrics



