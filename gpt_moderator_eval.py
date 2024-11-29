import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from gpt_moderator import GPTModerator


class GPTModeratorEval:
    def __init__(self, OPENAI_API_KEY):
        self.OPENAI_API_KEY = OPENAI_API_KEY

    def get_test_data(self, dataset_path):
        """
        Extract test data from the dataset, mapping labels to hate (1) / non-hate (0).

        Args:
            dataset_path (str): Path to the input JSON dataset.

        Returns:
            list: A list of dictionaries, each containing:
                - "tweet_id" (str): The ID of the tweet.
                - "tweet_text" (str): The text of the tweet.
                - "mapped_label" (int): 0 for non-hate, 1 for hate.
        """
        with open(dataset_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        test_data = []

        for tweet_id, tweet_info in data.items():
            tweet_text = tweet_info.get("tweet_text", "").strip()
            if tweet_info.get("text_only_label"):
                labels = tweet_info.get("text_only_label")
            else:
                labels = tweet_info.get("labels", [])

            squashed_labels = [
                0 if label == 0
                else 1
                for label in labels
            ]
            # Map labels to hate/non-hate
            mapped_label = 1 if np.mean(np.array(squashed_labels)) > 0.5 else 0

            test_data.append({
                "tweet_id": tweet_id,
                "tweet_text": tweet_text,
                "mapped_label": mapped_label
            })

        return test_data

    def evaluate_moderator(self, test_dataset):
        """
        Evaluate GPTModerator against a test dataset.

        Args:
            test_dataset (list): List of test data, each with 'tweet_id', 'tweet_text' and 'mapped_label'.

        Returns:
            dict: Evaluation metrics (accuracy, precision, recall, F1-score).
        """
        moderator = GPTModerator(self.OPENAI_API_KEY)

        true_labels = []
        predicted_labels = []
        results = []  # To store the print content

        for item in test_dataset:
            tweet_text = item['tweet_text']
            true_label = item['mapped_label']
            try:
                result = moderator.moderate_text(tweet_text)

                # Collecting the print content
                evaluation_result = {
                    "text": tweet_text,
                    "compliant": result["compliant"],
                    "violations": result["violations"],
                    "explanations": result["explanations"],
                    "true_label": True if true_label == 0 else False,
                }
                results.append(evaluation_result)

                compliance = result["compliant"]
                if compliance is None:
                    raise ValueError("Invalid API response: 'compliant' key missing.")
                predicted_label = 0 if compliance else 1

                # Print the details if show is True
                show = True
                if show:
                    print("Text:", tweet_text)
                    print("Compliant:", result["compliant"])
                    print("Violations:", result["violations"])
                    print("Explanations:", result["explanations"])
                    print("True label (Compliant):", True if true_label == 0 else False)
                    print("-------------------------------------------------")

            except Exception as e:
                print(f"Error processing tweet_id {item['tweet_id']}: {e}")
                continue

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hate", "Non-Hate"])

        # Adjust the plot
        plt.figure(figsize=(8, 6))  # Increase the figure size
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix:\nEvaluating Community Guideline Compliance on Small Dataset")

        # Ensure the layout is adjusted to avoid cropping
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig('output/confusion_matrix.png')
        plt.close()

        # Compute evaluation metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels) * 100,
            "precision": precision_score(true_labels, predicted_labels) * 100,
            "recall": recall_score(true_labels, predicted_labels) * 100,
            "f1_score": f1_score(true_labels, predicted_labels) * 100,
        }

        # Save metrics to a JSON file
        with open('output/evaluation_metrics.json', 'w') as metrics_file:
            json.dump(metrics, metrics_file, indent=4)

        # Save evaluation results (print content) to a JSON file
        with open('output/evaluation_results.json', 'w') as results_file:
            json.dump(results, results_file, indent=4)

        return metrics
