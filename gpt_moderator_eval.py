from gpt_moderator import GPTModerator
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_test_data(dataset_path):
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
        labels = tweet_info.get("labels", [])

        # Map labels to hate/non-hate
        mapped_label = 0 if labels.count(0) > 1 else 1

        test_data.append({
            "tweet_id": tweet_id,
            "tweet_text": tweet_text,
            "mapped_label": mapped_label
        })

    return test_data


def evaluate_moderator(api_key, test_dataset):
    """
    Evaluate GPTModerator against a test dataset.

    Args:
        api_key (str): OpenAI API key.
        test_dataset (list): List of test data, each with 'tweet_id', 'tweet_text' and 'mapped_label'.

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, F1-score).
    """
    moderator = GPTModerator(api_key)

    true_labels = []
    predicted_labels = []

    for item in test_dataset:
        tweet_text = item['tweet_text']
        true_label = item['mapped_label']
        try:
            result = moderator.moderate_text(tweet_text)

            show = True

            if  show:
                print("Text:", tweet_text)
                print("Compliant:", result["compliant"])
                print("Violations:", result["violations"])
                print("Explanations:", result["explanations"])
                print("True label (Compliant):", True if true_label == 0 else False)
                print("-------------------------------------------------")

            compliance = result["compliant"]
            if compliance is None:
                raise ValueError("Invalid API response: 'compliant' key missing.")
            predicted_label = 0 if compliance else 1
        except Exception as e:
            print(f"Error processing tweet_id {item['tweet_id']}: {e}")
            continue

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Compute evaluation metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels) * 100,
        "precision": precision_score(true_labels, predicted_labels) * 100,
        "recall": recall_score(true_labels, predicted_labels) * 100,
        "f1_score": f1_score(true_labels, predicted_labels) * 100,
    }
    return metrics

if __name__ == "__main__":

    dataset_path = "MMHS150K/MMHS150KCuratedSmall_GT.json" 
    test_dataset = get_test_data(dataset_path)
    metrics = evaluate_moderator(api_key, test_dataset)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}%")