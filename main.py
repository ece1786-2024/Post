# main.py

import os
import numpy as np
import torch
from dotenv import load_dotenv
from enum import Enum

from MMHS150K_dataset import MMHS150KDataset
from process_dataset import squash_labels

from vqa_moderator import VQAModerator
from gpt_moderator_eval import GPTModeratorEval

class Datasets(Enum):
    FULL = "MMHS150K"
    SMALL = "MMHS150KCurated"
    WEB = "Webscraper"

class DatasetManager:
    # Constants
    PPP_BATCH_SIZE = 8
    USE_LLAVA = False
    USE_DATASET = Datasets.SMALL
    PPP_NO_OP = True

    def __init__(self, use_dataset=USE_DATASET):
        self.use_dataset = use_dataset

    def load_dataset(self):
        dataset_path = os.path.join(os.getcwd(), "MMHS150K")
        if self.use_dataset == Datasets.FULL:
            annotations_file = os.path.join(dataset_path, "MMHS150K_GT.json")
            image_dir = os.path.join(dataset_path, "img_resized")
        elif self.use_dataset == Datasets.SMALL:
            annotations_file = os.path.join(dataset_path, "MMHS150KCuratedSmall_GT.json")
            image_dir = os.path.join(dataset_path, "curated_images")
        elif self.use_dataset == Datasets.WEB:
            dataset_path = "Webscrape_result"
            annotations_file = os.path.join(dataset_path, "Webscrape_GT.json")
            image_dir = os.path.join(dataset_path, "images")
        else:
            raise ValueError(f"Unknown dataset {self.use_dataset}")

        dataset = MMHS150KDataset(
            annotations_file=annotations_file,
            image_dir=image_dir,
            target_transform=squash_labels
        )
        return dataset, image_dir, annotations_file

def main():
    load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    seed = 1786
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_manager = DatasetManager()

    # Specify which dataset to use
    print(f"Using dataset {dataset_manager.USE_DATASET}")
    dataset, image_dir, annotations_file = dataset_manager.load_dataset()

    ## Evaluate the VQAModerator, comment out if not needed

    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_manager.PPP_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: batch
    )

    # Import VQAModerator and process the batch
    vqa_moderator = VQAModerator(dataset_manager.USE_LLAVA, OPENAI_API_KEY, dataset_manager.PPP_NO_OP)
    batch = next(iter(train_data_loader))
    vqa_moderator.process_batch(batch, image_dir)


    ## Evaluate the GPTModerator, comment out if not needed

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
    main()
