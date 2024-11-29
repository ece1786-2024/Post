# main.py
import json
import os
import numpy as np
import torch
from dotenv import load_dotenv
from enum import Enum

from MMHS150K_dataset import MMHS150KDataset
from gpt_editor import GPTEditor
from process_dataset import squash_labels

from vqa_moderator import VQAModerator
from gpt_moderator import GPTModerator


class Datasets(Enum):
    FULL = "MMHS150K"
    SMALL = "MMHS150KCurated"
    WEB = "Webscraper"

class DatasetManager:
    # Constants
    PPP_BATCH_SIZE = 2
    USE_LLAVA = False
    USE_DATASET = Datasets.SMALL

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

def main(PPP_OUTPUT_FILE_NAME, PPP_EVAL=False, PPP_NO_OP=False, DEBUG_TERMINATE=False):
    load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    seed = 1786
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_manager = DatasetManager()

    # Specify which dataset to use
    print(f"Using dataset {dataset_manager.USE_DATASET}")
    dataset, image_dir, annotations_file = dataset_manager.load_dataset()

    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_manager.PPP_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: batch
    )

    vqa_moderator = VQAModerator(dataset_manager.USE_LLAVA, OPENAI_API_KEY, PPP_NO_OP)
    gpt_moderator = GPTModerator(OPENAI_API_KEY, PPP_NO_OP)
    gpt_editor = GPTEditor(OPENAI_API_KEY)

    #batch = next(iter(train_data_loader))
    results = {}
    for batch in train_data_loader:
        vqa_results = vqa_moderator.process_batch(batch, image_dir)
        mod_results = gpt_moderator.process_batch(batch)

        # Aggregate the models' results and pass the information to the Editor model
        for id in set(vqa_results.keys()).union(set(mod_results.keys())):
            results[id] = {}
            results[id].update(vqa_results.get(id, {}))
            results[id].update(mod_results.get(id, {}))

            if not PPP_NO_OP:
                if results[id].get("Compliance", True) == False:
                    edt_result = gpt_editor.edit_text(
                        results[id].get("Original Text"),
                        results[id].get("Moderator Reasoning")
                    )
                    results[id].update(edt_result)

        if DEBUG_TERMINATE:
            print("Debug terminate was enabled. Stopping after one batch.")
            break

    print(results)
    with open(PPP_OUTPUT_FILE_NAME, 'w') as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    # Early terminate to run on a single batch to save on API calls
    DEBUG_TERMINATE = True
    output_filename = "PPP_output.json"

    main(output_filename, PPP_EVAL=False, PPP_NO_OP=False, DEBUG_TERMINATE=DEBUG_TERMINATE)
