import accelerate
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from MMHS150K_dataset import MMHS150KDataset


def squash_labels(labels):
    label_mappings = {
        "NotHate": 0,
        "Racist": 1,
        "Sexist": 2,
        "Homophobe": 3,
        "Religion": 4,
        "OtherHate": 5
    }
    mapped_labels = [label_mappings[label] for label in labels]

    # Condense all versions of "hate" into one label
    squashed_labels = [
        0 if label == 0
        else 1
        for label in mapped_labels
    ]

    # Majority vote wins
    return 1 if np.mean(np.array(squashed_labels)) > 0.5 else 0


if __name__ == "__main__":
    seed = 1786
    torch.manual_seed(seed)

    dataset_path = os.path.join(os.getcwd(), "MMHS150K")
    annotations_file = os.path.join(dataset_path, "MMHS150K_GT.json")
    image_dir = os.path.join(dataset_path, "img_resized")

    dataset = MMHS150KDataset(annotations_file=annotations_file, image_dir=image_dir, target_transform=squash_labels)

    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda batch:batch) #MMHS150KDataset.collate_fn(batch))

    # Show a tweet, its linked image, and the label
    for (id, image, text), label in (next(iter(train_data_loader))):
        image = image.squeeze().permute((1,2,0))
        plt.imshow(image)
        plt.title(id)
        plt.show()
        print(f"tweet_id: {id} || tweet_text: {text} || label: {label}")






