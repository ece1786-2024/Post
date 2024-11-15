import os
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import pad

from torchvision.io import read_image

class MMHS150KDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None, **kwargs):
        super(MMHS150KDataset, self).__init__(**kwargs)

        self.image_labels = pd.read_json(
            annotations_file,
            orient="index",
            convert_dates=False,
            convert_axes=False
        )
        self.image_labels = self.image_labels.rename_axis("tweet_id")

        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        file_extension = ".jpg"

        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx].name+file_extension)
        image = read_image(image_path)
        id = self.image_labels.iloc[idx].name
        text = self.image_labels.iloc[idx].loc["tweet_text"]
        label = self.image_labels.iloc[idx].loc["labels_str"]

        if self.transform:
            id, image, text = self.transform(id, image, text)
        if self.target_transform:
            label = self.target_transform(label)

        return (id, image, text), label

    @staticmethod
    def collate_fn(batch):
        # Each image has shape (colour_channel, width, height)
        max_width = 0
        max_height = 0
        for (_id, image, _text), _label in batch:
            print(image)
            if image.shape[1] > max_width:
                max_width = image.shape[1]
            if image.shape[2] > max_height:
                max_height = image.shape[2]

        padded_batch = []
        for (_id, image, _text), _label in batch:
            # Padding for left, top, right, and bottom borders respectively
            padding = [
                int(np.ceil((max_width - image.shape[1]) / 2)),
                int(np.ceil((max_height - image.shape[2]) / 2)),
                int(np.floor((max_width - image.shape[1]) / 2)),
                int(np.floor((max_height - image.shape[2]) / 2))
            ]

            # TODO: For some reason, torch padding when specifying all 4 borders will randomly pick the permutation of
            #       padding borders
            padded_batch.append(
                (
                    (
                        _id,
                        pad(image, pad=padding, value=0, mode="constant"),
                        _text
                    ),
                    _label
                )
            )

            print(f"padding: {padding} image_size:{image.shape} max_width:{max_width} max_height:{max_height}")

        return padded_batch