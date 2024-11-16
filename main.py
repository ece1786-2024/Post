import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from MMHS150K_dataset import MMHS150KDataset
from process_dataset import squash_labels
from transformers import LlavaNextForConditionalGeneration
from transformers import LlavaNextProcessor

PPP_BATCH_SIZE = 1

PROMPT_FORMAT = """
<|start_header_id|>system<|end_header_id|>\n{PPP_LLAVA_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n<image>\n{PPP_LLAVA_USER_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
"""
PPP_LLAVA_SYSTEM_PROMPT = """
You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
"""
PPP_LLAVA_USER_PROMPT = """
What is shown in this image?
"""

PPP_LLAVA_CONVERSATION_TEMPLATE = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": f"{PPP_LLAVA_SYSTEM_PROMPT}",
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": f"{PPP_LLAVA_USER_PROMPT}",
            }
        ]
    }
]

if __name__ == "__main__":
    seed = 1786
    torch.manual_seed(seed)

    # True if using the full MMHS150K dataset (external download)
    # False if using the small curated dataset
    FULL_DATASET = False

    dataset_path = os.path.join(os.getcwd(), "MMHS150K")
    if FULL_DATASET:
        annotations_file = os.path.join(dataset_path, "MMHS150K_GT.json")
        image_dir = os.path.join(dataset_path, "img_resized")
    else:
        annotations_file = os.path.join(dataset_path, "MMHS150KCuratedSmall_GT.json")
        image_dir = os.path.join(dataset_path, "curated_images")

    dataset = MMHS150KDataset(annotations_file=annotations_file, image_dir=image_dir, target_transform=squash_labels)

    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=PPP_BATCH_SIZE, shuffle=True, collate_fn=lambda batch:batch)

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16, device_map="auto")

    for (id, image, text), label in (next(iter(train_data_loader))):
        image = image.squeeze().permute((1,2,0))
        plt.imshow(image)
        plt.title(id)
        plt.show()
        print(f"tweet_id: {id} || tweet_text: {text} || label: {label}")

        text_prompt = processor.apply_chat_template(PPP_LLAVA_CONVERSATION_TEMPLATE)

        inputs = processor(image, text_prompt, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=100)

        full_generated_text = processor.decode(output[0], skip_special_tokens=True)
        #print(full_generated_text)
        fragmented_text = full_generated_text.splitlines()

        reconstructed_prompt = []
        for fragment in fragmented_text:
            if fragment != '':
                reconstructed_prompt.append(fragment)

        print(reconstructed_prompt[4:])

