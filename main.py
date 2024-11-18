import base64
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from enum import Enum

from MMHS150K_dataset import MMHS150KDataset
from openai import OpenAI
from process_dataset import squash_labels
from transformers import LlavaNextForConditionalGeneration
from transformers import LlavaNextProcessor
import copy
from conversation_templates import PPP_LLAVA_CONVERSATION_TEMPLATE, PPP_GPT_SYSTEM_CONVERSATION_TEMPLATE


class Datasets(Enum):
    FULL = "MMHS150K"
    SMALL = "MMHS150KCurated"
    WEB = "Webscraper"

# Constants
PPP_BATCH_SIZE = 8
# True: use LLAVA, False: use GPT-4o
USE_LLAVA = False
USE_DATASET = Datasets.SMALL
PPP_NO_OP = True


def load_dataset(use_dataset:Datasets):
    dataset_path = os.path.join(os.getcwd(), "MMHS150K")
    if use_dataset == Datasets.FULL:
        annotations_file = os.path.join(dataset_path, "MMHS150K_GT.json")
        image_dir = os.path.join(dataset_path, "img_resized")
    elif use_dataset == Datasets.SMALL:
        annotations_file = os.path.join(dataset_path, "MMHS150KCuratedSmall_GT.json")
        image_dir = os.path.join(dataset_path, "curated_images")
    elif use_dataset == Datasets.WEB:
        dataset_path = "Webscrape_result"
        annotations_file = os.path.join(dataset_path, "Webscrape_GT.json")
        image_dir = os.path.join(dataset_path, "images")
    else:
        raise ValueError(f"Unknown dataset {use_dataset}")

    dataset = MMHS150KDataset(
        annotations_file=annotations_file,
        image_dir=image_dir,
        target_transform=squash_labels
    )
    return dataset, image_dir

def encode_image(image_path):
    # OpenAI requires images to be base64 encoded
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def print_multimodal_content(id, image, text, label):
    image_display = image.squeeze().permute((1, 2, 0))
    plt.imshow(image_display)
    plt.title(id)
    plt.show()
    print(f"tweet_id: {id} || tweet_text: {text} || label: {label}")

def process_batch_llava(batch, processor, model):
    for (id, image, text), label in batch:
        print_multimodal_content(id, image, text, label)

        if not PPP_NO_OP:
            text_prompt = processor.apply_chat_template(PPP_LLAVA_CONVERSATION_TEMPLATE)
            inputs = processor(image, text_prompt, return_tensors="pt").to("cuda:0")

            output = model.generate(**inputs, max_new_tokens=1000)
            full_generated_text = processor.decode(output[0], skip_special_tokens=True)
            reconstructed_prompt = [frag for frag in full_generated_text.splitlines() if frag != '']
            print(reconstructed_prompt)

def process_batch_gpt(batch, client, image_dir, temperature=1.1):
    for (id, _image, text), label in batch:
        print_multimodal_content(id, _image, text, label)

        if not PPP_NO_OP:
            image_path = os.path.join(image_dir, f"{id}.jpg")
            base64_image = encode_image(image_path)

            messages = copy.deepcopy(PPP_GPT_SYSTEM_CONVERSATION_TEMPLATE)
            messages[1]['content'][1]['image_url']['url'] = f"data:image/jpeg;base64,{base64_image}"

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "text"}
            )

            print(response.choices[0])

def main():
    load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    seed = 1786
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Specify which dataset to use

    print(f"Using dataset {USE_DATASET}")
    dataset, image_dir = load_dataset(USE_DATASET)
    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=PPP_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: batch
    )

    if USE_LLAVA:
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llama3-llava-next-8b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        batch = next(iter(train_data_loader))
        process_batch_llava(batch, processor, model)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
        batch = next(iter(train_data_loader))
        process_batch_gpt(batch, client, image_dir)


if __name__ == "__main__":
    main()

