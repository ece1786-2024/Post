import base64
import json
import os
import pandas as pd
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

PPP_BATCH_SIZE = 8

PROMPT_FORMAT = """
<|start_header_id|>system<|end_header_id|>\n{PPP_LLAVA_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n<image>\n{PPP_LLAVA_USER_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
"""
PPP_LLAVA_SYSTEM_PROMPT = """
You are a moderation assistant. You analyze a user's image and determine whether the content complies with the Platform Content Guidelines below.   

Platform Content Guidelines:
Violent Content is only allowed if it is properly labeled and not excessively gory. Threatening, inciting, glorifying, or expressing desire for violence is not allowed.
"""
PPP_LLAVA_USER_PROMPT = """
Does this comply with the Platform Content Guidelines? Explain your reasoning. Output 1 if the image is compliant or 0 if it is not.
"""

# Platform Content Guidelines have been adapted from X (formerly known as Twitter) https://help.x.com/en/rules-and-policies/x-rules
PPP_GPT_SYSTEM_PROMPT = """
You are a content moderation assistant. You analyze a user's image and determine whether the content complies with the following Platform Content Guidelines.

Platform Content Guidelines:
Violent Content: The user may share graphic media if it is properly labeled, not prominently displayed and is not excessively gory or depicting sexual violence, but explicitly threatening, inciting, glorifying, or expressing desire for violence is not allowed. 
Violent & Hateful Entities: The user can’t affiliate with or promote the activities of violent and hateful entities. 
Child Safety: The platform has zero tolerance for any forms of child sexual exploitation and remove certain media depicting physical child abuse to prevent the normalization of violence against children. 
Abuse/Harassment: The user may not share abusive content, engage in the targeted harassment of someone, or incite other people to do so. 
Hateful conduct: The user may not attack other people on the basis of race, ethnicity, national origin, caste, sexual orientation, gender, gender identity, religious affiliation, age, disability, or serious disease.  
Perpetrators of Violent Attacks: The platform will remove any accounts maintained by individual perpetrators of terrorist, violent extremist, or mass violent attacks, and may also remove posts disseminating manifestos or other content produced by perpetrators.  
Suicide: The user may not promote or encourage suicide or self-harm. 
Adult Content: The user may share consensually produced and distributed adult nudity or sexual behavior, provided it's properly labeled and not prominently displayed.  
Illegal or Certain Regulated Goods or Services: The user may not use the platform for any unlawful purpose or in furtherance of illegal activities. This includes selling, buying, or facilitating transactions in illegal goods or services, as well as certain types of regulated goods or services. 
"""

PPP_GPT_USER_PROMPT = """
Does this image comply with the Platform Content Guidelines? Explain why or why not. Then, output 1 if the image is compliant or 0 if it is not.
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

class Datasets(Enum):
    FULL = "MMHS150K"
    SMALL = "MMHS150KCurated"
    WEB = "Webscraper"

if __name__ == "__main__":
    load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    seed = 1786
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Specify which dataset to use
    USE_DATASET = Datasets.SMALL
    print(f"Using dataset {USE_DATASET}")

    dataset_path = os.path.join(os.getcwd(), "MMHS150K")
    if USE_DATASET == Datasets.FULL:
        annotations_file = os.path.join(dataset_path, "MMHS150K_GT.json")
        image_dir = os.path.join(dataset_path, "img_resized")
    elif USE_DATASET == Datasets.SMALL:
        annotations_file = os.path.join(dataset_path, "MMHS150KCuratedSmall_GT.json")
        image_dir = os.path.join(dataset_path, "curated_images")
    elif USE_DATASET == Datasets.WEB:
        dataset_path = "Webscrape_result"
        annotations_file = os.path.join(dataset_path, "Webscrape_GT.json")
        image_dir = os.path.join(dataset_path, "images")
    else:
        raise ValueError(f"Unknown dataset {USE_DATASET}")

    dataset = MMHS150KDataset(annotations_file=annotations_file, image_dir=image_dir, target_transform=squash_labels)

    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=PPP_BATCH_SIZE, shuffle=True, collate_fn=lambda batch:batch)

    USE_LLAVA = False
    if USE_LLAVA:
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
            output = model.generate(**inputs, max_new_tokens=1000)

            full_generated_text = processor.decode(output[0], skip_special_tokens=True)
            #print(full_generated_text)
            fragmented_text = full_generated_text.splitlines()

            reconstructed_prompt = []
            for fragment in fragmented_text:
                if fragment != '':
                    reconstructed_prompt.append(fragment)

            print(reconstructed_prompt)#[4:])

    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

        def encode_image(image_path):
            # OpenAI requires images to be base64 encoded
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')


        for (id, _image, text), label in (next(iter(train_data_loader))):
            image = _image.squeeze().permute((1, 2, 0))
            plt.imshow(image)
            plt.title(id)
            plt.show()
            print(f"tweet_id: {id} || tweet_text: {text} || label: {label}")

            image_path = os.path.join(image_dir, id + ".jpg")
            base64_image = encode_image(image_path)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": PPP_GPT_SYSTEM_PROMPT
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": PPP_GPT_USER_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=1.1,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={
                    "type": "text"
                }
            )

            print(response.choices[0])
