import base64
import copy
import os

import matplotlib.pyplot as plt
import torch
from openai import OpenAI
from transformers import LlavaNextForConditionalGeneration
from transformers import LlavaNextProcessor

from conversation_templates import PPP_GPT_SYSTEM_CONVERSATION_TEMPLATE
from conversation_templates import PPP_LLAVA_CONVERSATION_TEMPLATE


class VQAModerator:
    def __init__(self, USE_LLAVA, OPENAI_API_KEY, PPP_NO_OP=False):
        self.USE_LLAVA = USE_LLAVA
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.PPP_NO_OP = PPP_NO_OP

        if self.USE_LLAVA:
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llama3-llava-next-8b-hf",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.client = OpenAI(api_key=self.OPENAI_API_KEY)

    def encode_image(self, image_path):
        # OpenAI requires images to be base64 encoded
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def print_multimodal_content(self, id, image, text, label):
        image_display = image.squeeze().permute((1, 2, 0))
        plt.imshow(image_display)
        plt.title(id)
        plt.show()
        print(f"tweet_id: {id} || tweet_text: {text} || label: {label}")

    def process_batch(self, batch, image_dir):
        if self.USE_LLAVA:
            return self.process_batch_llava(batch)
        else:
            return self.process_batch_gpt(batch, image_dir)

    def process_batch_llava(self, batch):
        results = {}
        for (id, image, text), label in batch:
            self.print_multimodal_content(id, image, text, label)

            if not self.PPP_NO_OP:
                text_prompt = self.processor.apply_chat_template(PPP_LLAVA_CONVERSATION_TEMPLATE)
                inputs = self.processor(image, text_prompt, return_tensors="pt").to("cuda:0")

                output = self.model.generate(**inputs, max_new_tokens=1000)
                full_generated_text = self.processor.decode(output[0], skip_special_tokens=True)
                reconstructed_prompt = [frag for frag in full_generated_text.splitlines() if frag != '']

                #print(reconstructed_prompt)
                results[id] = {"VQA Response": reconstructed_prompt}
        return results

    def process_batch_gpt(self, batch, image_dir, temperature=1.1):
        results = {}

        for (id, _image, text), label in batch:
            self.print_multimodal_content(id, _image, text, label)

            if not self.PPP_NO_OP:
                image_path = os.path.join(image_dir, f"{id}.jpg")
                base64_image = self.encode_image(image_path)

                messages = copy.deepcopy(PPP_GPT_SYSTEM_CONVERSATION_TEMPLATE)
                messages[1]['content'][1]['image_url']['url'] = f"data:image/jpeg;base64,{base64_image}"

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={"type": "text"}
                )

                #print(response.choices[0])
                results[id] = {"VQA Response": response.choices[0].message.content.strip()}

        return results