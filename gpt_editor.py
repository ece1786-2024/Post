import copy

from openai import OpenAI

from conversation_templates import PPP_GPT_EDITOR_CONVERSATION_TEMPLATE
from prompts import PPP_GPT_EDITOR_USER_PROMPT


class GPTEditor:
    """
    Edit text based on the explainations from GPT Moderator using OpenAI's GPT API.
    """

    def __init__(self, api_key):
        self.OPENAI_API_KEY = api_key
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)

    def edit_text(self, text_to_edit, explanation):
        """
        Edit the given text based on the explanation from GPT Moderator.
        Args:
            text_to_edit (str): The text to edit.
            explanation (str): Explanation of why the text did not comply with the community guidelines.
        Returns:
            dict: A dictionary containing:
                - 'original_text' (str): The original input text.
                - 'revised_text' (str): The revised text.
                - 'explanation' (str): Explanation of how the text was revised.
        """

        messages_template = copy.deepcopy(PPP_GPT_EDITOR_CONVERSATION_TEMPLATE)

        user_prompt = PPP_GPT_EDITOR_USER_PROMPT.format(
            text_to_edit = text_to_edit,
            explanation = explanation,
        )

        messages_template[1]['content'][0]['text'] = user_prompt

        # Adjust those attribute to fine-tune the model's response
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages_template,
            # temperature=temperature,
            # max_tokens=2048,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # response_format={"type": "text"}
        )

        response_text = response.choices[0].message.content.strip()
        # Parse the response to extract the original text, revised text and explanation
        revised_text = ""
        explanation_of_changes = ""

        for line in response_text.splitlines():
                if line.lower().startswith("revised text:"):
                    revised_text = line.replace("Revised Text:", "").strip()
                elif line.lower().startswith("explanation:"):
                    explanation_of_changes = line.replace("Explanation:", "").strip()

        return {
                "original_text": text_to_edit,
                "revised_text": revised_text.strip("\""),
                "explanation": explanation_of_changes,
            }
