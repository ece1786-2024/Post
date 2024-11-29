import copy

from openai import OpenAI

from conversation_templates import PPP_GPT_MODERATOR_CONVERSATION_TEMPLATE
from prompts import PPP_COMMUNITY_GUIDELINES
from prompts import PPP_GPT_MODERATOR_USER_PROMPT
from prompts import PPP_TRAINING_EXAMPLES


class GPTModerator:
    """
    Moderate text against community guidelines using OpenAI's GPT API.

    """

    def __init__(self, api_key, PPP_NO_OP=False):
        self.OPENAI_API_KEY = api_key
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.PPP_NO_OP = PPP_NO_OP

    def process_batch(self, batch):
        results = {}
        for (id, _image, text), label in batch:
            if not self.PPP_NO_OP:
                result = self.moderate_text(text)

                results[id] = {
                    "Original Text": text,
                    "Compliance": result["compliant"],
                    "Violations": result["violations"],
                    "Moderator Reasoning": result["explanations"],
                }

        return results

    def moderate_text(self, text_to_evaluate):
        """
        Evaluate the given text against community guidelines.
        Args:
            text_to_evaluate (str): The text to evaluate.
        Returns:
            dict: A dictionary containing:
                - 'compliant' (bool): Whether the text complies with the guidelines.
                - 'violations' (list): List of specific guideline violations.
                - 'explanations' (str): Detailed explanations for each violation.
                - 'response': The raw response text from GPT.
        """

        messages_template = copy.deepcopy(PPP_GPT_MODERATOR_CONVERSATION_TEMPLATE)

        user_prompt = PPP_GPT_MODERATOR_USER_PROMPT.format(
            community_guidelines=PPP_COMMUNITY_GUIDELINES,
            training_examples=PPP_TRAINING_EXAMPLES,
            text_to_evaluate=text_to_evaluate
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

        # Parse the response to extract compliance, violations, and explanations
        compliant = None
        violations = []
        explanations = ""

        for line in response_text.splitlines():
            if line.lower().startswith("compliant:"):
                compliant = line.split(":")[1].strip().lower() == "yes"
            elif line.lower().startswith("violations:"):
                violations = [v.strip() for v in line.replace("Violations:", "").split(",") if v.strip()]
            elif line.lower().startswith("explanations:"):
                explanations = line.replace("Explanations:", "").strip()

        return {
            "compliant": compliant,
            "violations": violations,
            "explanations": explanations,
            "response": response_text,
        }
