import os
from openai import OpenAI

class GPTModerator:
    """
    Moderate text against community guidelines using OpenAI's GPT API.
    """

    def __init__(self, api_key):
        """
        Initialize the GPTModerator with an API key.
        Args:
            api_key (str): OpenAI API key.
        """
        if not api_key:
            raise ValueError("API key must be provided.")
        # Set the API key
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()

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

        # TODO: Discuss how the community guide line should be input to the GPT Moderator
        community_guidelines = """
        1. No hate speech or offensive language.
        2. Respect others' privacy.
        3. Avoid spreading misinformation.
        4. Keep conversations constructive and polite.
        """

        # TODO: Prompt Engineering
        prompt = (
            f"Community Guidelines:\n{community_guidelines}\n\n"
            f"Text to evaluate:\n{text_to_evaluate}\n\n"
            "Does the text comply with the guidelines? If not, list the specific "
            "guidelines violated along with explanations for each violation. "
            "Provide a structured response as follows:\n"
            "Compliant: <Yes/No>\n"
            "Violations: <List of Violations>\n"
            "Explanations: <Detailed Explanations>\n"
        )

        # Generate a response from GPT Moderator
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a content moderation assistant."},
                {"role": "user", "content": prompt},
            ],
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