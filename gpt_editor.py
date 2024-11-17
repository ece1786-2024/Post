import os
from openai import OpenAI

class GPTEditor:
    """
    Edit text based on the explainations from GPT Moderator using OpenAI's GPT API.
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

        # TODO: Dicuss the input for the GPT Editor (text, explanantion, guideline, violation...)
        prompt = (
            f"Explanation of Violations:\n{explanation}\n\n"
            f"Text to Edit:\n{text_to_edit}\n\n"
            "Revise the above text to comply with community guidelines based on the provided explanation. "
            "Provide the revised text and explain the changes you made. "
            "Output should be structured as follows:\n"
            "Revised Text: <The revised text>\n"
            "Explanation: <Explanation of the changes>\n"
        )

        # Generate a response from GPT Editor
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a text editor tasked with revising content based on explanations."},
                {"role": "user", "content": prompt},
            ],
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
                "revised_text": revised_text,
                "explanation": explanation_of_changes,
            }