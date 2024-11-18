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

        # Discuss how the community guide line should be input to the GPT Moderator
        community_guidelines = """

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

        training_examples = """
        1. Text: "@user You're a dumb retard."
        - **COMPLIANT**: No
        - **VIOLATIONS**: Hateful Conduct, Abuse/Harassment
        - **EXPLANATIONS**: The term "retard" is used as an insult, targeting a person's intellectual disability, violating both hateful conduct and abuse guidelines.

        2. Text: "SJWs are ruining society."
        - **COMPLIANT**: Yes
        - **VIOLATIONS**: None
        - **EXPLANATIONS**: While critical of a group (SJWs), the statement does not attack or demean based on protected characteristics like race or gender.

        3. Text: "I can’t believe you’re this stupid! What a moron."
        - **COMPLIANT**: No
        - **VIOLATIONS**: Abuse/Harassment
        - **EXPLANATIONS**: This text is a targeted insult intended to demean an individual, violating abuse guidelines.

        4. Text: "Nigger is a racial slur."
        - **COMPLIANT**: No
        - **VIOLATIONS**: Hateful Conduct
        - **EXPLANATIONS**: The term "nigger" is inherently offensive and violates guidelines on hateful conduct, regardless of the context in which it is used.

        5. Text: "You might be a redneck if..."
        - **COMPLIANT**: No
        - **VIOLATIONS**: Hateful Conduct
        - **EXPLANATIONS**: The term "redneck" perpetuates stereotypes and is used here in a derogatory manner, violating hateful conduct guidelines.

        6. Text: "Enjoy your weekend, idiot!"
        - **COMPLIANT**: Yes
        - **VIOLATIONS**: None
        - **EXPLANATIONS**: While mildly rude, the term "idiot" here does not target protected characteristics or involve harassment.

        7. Text: "This retard doesn't know how to play."
        - **COMPLIANT**: No
        - **VIOLATIONS**: Hateful Conduct
        - **EXPLANATIONS**: The term "retard" is used as a derogatory insult, violating guidelines against attacking individuals based on disability.

        8. Text: "@realDonaldTrump You’re a moron."
        - **COMPLIANT**: No
        - **VIOLATIONS**: Abuse/Harassment
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        - **EXPLANATIONS**: The term "moron" is a targeted insult, constituting harassment under community guidelines.
            
        """
        
        # Prompt Engineering
        prompt = f"""You are a highly skilled content moderator specifically trained to evaluate content against community guidelines.
        
        ### Task: 
=======
=======
>>>>>>> Stashed changes
        - **EXPLANATIONS**: The term "moron" is a targeted insult, constituting harassment under community guidelines.        
        
        """
        
<<<<<<< Updated upstream
        # TODO: Prompt Engineering
        prompt = f"""You are a skilled content moderator trained to evaluate content against Twitter's community guidelines. Analyze the following text and determine compliance. Ensure your response is structured and considers the context, tone, and intent of the text.
        
        Task: 
>>>>>>> Stashed changes
=======
        # Prompt Engineering
        prompt = f"""You are a skilled content moderator trained to evaluate content against Twitter's community guidelines. Analyze the following text and determine compliance. Ensure your response is structured and considers the context, tone, and intent of the text.
        
        Task: 
>>>>>>> Stashed changes
=======
        - **EXPLANATIONS**: The term "moron" is a targeted insult, constituting harassment under community guidelines.        
        
        """
        
        # Prompt Engineering
        prompt = f"""You are a skilled content moderator trained to evaluate content against Twitter's community guidelines. Analyze the following text and determine compliance. Ensure your response is structured and considers the context, tone, and intent of the text.
        
        Task: 
>>>>>>> Stashed changes
        1. Identify if the text violates any community guidelines.
        2. If violations exist, specify the violated guidelines and explain your reasoning in detail.
        3. If no violations are found, explicitly state that no violations are detected.
        
        Community Guidlines:
        {community_guidelines}
        
        Examples of content and their evaluations to guide your analysis:
        {training_examples}

        Text to evaluate: 
        "{text_to_evaluate}"
        
        Provide your analysis in the following format:
        COMPLIANT: [Yes/No]
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        VIOLATIONSL [List each specific guideline violated, if any]
=======
        Violations: [List each specific guideline violated, if any]
>>>>>>> Stashed changes
=======
        Violations: [List each specific guideline violated, if any]
>>>>>>> Stashed changes
=======
        Violations: [List each specific guideline violated, if any]
>>>>>>> Stashed changes
        Explanations: [Detailed explanation for each violation, citing specific parts of the text]       
        
        Notes:
        - Ensure the explanation addresses why the text is compliant or non-compliant.
        - Context matters: a term can be compliant in one situation and non-compliant in another. Be specific in your reasoning.
        - Include severity levels (Low/Medium/High) for each violation identified.
            
        """

        # Generate a response from GPT Moderator
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Twitter content moderation expert focused on accurate and consistent guideline enforcement."},
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
    