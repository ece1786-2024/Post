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
        1. "Nigga this shit hard... Had to add this to the playlist.."
        - This is NOT hate. The word "nigga" is used colloquially and not in a hateful or offensive manner.
        2. "Yes they can say dyke what u gonna do about it huh"
        - This is NOT hate. The term "dyke" can be reclaimed and is not used here to demean or harm.
        3. "Redneck"
        - This IS hate. The term is used as a derogatory slur targeting a group based on cultural or regional identity.
        4. "@realDonaldTrump Enjoy your weekend you cunt."
        - This IS hate. The term "cunt" is a derogatory insult intended to demean a person.
        """
        
        # TODO: Prompt Engineering
        prompt = f"""You are a precise content moderator specifically trained to evaluate content against Twitter's community guidelines.
        
        Task: Analyze the following text for violations of Twitter's community guidelines. Provide a structured evaluation.
        
        Community Guidlines:
        {community_guidelines}
        
        Examples of content and whether they comply with the guidelines:
        {training_examples}

        Text to evaluate: 
        "{text_to_evaluate}"
        
        Provide your analysis in the following format:
        COMPLIANT: [Yes/No]
        Violations: [List each ...]
        Explanations: [Detailed explanation for each violation, citing specific parts of the text]       
            
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
        severity = []
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
    