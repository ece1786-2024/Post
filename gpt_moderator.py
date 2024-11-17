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
        1. Hateful Conduct and Harassment:
            - No hate speech targeting individuals or groups based on:
             * Race, ethnicity, or national origin
             * Religious affiliation
             * Sexual orientation or gender identity
             * Age, disability, or serious disease
             * Caste, social class, or occupation
            - No dehumanizing language or comparisons
            - No harmful stereotypes or racist/sexist tropes
            - No deadnaming or misgendering
            - No inciting others to harass or discriminate
            - No hateful imagery or symbols
            - No denial or minimization of violent events
            - No promoting hate groups or their ideologies
            
        2. Violence and Physical Harm:
            - No violent threats or glorification of violence
            - No terrorism or violent extremism
            - No violent criminal organizations
            - No suicide or self-harm promotion
            - No graphic violence or gore
            - No violent sexual conduct
            - No animal cruelty
            - No incitement to violence
            - No wishing harm on others

        3. Child Safety:
            - No exploitation of minors
            - No content promoting child sexual exploitation
            - No sexualization of minors
            - No depictions of minors in violent or inappropriate contexts
            - No sharing of private media of minors
            
        4. Privacy and Personal Information:
            - No doxxing or sharing private information
            - No sharing of private media without consent
            - No hacked materials
            - No intimate media without consent
            - No threatening to expose private information
            - No sharing of government-issued IDs
            - No sharing of financial information
           
        5. Platform Manipulation and Spam:
            - No coordinated platform manipulation
            - No artificial amplification of content
            - No bulk/aggressive tweeting
            - No engagement farming
            - No creation of multiple accounts
            - No buying/selling of engagement
            - No spreading of malware/viruses
            - No phishing attempts
        
        6. Authenticity and Misinformation:
            - No impersonation of individuals or organizations
            - No synthetic or manipulated media intended to deceive
            - No false claims about civic processes
            - No coordinated disinformation campaigns
            - No misleading content about public health
            - No false affiliation claims
            - No deceptive identities
            - No manipulated media labeled as real
            - No spreading of conspiracy theories that target protected groups
            
        7. Sensitive Media:
            - No excessive gore or violence
            - No non-consensual nudity
            - No sexual content in live video or profile media
            - No gratuitously graphic content
            - No media depicting deceased individuals
            - No excessively violent or disturbing content    
           
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
        VIOLATIONSL [List each specific guideline violated, if any]
        SEVERITY: [Low/Medium/High for each violation]
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
            elif line.lower().startswith("SEVERITY:"):
                severity = [s.strip() for s in line.replace("SEVERITY:", "").split(",") if s.strip()]
            elif line.lower().startswith("explanations:"):
                explanations = line.replace("Explanations:", "").strip()
                
        return {
            "compliant": compliant,
            "violations": violations,
            "severity": severity,
            "explanations": explanations,
            "response": response_text,
        }
    