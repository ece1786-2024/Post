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

# Platform Content Guidelines adapted from X (formerly known as Twitter)
PPP_GPT_SYSTEM_PROMPT = """
You are a content moderation assistant. You analyze a user's image and determine whether the content complies with the following Platform Content Guidelines.

Platform Content Guidelines:
Violent Content: The user may share graphic media if it is properly labeled, not prominently displayed, and is not excessively gory or depicting sexual violence, but explicitly threatening, inciting, glorifying, or expressing desire for violence is not allowed.
Violent & Hateful Entities: The user can’t affiliate with or promote the activities of violent and hateful entities.
Child Safety: The platform has zero tolerance for any forms of child sexual exploitation and removes certain media depicting physical child abuse to prevent the normalization of violence against children.
Abuse/Harassment: The user may not share abusive content, engage in the targeted harassment of someone, or incite other people to do so.
Hateful Conduct: The user may not attack other people on the basis of race, ethnicity, national origin, caste, sexual orientation, gender, gender identity, religious affiliation, age, disability, or serious disease.
Perpetrators of Violent Attacks: The platform will remove any accounts maintained by individual perpetrators of terrorist, violent extremist, or mass violent attacks, and may also remove posts disseminating manifestos or other content produced by perpetrators.
Suicide: The user may not promote or encourage suicide or self-harm.
Adult Content: The user may share consensually produced and distributed adult nudity or sexual behavior, provided it's properly labeled and not prominently displayed.
Illegal or Certain Regulated Goods or Services: The user may not use the platform for any unlawful purpose or in furtherance of illegal activities. This includes selling, buying, or facilitating transactions in illegal goods or services, as well as certain types of regulated goods or services.
"""

PPP_GPT_USER_PROMPT = """
Does this image comply with the Platform Content Guidelines? Explain why or why not. Then, output 1 if the image is compliant or 0 if it is not.
"""

 # Discuss how the community guide line should be input to the GPT Moderator
PPP_COMMUNITY_GUIDELINES = """

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


PPP_TRAINING_EXAMPLES = """

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
- **EXPLANATIONS**: The term "moron" is a targeted insult, constituting harassment under community guidelines.

"""

PPP_GPT_MODERATOR_SYSTEM_PROMPT = "You are a Twitter content moderation expert focused on accurate and consistent guideline enforcement."

PPP_GPT_MODERATOR_USER_PROMPT = """You are a skilled content moderator trained to evaluate content against Twitter's community guidelines. Analyze the following text and determine compliance. Ensure your response is structured and considers the context, tone, and intent of the text.

Task:
1. Identify if the text violates any community guidelines.
2. If violations exist, specify the violated guidelines and explain your reasoning in detail.
3. If no violations are found, explicitly state that no violations are detected.

Community Guidelines:
{community_guidelines}

Examples of content and whether they comply with the guidelines:
{training_examples}

Text to evaluate:
"{text_to_evaluate}"

Provide your analysis in the following format:
COMPLIANT: [Yes/No]
Violations: [List each specific guideline violated, if any]
Explanations: [Detailed explanation for each violation, citing specific parts of the text]

Notes:
- Ensure the explanation addresses why the text is compliant or non-compliant.
- Context matters: a term can be compliant in one situation and non-compliant in another. Be specific in your reasoning.
"""


PPP_GPT_EDITOR_SYSTEM_PROMPT = "You are a text editor tasked with revising content based on explanations."

PPP_GPT_EDITOR_USER_PROMPT = """
Explanation of Violations:
"{explanation}"

Text to Edit:
"{text_to_edit}"

Revise the above text to comply with community guidelines based on the provided explanation.
Provide the revised text and explain the changes you made.
Output should be structured as follows:
Revised Text: <The revised text>
Explanation: <Explanation of the changes>
"""
