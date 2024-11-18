# Post++

## Project Structure
- **MMHS150K**: Contains curated datasets and related files.
- **Webscrape_result**: Directory for storing web scraping results (images and annotations).
- **Scripts**:
  - **Core Functionality**:
    - `gpt_editor.py`: Uses OpenAI's GPT API to revise text based on guideline violations explained by the moderator.
    - `gpt_moderator.py`: Moderates text against community guidelines using OpenAI's GPT API.
  - **Data Handling**:
    - `process_dataset.py`: Cleans and preprocesses datasets, including label adjustments.
    - `webscrape.py`: Script for collecting and saving data via web scraping.
  - **Pipeline Management**:
    - `main.py`: Central script to manage dataset processing and moderation tasks.
  - **Evaluation**:
    - `compute_metrics.py`: Calculates performance metrics for text analysis or moderation tasks.
  - **Support Files**:
    - `prompts.py`: Stores reusable templates for GPT-based moderation and editing.
    - `conversation_templates.py`: Structures predefined conversations for moderation and editing flows.


## Requirements

This project requires Python 3.10 and specific library dependencies. Ensure a proper environment setup using the instructions below.

### Package Requirements

#### Windows Environment
```plaintext
pandas~=2.2.3
matplotlib~=3.9.2
numpy~=1.26.4
torch~=2.5.1+cu124
transformers~=4.46.2
torchvision~=0.20.1+cu124
openai~=0.28.0
python-dotenv~=1.0.0
```

#### macOS Environment
```plaintext
pandas~=2.2.3
matplotlib~=3.9.2
numpy~=1.26.4
torch~=2.5.1
transformers~=4.46.2
torchvision~=0.20.1
openai~=0.28.0
python-dotenv~=1.0.0
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ece1786-2024/Post.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Post
   ```
3. Create and activate a Conda environment:
   ```bash
   conda create -n post python=3.10
   conda activate post
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Project Pipeline
1. To process datasets and run text moderation or editing tasks, execute:
   ```bash
   python main.py
   ```

### Individual Script Execution
- Moderate a single text sample:
  ```bash
  python gpt_moderator.py
  ```
- Edit a text based on guideline violations:
  ```bash
  python gpt_editor.py
  ```
- Web scraping:
  ```bash
  python webscrape.py
  ```
- Compute evaluation metrics:
  ```bash
  python compute_metrics.py
  ```

---

## Features

### GPT Moderator
- Evaluates text against defined community guidelines, including:
  - Violent Content
  - Hateful Conduct
  - Abuse/Harassment
  - Child Safety
- Outputs detailed analysis, specifying:
  - Whether the text is compliant.
  - List of guideline violations.
  - Explanation for each violation.

#### Example Usage
Input:
```python
moderator = GPTModerator(api_key="your_api_key")
result = moderator.moderate_text("This retard doesn't know how to play.")
print(result)
```

Output:
```plaintext
{
  "compliant": False,
  "violations": ["Hateful Conduct"],
  "explanations": "The term 'retard' is used as a derogatory insult, violating guidelines against attacking individuals based on disability."
}
```

---

### GPT Editor
- Revises text based on moderator-provided explanations.
- Ensures compliance with guidelines while preserving the original message's intent.

#### Example Usage
Input:
```python
editor = GPTEditor(api_key="your_api_key")
edited_text = editor.edit_text(
    "This retard doesn't know how to play.",
    "The term 'retard' is used as a derogatory insult, violating guidelines against attacking individuals based on disability."
)
print(edited_text)
```

Output:
```plaintext
{
  "original_text": "This retard doesn't know how to play.",
  "revised_text": "This person doesn't know how to play.",
  "explanation": "Revised the derogatory term 'retard' to 'person' to remove offensive language while maintaining the original sentence's intent."
}
```



