# qa-chat-bot

Custom Chat-bot

# Getting started

## Dependency
python=3.8 (< 3.10)

## Prerequisite
1. environment variable
 * make '.env' file
 * add my open api key
  > OPENAI_API_KEY="{MY_KEY}"
2. packages
 > pip install -r requirements.txt

## Run
 > streamlit run app.py

# with llama3
## Prerequiste
1. ollama server
 * Install
  > Install by ollama.com
 * Run
  > ollama run llama3.1:8b
2. set url (using OpenAI)
> base_url='http://localhost:11434/v1'