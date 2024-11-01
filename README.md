# qa-chat-bot
smart store Chat-bot (#RAG)
## Features
- RAG
  - retrieval with Milvus (Vector Database)
  - test dataset: Naver Smart Store FAQ
- LLM
  - response generation with OpenAI API
  - test model: gpt3.5-turbo | llama3.1

## Server archi
![architecture](./img/architecture.png)
## Pipeline
![pipeline](./img/pipeline.png)

# Prerequisite

## Dependency
* python=3.8 (< 3.10)

## Server
1.environment variable
make file first: '.env'
```
OPENAI_API_KEY="{MY_KEY}"
```
2.packages
```
# BASH
pip install -r requirements.txt
```

## database
local server # linux or mac
```
# BASH
wget https://github.com/milvus-io/milvus/releases/download/v2.2.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
docker-compose ps
```

# Getting started

## Run
```
streamlit run app.py
```
  
  
---
## [optional] with llama3
1.ollama server    
1.1. Install OLLAMA    
1.2. Run
```
# Bash
ollama run llama3.1:8b
```
2.set model
```
# 'app.py'
model_name = 'llama3.1'
```