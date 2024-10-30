# qa-chat-bot
Custom Chat-bot


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
local server
```
# BASH
docker run -d --name milvus \
-p 19530:19530 \
-p 19121:19121 \
milvusdb/milvus:latest
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