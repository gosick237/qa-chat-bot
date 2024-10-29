# qa-chat-bot
Custom Chat-bot

# Getting started
### Dependency
* python=3.8 (< 3.10)
### Prerequisite
1.environment variable
```'.env'
OPENAI_API_KEY="{MY_KEY}"
```
2.packages
```
BASH
pip install -r requirements.txt
```
3.database
3.1 milvus server
```
# 'miluvs.py'
client = MilvusClient(uri="http://localhost:19530")
```
```
# BASH
docker run -d --name milvus \
-p 19530:19530 \
-p 19121:19121 \
milvusdb/milvus:latest
```
3.2 milvus-lite (linux or mac only): replace with loca path
```
#'miluvs.py'
clinet = MilvusClient(uri="{MY_MILVUS_PATH}.db")
```
## Run
```
streamlit run app.py
```
---
### [optional] with llama3
1.ollama server
1.1 Install OLLAMA
1.2 Run
```
# Bash
ollama run llama3.1:8b
```
2.set model
```
# 'app.py'
model_name = 'llama3.1'
```