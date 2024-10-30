import streamlit as st
from milvus import MilvusPipeline
from llm import get_response, OpenAI
import json, os

model_name = "gpt-3.5-turbo-0125" #'llama3.1'

def load_model_config(model="gpt-3.5-turbo-0125", file_path='./model/model_config.json'):
    with open(file_path, 'r') as file:
        model_configs = json.load(file)
    
    for config in model_configs:
        if config["model"] == model:
            config["api_key"] = os.getenv(config["api_key"])
            return config
    
    raise ValueError(f"Model '{model}' not found in the configuration.")

def display_msg(msg):
    with st.chat_message(msg["role"]):
        st.markdown(f"**{msg['role']}:** {msg['content']}")
        
def init_milvus(_client, ebedd_modelname):
    if 'milvus_pipe' not in st.session_state:
        with st.spinner('Initializing Milvus...'):
            # 0) set retrieval pipeline
            pipeline = MilvusPipeline(
                client=_client,
                ebedd_modelname=ebedd_modelname,
                embedding_dim=model_config["embedding_dim"]
            )
            st.session_state.milvus_pipe = pipeline
            # 1) set collection
            pipeline.create_collection("qa_collection_gpt35_embeddgins")
            # 2) data insert # Not nessasary if db is already set.
            pipeline.insert_data("./data/processed_data_2717.jsonl")
            # 3) create idx
            pipeline.create_index("embedding", "qa_collection_gpt35_embeddgins")
            # 4) load collection
            pipeline.load_collection()
    return st.session_state.milvus_pipe

if __name__ == "__main__":
    # UI
    st.set_page_config(layout="wide")
    st.title('Smart Store Chatbot')

    # Initialization
    model_config = load_model_config(model_name)
    client = OpenAI(
        base_url=model_config['base_url'],
        api_key=model_config['api_key']
    )
    pipeline = init_milvus(client, model_config['embedding_model'])

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        display_msg(message)
    
    # Request
    if prompt := st.chat_input("enter question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Retrival (Augmented)
            top_k=5
            ranked_texts = pipeline.retrieve_similar_questions(prompt, top_k, 0.8)

            # Generate Response
            stream = get_response(
                client=client,
                model=model_name,
                question=prompt,
                chat_history=st.session_state.messages,
                context=ranked_texts
            )
            response = st.write_stream(stream)
        
            st.session_state.messages.append({
                "role": "assistant",
                "content" : response
            })