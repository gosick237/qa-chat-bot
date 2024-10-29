import streamlit as st
from milvus import MilvusPipeline
from llm import get_response, OpenAI
import json, os

model_name = 'llama3.1' #"gpt-3.5-turbo-0125"

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

#@st.cache_resource
def get_pipeline(_client, model_name):
    pipeline = MilvusPipeline(
        client=_client,
        model_name=model_name,
        embedding_dim=model_config["embedding_dim"]
    )
    # 2) data insert
    pipeline.insert_data("./data/processed_data_2717.jsonl")
    # 3) create idx
    pipeline.create_index("embedding")
    # 4) load collection
    pipeline.load_collection()
    return pipeline

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
    pipeline = get_pipeline(client, model_name)

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
            ranked_texts = pipeline.retrieve_similar_questions(prompt, 5)
            # test
            for i, item in enumerate(ranked_texts):
                print(f"Result {i + 1}:")
                print(f"  Question: {item['question']}")
                print(f"  Category: {item['category']}")
                print(f"  Answer: {item['answer']}")
                print(f"  Related: {item['related']}")
                print(f"  Distance: {item['distance']:.4f}")
                print()

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