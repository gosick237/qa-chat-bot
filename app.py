import streamlit as st
st.set_page_config(layout="wide")

from llm import get_response, OpenAI
from encoder import embedding

# Parameters
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)
model = 'llama3.1' #"gpt-3.5-turbo-0125"
# For OpenAI Models
#openai.api_key = os.getenv("OPENAI_API_KEY")


# UI
st.title('Smart Store Chatbot')
st.markdown("""
<style>
.wrapper {
    height: auto;
    min-height: 100%;
    padding-bottom: 200px;
    background-color: blue;
}
.chat-container {
    height: 60dvh;
    height: 60vh;
    overflow-y: auto;
    border: 1px solid #ddd;
    border-radius: 5px;
}
.input_container {
    height: 200px;
    position: relative;
    transform: translateY(-100%);
}
</style>
""", unsafe_allow_html=True)

def display_msg(msg):
    with st.chat_message(msg["role"]):
        st.markdown(f"**{msg['role']}:** {msg['content']}")

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
        stream = get_response(
            client=client,
            model=model,
            question=prompt,
            chat_history=st.session_state.messages,
            context=""
        )
        response = st.write_stream(stream)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content" : response   
        })