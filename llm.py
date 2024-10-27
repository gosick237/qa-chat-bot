from openai import OpenAI

def get_response(client: OpenAI, model: str, question: str, chat_history: list, context: str):
    SYSTEM_PROMPT = """
    Answer to the questions from the contextual passage snippets provided.
    if context is given, refer the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    """
    USER_PROMPT = f"<context>\n{context}\n</context>\n<question>\n{question}\n</question>"

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": USER_PROMPT}
    messages = [system_msg] + chat_history
    messages.append(user_msg)

    stream = client.chat.completions.create(
        model=model,
        messages= messages,
        stream=True
    )

    return stream