from openai import OpenAI

def get_response(client: OpenAI, model: str, question: str, chat_history: list, context: dict):
    """
    # Context: Something helpfule resource
    #    That is, it could include multiple 'feature's
    #    Use 'key-val' data (dict, json) type.
    """
    SYSTEM_PROMPT = """
    Answer to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    - Question : {question}
    Here are some related questions and answers that might help:
    - Related Question: {context['question']}
    - Related Answer: {context['answer']}
    Based on the above information, please provide a comprehensive answer to the user's question.
    """
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