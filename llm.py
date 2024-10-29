import json
from openai import OpenAI

def get_response(client: OpenAI, model: str, question: str, chat_history: list, context: list):
    """
    # Context: Something helpfule resource
    #    That is, it could include multiple 'feature's
    #    Use 'key-val' data (dict, json) type.
    """
    SYSTEM_PROMPT = """
    Answer to the questions from the contextual passage snippets provided.
    """
    related_qa = []
    for id, item in enumerate(context):
        related_qa.append(f"""{id}. question: {item["question"]}, answer: {item["answer"]}""")
    related_qa = "\n".join(related_qa)
    USER_PROMPT = f"""
    - Question : {question}
    Here are some related questions and answers that might help:
    {related_qa}
    * Based on the above information, please provide a comprehensive answer to the user's question.
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