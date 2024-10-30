import json
from openai import OpenAI

def get_response(client: OpenAI, model: str, question: str, chat_history: list, context: list):
    """
    # Context: Something helpfule resource
    #    That is, it could include multiple 'feature's
    #    Use 'key-val' data (dict, json) type.
    """
    cnt_reference = 3
    related_qa = []
    for id, item in enumerate(context):
        if id < cnt_reference :
            related_qa.append(f"""{id}. question: {item["question"]}, answer: {item["answer"]}""")
    related_qa = "\n".join(related_qa)

    SYSTEM_PROMPT = """You are an AI assistant for a smart store FAQ chatbot. Your role is to provide appropriate answers to questions related to the smart store.
When given 'Related_QA', use them as a reference to answer the 'Question'.
Additionaly show related questions ('Related: ') that is related to given 'Question'.

Instructions:
1. Analyze the 'Question' carefully.
2. Review the 'Related Questions and Answers' for relevant information.
3. If the question is about the smart store, provide a concise and accurate answer based on the related information.
4. If the question is unrelated to the smart store, respond with "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.".
5. Always maintain a professional and helpful tone.
6. Add 'Related: ', based on given 'Related_QA' and chat_history (maximum 5 question displayed with new line)
"""
    USER_PROMPT = f"""
Related_QA :
{related_qa}
Question :
{question}
Answer :
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