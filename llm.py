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

    SYSTEM_PROMPT = """You are an AI assistant for a smart store FAQ chatbot. Your role is to provide appropriate answers to questions related to the smart store. If a question is unrelated to the smart store, respond with "Not relevant"."""
    """
    당신은 주어진 질문에 적절한 답변을 하는 스마트 스토어 FAQ 챗봇입니다. (역할을 벗어나는 질문은 "관련 없음"으로 답변)
    '관련 질문과 답변들'이 주어질 경우, '관련 질문과 답변들'을 근거로 사용하여 '질문'에 대한 답변을 해주세요.
    (단, 주어진 '관련 질문과 답변들'이 '질문'과 연관성이 없다면 무시.)
    """
    USER_PROMPT = f"""
When given 'Related qa', use them as a reference to answer the 'Question'. If the 'Related Questions and Answers' are not relevant to the 'Question', ignore them.
Instructions:
1. Analyze the 'Question' carefully.
2. Review the 'Related Questions and Answers' for relevant information.
3. If the question is about the smart store, provide a concise and accurate answer based on the related information.
4. If the question is unrelated to the smart store, respond with "Not relevant".
5. Always maintain a professional and helpful tone.
6. Use Korean when question is korean
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