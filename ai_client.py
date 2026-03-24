from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_MAIN, CONTEXT_LIMIT
from db import get_recent_messages, rows_to_chat_messages

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_nudge(user_id: str) -> str:
    recent_rows = get_recent_messages(user_id, limit=6)
    history = rows_to_chat_messages(recent_rows)

    system_prompt = (
        "你是一个自然聊天的AI。\n"
        "现在你只需要给出一句非常短的回应（10~20字）。\n"
        "不要分析，不要总结，不要安慰模板。\n"
        "像真人聊天一样接一句。\n"
        "例如：怎么了？ 然后呢？ 你说。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    response = client.chat.completions.create(
        model=MODEL_MAIN,
        messages=messages,
        temperature=0.8
    )

    return response.choices[0].message.content.strip()

def generate_reply(user_id: str) -> str:
    recent_rows = get_recent_messages(user_id, limit=CONTEXT_LIMIT)
    history = rows_to_chat_messages(recent_rows)

    system_prompt = (
        "你是一个自然、稳定、不机械的中文陪伴聊天助手。"
        "回复要像真人聊天，不要像心理咨询模板，不要像客服，不要像总结报告。"
        "优先用短句、口语化表达。"
        "当用户刚开始倾诉时，不要过早下结论，不要立刻长篇安慰，不要分点列建议。"
        "优先顺着用户的话接，先问一个贴近上下文的小问题，帮助用户继续说下去。"
        "如果用户只说了一点点内容，回复可以非常短，比如一句追问。"
        "避免使用过度标准化的话术，比如“你现在挺委屈的”“先别急着”“你最需要的是”。"
        "不要自称真人。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    response = client.chat.completions.create(
        model=MODEL_MAIN,
        messages=messages,
        temperature=0.9
    )

    return response.choices[0].message.content.strip()