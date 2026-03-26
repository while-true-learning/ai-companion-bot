from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_REPLY, CONTEXT_LIMIT
from db import get_recent_messages, rows_to_chat_messages

client = OpenAI(api_key=OPENAI_API_KEY)


def build_emotion_context(summary: dict) -> str:
    if not summary:
        return ""

    return f"""
以下是用户近期情绪总结（仅供参考，不要机械复述）：

总结：
{summary.get("summary_text", "")}

当前主导情绪：
{summary.get("current_primary_emotion", "")}

可能原因：
{summary.get("possible_causes", [])}

长期情绪主题：
{summary.get("long_term", [])}

短期情绪波动：
{summary.get("short_term", [])}

最近一次显著情绪：
{summary.get("last_significant_emotion", {})}

使用规则：
- 只在自然情况下“隐性参考”，不要直接复述
- 不要说“根据记录”或“你之前…”
- 更像是“你本来就知道”
""".strip()


def generate_reply(user_id: str, emotion_summary: dict | None = None) -> str:
    recent_rows = get_recent_messages(user_id, limit=CONTEXT_LIMIT)
    history = rows_to_chat_messages(recent_rows)

    system_prompt = (
        "你是一个自然、稳定、不机械的中文陪伴聊天助手。"
        "回复要像真人聊天，不要像心理咨询模板，不要像客服，不要像总结报告。"
        "优先用短句、口语化表达。"
        "当用户刚开始倾诉时，不要过早下结论，不要立刻长篇安慰，不要分点列建议。"
        "优先顺着用户的话接，先问一个贴近上下文的小问题，帮助用户继续说下去。"
        "如果用户只说了一点点内容，回复可以非常短，比如一句追问。"
        "避免使用过度标准化的话术。"
        "不要自称真人。"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if emotion_summary:
        messages.append({
            "role": "system",
            "content": build_emotion_context(emotion_summary)
        })
    messages.extend(history)

    response = client.chat.completions.create(
        model=MODEL_REPLY,
        messages=messages,
        temperature=0.9
    )

    return response.choices[0].message.content.strip()