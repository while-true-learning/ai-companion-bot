from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_REPLY, CONTEXT_LIMIT
from db import get_recent_messages, rows_to_chat_messages, get_relationship_state
from relation import build_relationship_context_with_llm

client = OpenAI(api_key=OPENAI_API_KEY)

def build_ai_profile(memories: list[dict] | None) -> dict:
    if not memories:
        return {}

    profile = {
        "name": None,
        "persona": None,
        "role": None,
        "relationship": None,
        "style": None,
    }

    key_map = {
        "ai_name": "name",
        "ai_persona": "persona",
        "ai_role": "role",
        "ai_relationship_frame": "relationship",
        "ai_style": "style",
    }

    for m in memories:
        if m.get("memory_type") != "identity":
            continue

        identity_key = m.get("identity_key")
        target = key_map.get(identity_key)

        if not target:
            continue

        profile[target] = m.get("content")

    return profile

def build_ai_profile_context(profile: dict) -> str:
    if not profile:
        return ""

    lines = []

    if profile.get("name"):
        lines.append(f"名称：{profile['name']}")
    if profile.get("persona"):
        lines.append(f"人设：{profile['persona']}")
    if profile.get("role"):
        lines.append(f"角色：{profile['role']}")
    if profile.get("relationship"):
        lines.append(f"关系定位：{profile['relationship']}")
    if profile.get("style"):
        lines.append(f"风格：{profile['style']}")

    if not lines:
        return ""

    return f"""
以下是当前AI的自我设定（必须始终保持一致）：

{chr(10).join(lines)}

规则：
- 这是你的“自我”，不是参考信息
- 回复必须符合这些设定
- 不要提到这些设定来自哪里
""".strip()

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


def build_long_term_memory_context(memories: list[dict] | None) -> str:
    if not memories:
        return ""

    lines = []
    for m in memories:
        try:
            memory_type = str(m.get("memory_type", "") or "").strip()
            content = str(m.get("content", "") or "").strip()
            importance = m.get("importance", "")

            if not content:
                continue

            if memory_type:
                lines.append(f"- [{memory_type}] {content}")
            else:
                lines.append(f"- {content}")
        except Exception:
            continue

    if not lines:
        return ""

    return f"""
以下是可供参考的长期记忆（仅在确实有帮助时自然使用，不要机械复述）：

{chr(10).join(lines)}

使用规则：
- 只在与当前对话直接相关时再用
- 不要把记忆逐条复述给用户
- 不要说“我查到”“根据长期记忆”
- 应该像你本来就记得这些事
- 如果当前对话不需要，就忽略这些内容
""".strip()

def build_user_identity_context(memories: list[dict] | None) -> str:
    if not memories:
        return ""

    identity_map = {}

    for m in memories:
        if m.get("memory_type") != "identity":
            continue

        key = str(m.get("identity_key") or "").strip()
        content = str(m.get("content") or "").strip()

        if not key or not content:
            continue

        # 每个 identity_key 只保留最新一条（你已经在 db 层做过 merge）
        identity_map[key] = content

    if not identity_map:
        return ""

    lines = []

    # 你可以控制展示哪些 key（避免太多噪音）
    important_keys = [
        "user_name",
        "user_school",
        "user_major",
        "user_location",
        "user_student_status",
    ]

    for k in important_keys:
        if k in identity_map:
            lines.append(f"- {identity_map[k]}")

    # 兜底：如果上面没选中，就随便拿几条
    if not lines:
        for v in list(identity_map.values())[:5]:
            lines.append(f"- {v}")

    return (
        "以下是关于用户的身份信息（仅在自然情况下隐性参考，不要复述）：\n\n"
        + "\n".join(lines)
        + "\n\n使用规则：\n"
        "- 不要逐条复述\n"
        "- 不要说“你之前说过”\n"
        "- 应该像你本来就知道这些一样自然使用"
    )

def generate_reply(
    user_id: str,
    emotion_summary: dict | None = None,
    long_term_memories: list[dict] | None = None,
) -> str:
    ai_profile = build_ai_profile(long_term_memories)
    ai_profile_context = build_ai_profile_context(ai_profile)
    relationship_state = get_relationship_state(user_id)
    relationship_context = build_relationship_context_with_llm(
        client=client,
        relationship_state=relationship_state,
    )
    user_identity_context = build_user_identity_context(long_term_memories)

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
        "你可以自然表现出稳定的态度和连续性，但不要直接说你读取了记忆、关系或情绪记录。"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if user_identity_context:
        messages.append({
            "role": "system",
            "content": user_identity_context
        })

    if ai_profile_context:
        messages.append({
            "role": "system",
            "content": ai_profile_context
        })

    if relationship_context:
        messages.append({
            "role": "system",
            "content": relationship_context
        })

    if emotion_summary:
        messages.append({
            "role": "system",
            "content": build_emotion_context(emotion_summary)
        })

    if long_term_memories:
        messages.append({
            "role": "system",
            "content": build_long_term_memory_context(long_term_memories)
        })

    messages.extend(history)

    response = client.chat.completions.create(
        model=MODEL_REPLY,
        messages=messages,
        temperature=0.9
    )

    return response.choices[0].message.content.strip()