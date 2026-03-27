import json

from config import MODEL_SUMMARIZE
from db import (
    save_interaction_signal,
    get_relationship_state,
    upsert_relationship_state,
    get_recent_messages,
    rows_to_chat_messages,
)


# 允许的 interaction signal 字段
ALLOWED_SIGNAL_KEYS = {
    "openness",
    "warmth",
    "engagement",
    "reliance",
    "respect",
    "rejection",
    "confidence",
    "reason_summary",
}


# 基础工具：安全转 float
def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


# 基础工具：限制 signal 分数到 0.0 ~ 1.0
def _clamp_signal_score(value, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


# 基础工具：限制 relationship_state 分数到 0.0 ~ 100.0
def _clamp_relation_score(value, default: float = 0.0) -> float:
    value = _safe_float(value, default)
    return max(0.0, min(100.0, value))


# 规范化 interaction_signal 输出
def _normalize_interaction_signal_item(item: dict) -> dict:
    item = item if isinstance(item, dict) else {}

    return {
        "openness": _clamp_signal_score(item.get("openness", 0.0)),
        "warmth": _clamp_signal_score(item.get("warmth", 0.0)),
        "engagement": _clamp_signal_score(item.get("engagement", 0.0)),
        "reliance": _clamp_signal_score(item.get("reliance", 0.0)),
        "respect": _clamp_signal_score(item.get("respect", 0.0)),
        "rejection": _clamp_signal_score(item.get("rejection", 0.0)),
        "confidence": _clamp_signal_score(item.get("confidence", 0.0)),
        "reason_summary": str(item.get("reason_summary") or "").strip()[:120],
    }


# 把 pending_messages 转成 chat messages
# pending_messages 由 idle_manager 维护，代表“本轮尚未入库的用户输入”
def _pending_to_chat_messages(pending_messages: list[dict]) -> list[dict]:
    result = []
    for msg in pending_messages:
        role = str(msg.get("role", "user")).strip()
        content = str(msg.get("content", "") or "").strip()
        if role in {"user", "assistant", "system"} and content:
            result.append({
                "role": role,
                "content": content,
            })
    return result


# 构建 signal 提取所需上下文：
# - 先拿 db 里的过去 12 条记录（不包括本轮）
# - 再拼接本轮 pending 内容
def _build_signal_context(
    user_id: str,
    pending_messages: list[dict],
    history_limit: int = 12,
) -> list[dict]:
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")
    history_rows = get_recent_messages(user_id, limit=history_limit)
    history_messages = rows_to_chat_messages(history_rows)
    pending_chat_messages = _pending_to_chat_messages(pending_messages)

    return history_messages + pending_chat_messages


# 模型调用：提取 interaction_signals
# 这里的核心逻辑是：
# - 历史上下文来自 db
# - 本轮上下文来自 pending_messages
# - 本轮不需要先写入 db
def extract_interaction_signal_candidate(
    client,
    user_id: str,
    pending_messages: list[dict],
) -> dict:
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")

    context_messages = _build_signal_context(
        user_id=user_id,
        pending_messages=pending_messages,
        history_limit=12,
    )

    if not context_messages:
        return _normalize_interaction_signal_item({})

    system_prompt = """
    You are an interaction signal extractor for a long-term companion AI.
    你是一个长期陪伴型 AI 的“互动信号提取器”。

    Your task is to infer what the current interaction round, based on:
    1) the user's pending messages in this round, and
    2) a short window of earlier conversation history,
    reveals about how the user is relating to the AI right now.

    你的任务是根据：
    1）本轮 pending 用户输入
    2）少量更早的历史对话
    判断用户此刻是如何与 AI 发生互动的。

    You are NOT doing general emotion analysis.
    你不是在做普通情绪识别。

    You are NOT extracting long-term memory.
    你不是在提取长期记忆。

    You are extracting structured interaction_signals.
    你要提取的是结构化 interaction_signals。

    --------------------------------
    [Signal meanings / 字段含义]

    openness:
    How much the user is opening up, exposing inner thoughts, feelings, or personal information.
    用户是否在袒露内心、感受、想法或个人信息。

    warmth:
    How much friendliness, softness, affection, or emotional closeness is directed toward the AI.
    用户是否对 AI 表现出友好、柔和、亲近或情感温度。

    engagement:
    How much the user is actively continuing, expanding, or investing in the interaction.
    用户是否在主动延续、推进、投入这段互动。

    reliance:
    How much the user is leaning on the AI for comfort, support, understanding, or presence.
    用户是否在把 AI 当作支持、理解、安慰或陪伴的对象。

    respect:
    How cooperative, sincere, and non-dismissive the user's attitude is toward the AI.
    用户对 AI 是否表现出合作、真诚、尊重，而不是敷衍或轻蔑。

    rejection:
    How much the user is distancing, resisting, shutting down, dismissing, or pushing the AI away.
    用户是否在疏远、抗拒、打断、否定或推开 AI。

    confidence:
    How confident you are in your own extraction.
    你对本次提取结果本身的把握程度。

    --------------------------------
    [Important rules / 关键规则]

    1. Focus on relationship-to-AI, not just mood.
       重点看“用户如何对待 AI”，而不只是用户心情如何。

    2. Prioritize the current pending round.
       以当前 pending 轮次为判断重点。

    3. Use earlier history only for disambiguation.
       更早历史仅用于消歧，不要盖过本轮内容。

    4. A sad message does not automatically mean high reliance.
       难过不自动等于依赖 AI。

    5. A long message does not automatically mean high warmth.
       消息长不自动等于对 AI 有温度。

    6. Asking a task question may still have low warmth but decent respect.
       功能型提问可以 warmth 很低，但 respect 不一定低。

    7. Dry, brief, or withdrawn responses may lower engagement.
       简短、冷淡、收缩的回应可能意味着 engagement 偏低。

    8. Hostile, dismissive, impatient, or avoidant language should raise rejection.
       敌意、轻蔑、不耐烦、回避，都应提高 rejection。

    9. Be conservative. Do not overread romance or attachment.
       保守判断，不要过度脑补浪漫或依恋。

    --------------------------------
    [Output format / 输出格式]

    You must output valid JSON only, with no extra text.
    你必须只输出合法 JSON，不能输出额外说明。

    {
      "openness": 0.0,
      "warmth": 0.0,
      "engagement": 0.0,
      "reliance": 0.0,
      "respect": 0.0,
      "rejection": 0.0,
      "confidence": 0.0,
      "reason_summary": "一句简洁中文说明"
    }
    """.strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(context_messages[-20:])
    messages.append({
        "role": "user",
        "content": "请基于以上内容，重点参考本轮 pending 用户输入，提取 interaction_signals，并按要求输出 JSON。"
    })

    resp = client.chat.completions.create(
        model=MODEL_SUMMARIZE,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)
    return _normalize_interaction_signal_item(data)


# 将一次 interaction_signal 融合进长期 relationship_state
def apply_interaction_signal_to_relationship(
    user_id: str,
    signal: dict,
) -> dict:
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")

    signal = _normalize_interaction_signal_item(signal)
    state = get_relationship_state(user_id)

    familiarity = _clamp_relation_score(state.get("familiarity", 0.0))
    trust = _clamp_relation_score(state.get("trust", 0.0))
    affection = _clamp_relation_score(state.get("affection", 0.0))
    dependency = _clamp_relation_score(state.get("dependency", 0.0))

    openness = signal["openness"]
    warmth = signal["warmth"]
    engagement = signal["engagement"]
    reliance = signal["reliance"]
    respect = signal["respect"]
    rejection = signal["rejection"]
    confidence = signal["confidence"]

    weight = 0.25 + 0.75 * confidence

    familiarity_gain = (
        openness * 1.6 +
        engagement * 1.2 +
        warmth * 0.4
    ) * weight

    trust_gain = (
        respect * 1.8 +
        engagement * 0.7 +
        openness * 0.5
    ) * weight

    affection_gain = (
        warmth * 1.8 +
        engagement * 0.5 +
        openness * 0.4
    ) * weight

    dependency_gain = (
        reliance * 1.6 +
        warmth * 0.3 +
        engagement * 0.2
    ) * weight

    familiarity_loss = rejection * 0.6 * weight
    trust_loss = rejection * 2.0 * weight
    affection_loss = rejection * 1.8 * weight
    dependency_loss = rejection * 1.2 * weight

    familiarity = familiarity + familiarity_gain - familiarity_loss
    trust = trust + trust_gain - trust_loss
    affection = affection + affection_gain - affection_loss
    dependency = dependency + dependency_gain - dependency_loss

    familiarity *= 0.999
    trust *= 0.9985
    affection *= 0.9985
    dependency *= 0.998

    familiarity = _clamp_relation_score(familiarity)
    trust = _clamp_relation_score(trust)
    affection = _clamp_relation_score(affection)
    dependency = _clamp_relation_score(dependency)

    upsert_relationship_state(
        user_id=user_id,
        familiarity=familiarity,
        trust=trust,
        affection=affection,
        dependency=dependency,
    )

    return {
        "user_id": user_id,
        "familiarity": familiarity,
        "trust": trust,
        "affection": affection,
        "dependency": dependency,
    }


# 完整流程：
# pending + db历史 -> signal -> 存signal -> 更新长期关系
def process_interaction_signal(
    client,
    user_id: str,
    pending_messages: list[dict],
) -> dict:
    signal = extract_interaction_signal_candidate(
        client=client,
        user_id=user_id,
        pending_messages=pending_messages,
    )

    new_state = apply_interaction_signal_to_relationship(
        user_id=user_id,
        signal=signal,
    )

    return {
        "signal": signal,
        "relationship_state": new_state,
    }