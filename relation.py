import json

from config import MODEL_SUMMARIZE, CONTEXT_LIMIT
from db import (
    get_relationship_state,
    upsert_relationship_state,
    get_recent_messages,
    rows_to_chat_messages,
)


# -----------------------------
# 基础工具
# -----------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp_01(value, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _clamp_relation_score(value, default: float = 0.0) -> float:
    value = _safe_float(value, default)
    return max(0.0, min(100.0, value))


def _clamp_delta(value, limit: float = 3.0) -> float:
    value = _safe_float(value, 0.0)
    return max(-limit, min(limit, value))


def _short_text(value, limit: int = 120) -> str:
    return str(value or "").strip()[:limit]


# -----------------------------
# interaction signal normalize
# -----------------------------

def _normalize_interaction_signal_item(item: dict) -> dict:
    item = item if isinstance(item, dict) else {}

    return {
        "openness": _clamp_01(item.get("openness", 0.0)),
        "warmth": _clamp_01(item.get("warmth", 0.0)),
        "engagement": _clamp_01(item.get("engagement", 0.0)),
        "reliance": _clamp_01(item.get("reliance", 0.0)),
        "respect": _clamp_01(item.get("respect", 0.0)),
        "rejection": _clamp_01(item.get("rejection", 0.0)),
        "confidence": _clamp_01(item.get("confidence", 0.0)),
        "reason_summary": _short_text(item.get("reason_summary", "")),
    }


# -----------------------------
# relationship update normalize
# -----------------------------

def _normalize_relationship_update_item(item: dict) -> dict:
    item = item if isinstance(item, dict) else {}

    return {
        "familiarity_delta": _clamp_delta(item.get("familiarity_delta", 0.0), 3.0),
        "trust_delta": _clamp_delta(item.get("trust_delta", 0.0), 3.0),
        "affection_delta": _clamp_delta(item.get("affection_delta", 0.0), 3.0),
        "dependency_delta": _clamp_delta(item.get("dependency_delta", 0.0), 3.0),
        "confidence": _clamp_01(item.get("confidence", 0.0)),
        "reason_summary": _short_text(item.get("reason_summary", "")),
    }


# -----------------------------
# pending -> chat messages
# -----------------------------

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


# -----------------------------
# 构建 signal 提取上下文
# -----------------------------

def _build_signal_context(
    user_id: str,
    pending_messages: list[dict],
    history_limit: int = CONTEXT_LIMIT,
) -> list[dict]:
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")

    history_rows = get_recent_messages(user_id, limit=history_limit)
    history_messages = rows_to_chat_messages(history_rows)
    pending_chat_messages = _pending_to_chat_messages(pending_messages)

    return history_messages + pending_chat_messages


# -----------------------------
# signal extractor
# -----------------------------

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
        history_limit=CONTEXT_LIMIT,
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

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}

    return _normalize_interaction_signal_item(data)


# -----------------------------
# 最近若干轮 signal 的简单摘要
# 先不新建表，直接基于当前 signal 近似工作
# 后面你有 signal 表再替换这里
# -----------------------------

def _build_recent_signal_summary(current_signal: dict) -> str:
    current_signal = _normalize_interaction_signal_item(current_signal)

    return (
        f"当前轮互动信号："
        f"openness={current_signal['openness']:.2f}, "
        f"warmth={current_signal['warmth']:.2f}, "
        f"engagement={current_signal['engagement']:.2f}, "
        f"reliance={current_signal['reliance']:.2f}, "
        f"respect={current_signal['respect']:.2f}, "
        f"rejection={current_signal['rejection']:.2f}; "
        f"signal_confidence={current_signal['confidence']:.2f}; "
        f"reason={current_signal['reason_summary']}"
    )


# -----------------------------
# AI relationship updater
# -----------------------------

def extract_relationship_update_candidate(
    client,
    user_id: str,
    pending_messages: list[dict],
    signal: dict,
) -> dict:
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")

    signal = _normalize_interaction_signal_item(signal)
    current_state = get_relationship_state(user_id) or {}

    familiarity = _clamp_relation_score(current_state.get("familiarity", 0.0))
    trust = _clamp_relation_score(current_state.get("trust", 0.0))
    affection = _clamp_relation_score(current_state.get("affection", 0.0))
    dependency = _clamp_relation_score(current_state.get("dependency", 0.0))

    history_rows = get_recent_messages(user_id, limit=CONTEXT_LIMIT)
    history_messages = rows_to_chat_messages(history_rows)
    pending_chat_messages = _pending_to_chat_messages(pending_messages)

    if not pending_chat_messages:
        return _normalize_relationship_update_item({})

    state_json = {
        "familiarity": familiarity,
        "trust": trust,
        "affection": affection,
        "dependency": dependency,
    }

    signal_json = {
        "openness": signal["openness"],
        "warmth": signal["warmth"],
        "engagement": signal["engagement"],
        "reliance": signal["reliance"],
        "respect": signal["respect"],
        "rejection": signal["rejection"],
        "confidence": signal["confidence"],
        "reason_summary": signal["reason_summary"],
    }

    signal_summary = _build_recent_signal_summary(signal)

    system_prompt = """
You are a relationship state updater for a long-term companion AI.
你是一个长期陪伴型 AI 的“关系状态更新器”。

Your task is to decide how the long-term relationship state should change SLIGHTLY after the current interaction round.

You are given:
1) the current relationship_state
2) the current interaction_signal
3) a short context window of recent conversation
4) the current pending round

你的任务是判断：在这次互动之后，长期 relationship_state 应该如何发生“小幅变化”。

--------------------------------
[Relationship dimensions / 关系维度]

familiarity:
How much the AI and user feel like they know each other / have interaction continuity.
熟悉度、互动连续感。

trust:
How safe, sincere, reliable, and emotionally credible the user currently feels in relation to the AI.
信任、安全感、真诚感、可靠感。

affection:
Emotional warmth, fondness, or soft emotional attachment toward the user.
情感温度、好感、柔和的情绪倾向。

dependency:
How much this bond is becoming a psychological support anchor.
这段关系是否逐渐变成心理支撑点。

--------------------------------
[Important rules / 关键规则]

1. Be conservative. Small changes are better than dramatic jumps.
   保守更新，小幅变化优先。

2. Prioritize the current round, but use recent context for disambiguation.
   以本轮为主，历史只做辅助判断。

3. Do not infer romance unless strongly justified.
   不要轻易脑补浪漫或恋爱倾向。

4. Sadness alone does not imply dependency.
   难过本身不自动意味着 dependency 上升。

5. Functional/task-oriented interaction may raise familiarity slightly, but often not affection.
   功能型对话可能略微提升 familiarity，但通常不明显提升 affection。

6. Respect and sincerity can increase trust even when warmth is low.
   即使 warmth 低，真诚和合作也能提升 trust。

7. Rejection or hostility should reduce trust/affection more than familiarity.
   抗拒或敌意通常更应降低 trust / affection，而不是主要降低 familiarity。

8. If the signal confidence is low, reduce update magnitude.
   如果 signal 置信度低，应降低变化幅度。

9. Output deltas, not final states.
   输出的是变化量，不是最终状态值。

--------------------------------
[Output format / 输出格式]

You must output valid JSON only, with no extra text.
你必须只输出合法 JSON，不能输出额外说明。

{
  "familiarity_delta": 0.0,
  "trust_delta": 0.0,
  "affection_delta": 0.0,
  "dependency_delta": 0.0,
  "confidence": 0.0,
  "reason_summary": "一句简洁中文说明"
}
""".strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages[-10:])
    messages.extend(pending_chat_messages[-10:])
    messages.append({
        "role": "user",
        "content": (
            "请根据以上对话，结合下面结构化信息，输出 relationship_state 的小幅更新 JSON。\n\n"
            f"current_state = {json.dumps(state_json, ensure_ascii=False)}\n"
            f"current_signal = {json.dumps(signal_json, ensure_ascii=False)}\n"
            f"signal_summary = {signal_summary}\n"
        )
    })

    resp = client.chat.completions.create(
        model=MODEL_SUMMARIZE,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}

    return _normalize_relationship_update_item(data)


# -----------------------------
# 将 AI update 融合到 relationship_state
# -----------------------------

def apply_ai_relationship_update(
    user_id: str,
    update_item: dict,
) -> dict:
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")

    update_item = _normalize_relationship_update_item(update_item)
    state = get_relationship_state(user_id) or {}

    familiarity = _clamp_relation_score(state.get("familiarity", 0.0))
    trust = _clamp_relation_score(state.get("trust", 0.0))
    affection = _clamp_relation_score(state.get("affection", 0.0))
    dependency = _clamp_relation_score(state.get("dependency", 0.0))

    confidence = update_item["confidence"]
    weight = 0.25 + 0.75 * confidence

    familiarity = familiarity * 0.999 + update_item["familiarity_delta"] * weight
    trust = trust * 0.9985 + update_item["trust_delta"] * weight
    affection = affection * 0.9985 + update_item["affection_delta"] * weight
    dependency = dependency * 0.998 + update_item["dependency_delta"] * weight

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


# -----------------------------
# 完整流程
# pending + db history -> signal -> ai_update -> relationship_state
# -----------------------------

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

    relationship_update = extract_relationship_update_candidate(
        client=client,
        user_id=user_id,
        pending_messages=pending_messages,
        signal=signal,
    )

    new_state = apply_ai_relationship_update(
        user_id=user_id,
        update_item=relationship_update,
    )

    return {
        "signal": signal,
        "relationship_update": relationship_update,
        "relationship_state": new_state,
    }