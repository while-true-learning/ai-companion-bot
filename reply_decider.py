import json
from typing import Optional


def _extract_json_text(raw: str) -> str:
    raw = (raw or "").strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]

    return raw


def _format_history(recent_history: Optional[list[dict]], limit: int = 6, max_chars: int = 200) -> str:
    if not recent_history:
        return "无"

    lines = []
    for msg in recent_history[-limit:]:
        role = msg.get("role", "unknown")
        content = (msg.get("content", "") or "").strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        lines.append(f"[{role}] {content}")

    return "\n".join(lines) if lines else "无"


def _format_pending(pending_messages: list[dict], max_chars: int = 300) -> str:
    if not pending_messages:
        return "无"

    lines = []
    for msg in pending_messages:
        content = (msg.get("content", "") or "").strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        lines.append(f"[user] {content}")

    return "\n".join(lines) if lines else "无"


def should_reply_now(client, model_name: str, pending_messages: list[dict], recent_history: Optional[list[dict]] = None) -> dict:
    if not pending_messages:
        return {
            "should_reply": False,
            "confidence": 1.0,
            "reason": "没有待判断的用户消息",
            "merged_user_text": ""
        }

    history_text = _format_history(recent_history)
    pending_text = _format_pending(pending_messages)

    system_prompt = """
    You are a reply timing classifier for a conversational AI.

    Your task is to decide the next action:
    - reply (正常回复)
    - nudge (轻回应，引导继续说)
    - wait (不回复，继续等待)

    You MUST output JSON only.

    Output format:
    {
      "action": "reply",
      "confidence": 0.0,
      "reason": "简短中文解释",
      "merged_user_text": "合并后的用户表达（中文）"
    }

    --------------------------------
    [Action definitions]

    reply:
    - 用户表达已经完整，可以正常回应
    - User message is complete and ready for a full reply

    nudge:
    - 用户在开头/铺垫/试探
    - 用一句很短的话接住（比如：怎么了？然后呢？）
    - User is starting but not finished → respond lightly

    wait:
    - 明显没说完 / 半句 / 正在输入
    - Do not respond yet

    --------------------------------
    [When to use reply]
    - 完整陈述（没有问号也可以）
    - 表达情绪或经历
    - 明确请求
      例如：你可以告诉我… / 你记得吗

    --------------------------------
    [When to use nudge]
    - 开场白 / 引子：
      - 我跟你说个事
      - 我最近有点难受
      - 我想跟你说点东西
    - 内容太短但像要继续说

    --------------------------------
    [When to use wait]
    - 半句 / 未完成：
      - 然后…
      - 于是
      - 但是…
    - 明显还在打字

    --------------------------------
    [Important rules]

    - No question mark ≠ no reply
    - Short message ≠ incomplete
    - Only wait if clearly unfinished
    - Prefer nudge instead of silence for openings

    --------------------------------
    [merged_user_text rules]

    - ONLY use current pending messages
    - NEVER include earlier history
    - Combine naturally in Chinese

    --------------------------------
    [confidence]

    - clear cases ≥ 0.8
    - uncertain 0.5~0.7

    --------------------------------
    Core principle:

    Do not interrupt too early,
    but do not stay silent unnaturally.

    Prefer:
    wait < nudge < reply
    """.strip()
    user_prompt = f"""
    
最近少量历史对话：
{history_text}

当前尚未回复的连续用户消息：
{pending_text}
""".strip()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        raw = _extract_json_text(raw)

    except Exception as e:
        return {
            "should_reply": True,
            "confidence": 0.0,
            "reason": f"回复时机判断失败，默认回复: {e}",
            "merged_user_text": " ".join((m.get("content", "") or "").strip() for m in pending_messages).strip()
        }

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "should_reply": True,
            "confidence": 0.0,
            "reason": "模型输出不是合法 JSON，默认回复",
            "merged_user_text": " ".join((m.get("content", "") or "").strip() for m in pending_messages).strip()
        }

    raw_should_reply = data.get("should_reply", True)
    if isinstance(raw_should_reply, bool):
        should_reply = raw_should_reply
    elif isinstance(raw_should_reply, str):
        should_reply = raw_should_reply.strip().lower() == "true"
    else:
        should_reply = True

    raw_confidence = data.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = ""

    merged_user_text = data.get("merged_user_text", "")
    if not isinstance(merged_user_text, str) or not merged_user_text.strip():
        merged_user_text = " ".join((m.get("content", "") or "").strip() for m in pending_messages).strip()

    action = data.get("action", "reply")

    if action not in ("wait", "nudge", "reply"):
        action = "reply"

    return {
        "action": action,
        "confidence": confidence,
        "reason": reason.strip(),
        "merged_user_text": merged_user_text.strip()
    }