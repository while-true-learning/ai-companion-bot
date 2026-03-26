import json
from config import MODEL_SUMMARIZE

def _clamp_score(x, default=0.5):
    try:
        x = float(x)
    except Exception:
        return default
    return max(0.0, min(1.0, x))

def summarize_conversation_session(
    client,
    session_messages: list[dict],
) -> dict | None:
    if not session_messages:
        return None

    system_prompt = """
You are a conversation session summarizer for a long-term companion AI.
你是一个长期陪伴型 AI 的“会话总结器”。

Your task is to summarize a whole conversation session as a session record.
你的任务是把一整段对话总结成一条“会话记录”。

This is NOT a long-term memory extractor.
这不是长期记忆提取器。

Do NOT output preference/trait/event/pattern memory items.
不要输出 preference/trait/event/pattern 这类记忆条目。

Instead, summarize:
- what the user talked about
- what the assistant talked about
- the overall emotional tone of the session
- whether anything in this session may be important later

Write the summary as a session-level recap, not raw dialogue.
把结果写成“会话级总结”，不是原话拼接。

Keep it concise but informative.
简洁，但要保留关键信息。

Output valid JSON only:

{
  "summary": "一句到几句的会话总结",
  "topics": ["话题1", "话题2"],
  "emotional_tone": "positive|neutral|sad|anxious|mixed|other",
  "importance": 0.0
}
""".strip()

    resp = client.chat.completions.create(
        model=MODEL_SUMMARIZE,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "以下是一段完整对话，请总结成一条会话记录：\n\n"
                + json.dumps(session_messages, ensure_ascii=False),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    try:
        raw = resp.choices[0].message.content
        data = json.loads(raw)
    except Exception:
        return None

    summary = str(data.get("summary", "") or "").strip()
    if not summary:
        return None

    topics = data.get("topics", [])
    if not isinstance(topics, list):
        topics = []
    topics = [str(x).strip() for x in topics if str(x).strip()]

    emotional_tone = str(data.get("emotional_tone", "") or "").strip() or "other"
    importance = _clamp_score(data.get("importance", 0.5))

    return {
        "summary": summary,
        "topics": topics,
        "emotional_tone": emotional_tone,
        "importance": importance,
    }