import json


ALLOWED_MEMORY_TYPES = {"preference", "trait", "event", "pattern"}


def _safe_float(value, default=0.5) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_memory_item(item: dict) -> dict | None:
    memory_type = (item.get("memory_type") or "pattern").strip()
    content = (item.get("content") or "").strip()

    if memory_type not in ALLOWED_MEMORY_TYPES:
        memory_type = "pattern"

    if not content:
        return None

    importance = max(0.0, min(1.0, _safe_float(item.get("importance", 0.5), 0.5)))
    confidence = max(0.0, min(1.0, _safe_float(item.get("confidence", 0.5), 0.5)))

    return {
        "memory_type": memory_type,
        "content": content,
        "importance": importance,
        "confidence": confidence,
    }


def extract_memory_candidate(client, model_name: str, user_text: str, recent_messages: list[dict]) -> dict | None:
    system_prompt = """
你是一个对话记忆提取器。
你的任务是判断，用户这次输入中是否包含值得长期记忆的信息。

只有以下内容值得记忆：
1. 稳定偏好（喜欢/讨厌什么风格、方式、习惯）
2. 稳定特征（长期行为模式、沟通特点）
3. 重要事件（会影响之后互动的重要信息）
4. 持续性需求（以后回复时需要参考的要求）

不要记忆：
1. 一次性的闲聊
2. 没有长期价值的情绪波动
3. 重复、空泛、过于普通的话
4. 单纯的寒暄

输出必须是 JSON：
{
  "should_store": true/false,
  "memory_type": "preference|trait|event|pattern|null",
  "content": "一句简洁明确的话",
  "importance": 0.0-1.0,
  "confidence": 0.0-1.0
}
""".strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_messages[-6:])
    messages.append({
        "role": "user",
        "content": f"请判断下面这句话是否值得纳入长期记忆，并按要求输出 JSON：\n\n{user_text}"
    })

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    text = resp.choices[0].message.content
    data = json.loads(text)

    if not data.get("should_store"):
        return None

    return _normalize_memory_item(data)


def summarize_session_memories(client, model_name: str, session_messages: list[dict]) -> list[dict]:
    if not session_messages:
        return []

    system_prompt = """
你是一个会话级记忆提取器。
请从一整段对话中提取最多 3 条值得长期记忆的信息。

只保留：
1. 稳定偏好
2. 稳定特征
3. 重要事件
4. 持续性需求

不要保留：
1. 瞬时情绪
2. 普通闲聊
3. 与已有表达重复的细枝末节
4. 只对当前一轮有用的信息

要求：
- 输出简洁、可复用、面向未来的记忆句子
- 不要复读原话
- 每条记忆一句话
- 最多输出 3 条

输出 JSON：
{
  "memories": [
    {
      "memory_type": "preference|trait|event|pattern",
      "content": "一句简洁明确的话",
      "importance": 0.0,
      "confidence": 0.0
    }
  ]
}
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "以下是一段完整对话，请提取值得长期保存的记忆：\n\n"
                           + json.dumps(session_messages, ensure_ascii=False)
            }
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    data = json.loads(resp.choices[0].message.content)
    raw_memories = data.get("memories", [])

    result = []
    for item in raw_memories[:3]:
        normalized = _normalize_memory_item(item)
        if normalized is not None:
            result.append(normalized)

    return result


def resolve_memory_actions(client, model_name: str, candidates: list[dict], existing_memories: list[dict]) -> list[dict]:
    if not candidates:
        return []

    system_prompt = """
你是一个记忆去重与合并决策器。
你会看到“新记忆候选”和“已有记忆”。

对每个新候选，只能输出以下动作之一：
1. insert: 这是新的记忆，应插入
2. update_existing: 这是已有记忆的更完整版本，应更新旧记忆
3. skip: 这和已有记忆重复，跳过

要求：
- 如果语义基本相同，不要重复插入
- 如果只是旧记忆的更清晰或更完整版本，优先 update_existing
- 只有明确是新信息时才 insert
- existing_memory_id 只有在 update_existing 时填写
- candidate_index 对应 candidates 的下标

输出 JSON：
{
  "actions": [
    {
      "action": "insert|update_existing|skip",
      "candidate_index": 0,
      "existing_memory_id": 12,
      "memory_type": "preference|trait|event|pattern",
      "content": "更新后或插入时要保存的内容",
      "importance": 0.0,
      "confidence": 0.0
    }
  ]
}
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps({
                    "candidates": candidates,
                    "existing_memories": existing_memories
                }, ensure_ascii=False)
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    data = json.loads(resp.choices[0].message.content)
    raw_actions = data.get("actions", [])

    actions = []
    for action in raw_actions:
        kind = (action.get("action") or "").strip()
        if kind not in {"insert", "update_existing", "skip"}:
            continue

        candidate_index = action.get("candidate_index")
        if not isinstance(candidate_index, int):
            continue
        if candidate_index < 0 or candidate_index >= len(candidates):
            continue

        normalized = _normalize_memory_item(action)
        if kind != "skip" and normalized is None:
            continue

        item = {
            "action": kind,
            "candidate_index": candidate_index,
            "existing_memory_id": action.get("existing_memory_id"),
        }

        if normalized is not None:
            item.update(normalized)

        actions.append(item)

    return actions