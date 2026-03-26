import json

from config import MODEL_DECIDER, MODEL_SUMMARIZE
from db import (
    get_recent_memories_for_dedup,
    save_memory,
    update_memory,
)

ALLOWED_MEMORY_TYPES = {"preference", "trait", "event", "pattern"}

IMPORTANCE_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.70
DEDUP_LOOKBACK_LIMIT = 20


# =========================
# 基础工具
# =========================

def _safe_float(value, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp_score(value, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _normalize_memory_type(memory_type: str | None) -> str:
    memory_type = (memory_type or "pattern").strip()
    if memory_type not in ALLOWED_MEMORY_TYPES:
        return "pattern"
    return memory_type


def _normalize_memory_item(item: dict) -> dict | None:
    content = str(item.get("content", "") or "").strip()
    if not content:
        return None

    return {
        "memory_type": _normalize_memory_type(item.get("memory_type")),
        "content": content,
        "importance": _clamp_score(item.get("importance", 0.5)),
        "confidence": _clamp_score(item.get("confidence", 0.5)),
    }

# =========================
# 消息工具
# 候选记忆只依赖 pending
# 会话总结依赖 session_messages
# =========================

def _normalize_pending_messages(pending_messages: list[dict] | None) -> list[dict]:
    if not pending_messages:
        return []

    result = []
    for msg in pending_messages:
        try:
            role = str(msg.get("role", "user")).strip()
            content = str(msg.get("content", "") or "").strip()
            created_at = str(msg.get("created_at", "") or "").strip()

            if not content:
                continue

            result.append({
                "role": role,
                "content": content,
                "created_at": created_at,
            })
        except Exception:
            continue

    return result


def _format_messages(messages: list[dict] | None, max_items: int = 8) -> str:
    if not messages:
        return "无"

    lines = []
    for msg in messages[-max_items:]:
        role = str(msg.get("role", "unknown")).strip()
        content = str(msg.get("content", "") or "").strip()
        created_at = str(msg.get("created_at", "") or "").strip()

        if not content:
            continue

        if created_at:
            lines.append(f"[{created_at}] {role}: {content}")
        else:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "无"


def _merge_messages_text(messages: list[dict] | None) -> str:
    if not messages:
        return ""

    parts = []
    for msg in messages:
        content = str(msg.get("content", "") or "").strip()
        if content:
            parts.append(content)
    return " ".join(parts).strip()


# =========================
# 当前轮候选记忆提取
# 只看 pending，不看历史
# =========================

def extract_preference_memory(
    client,
    pending_messages: list[dict],
) -> dict | None:
    normalized_pending = _normalize_pending_messages(pending_messages)

    pending_text = _merge_messages_text(normalized_pending)
    if not pending_text:
        return None

    pending_context_text = _format_messages(normalized_pending, max_items=12)

    system_prompt = """
You are a conversational preference memory extractor.
你是一个对话中的“记忆候选提取器”。

Your task:
Read the current user input in this turn only, and decide whether it contains information that can be extracted as a reusable memory.

你只需要判断：当前这一轮用户输入中，是否包含可以提取为“偏好”的信息。

[What SHOULD be extracted]
1. Preferences
2. Personal facts
3. Clear habits or tendencies
4. Interaction expectations
5. Explicit remember signals

[What should NOT be extracted]
1. Pure small talk
2. Noise / filler
3. Single short emotional expressions without information

[How to write content]
- One sentence
- Clear and reusable
- Rewrite into a memory-style statement

[Memory type]
- preference
- trait
- event
- pattern
- null (only if should_store = false)

[Scoring]
importance = usefulness of the information itself
confidence = confidence in the extraction

Return JSON only:
{
  "should_store": true,
  "memory_type": "preference|trait|event|pattern|null",
  "content": "必须是数据库风格总结句",
  "importance": 0.0,
  "confidence": 0.0
}
""".strip()

    user_prompt = f"""
当前轮内容：
{pending_context_text}
""".strip()

    resp = client.chat.completions.create(
        model=MODEL_DECIDER,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)

    if not data.get("should_store"):
        return None

    return _normalize_memory_item(data)

# =========================
# 去重 / 合并决策
# =========================

def resolve_memory_actions(
    client,
    candidates: list[dict],
    existing_memories: list[dict],
) -> list[dict]:
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
        model=MODEL_DECIDER,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "candidates": candidates,
                        "existing_memories": existing_memories,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)
    raw_actions = data.get("actions", [])

    actions = []
    for action in raw_actions:
        kind = str(action.get("action", "") or "").strip()
        if kind not in {"insert", "update_existing", "skip"}:
            continue

        candidate_index = action.get("candidate_index")
        if not isinstance(candidate_index, int):
            continue
        if candidate_index < 0 or candidate_index >= len(candidates):
            continue

        item = {
            "action": kind,
            "candidate_index": candidate_index,
            "existing_memory_id": action.get("existing_memory_id"),
        }

        if kind != "skip":
            normalized = _normalize_memory_item(action)
            if normalized is None:
                continue
            item.update(normalized)

        actions.append(item)

    return actions


# =========================
# 通用写库执行层
# 这里注释掉筛选器
# =========================

def _apply_memory_actions(
    user_id: str,
    actions: list[dict],
    log_prefix: str = "[MEMORY]",
):
    if not actions:
        print(f"{log_prefix} no actions")
        return

    for action in actions:
        kind = action.get("action")

        if kind == "skip":
            print(f"{log_prefix} skip -> candidate_index={action.get('candidate_index')}")
            continue

        content = str(action.get("content", "") or "").strip()
        if not content:
            print(f"{log_prefix} empty content after resolve")
            continue

        memory_type = _normalize_memory_type(action.get("memory_type"))
        importance = _clamp_score(action.get("importance", 0.5))
        confidence = _clamp_score(action.get("confidence", 0.5))

        # if not _is_memory_score_passed(importance, confidence):
        #     print(
        #         f"{log_prefix} below threshold -> "
        #         f"importance={importance}, confidence={confidence}, content={content}"
        #     )
        #     continue

        if kind == "insert":
            memory_id = save_memory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                importance=importance,
                confidence=confidence,
                source_message_id=None,
            )
            print(f"{log_prefix} insert #{memory_id} -> {content}")

        elif kind == "update_existing":
            existing_id = action.get("existing_memory_id")
            if existing_id is None:
                print(f"{log_prefix} update_existing but missing existing_memory_id")
                continue

            update_memory(
                memory_id=int(existing_id),
                content=content,
                importance=importance,
                confidence=confidence,
            )
            print(f"{log_prefix} update #{existing_id} -> {content}")


def process_memory(
    client,
    user_id: str,
    pending_messages: list[dict],
):
    normalized_pending = _normalize_pending_messages(pending_messages)
    merged_text = _merge_messages_text(normalized_pending)

    print(f"[MEMORY] input -> {merged_text or '空'}")

    memory_item = extract_preference_memory(
        client=client,
        pending_messages=normalized_pending,
    )

    if not memory_item:
        print("[MEMORY] no memory extracted")
        return

    print("[MEMORY EXTRACTED]", memory_item)

    existing_memories = get_recent_memories_for_dedup(
        user_id=user_id,
        limit=DEDUP_LOOKBACK_LIMIT,
    )

    actions = resolve_memory_actions(
        client=client,
        candidates=[memory_item],
        existing_memories=existing_memories,
    )

    _apply_memory_actions(
        user_id=user_id,
        actions=actions,
        log_prefix="[MEMORY]",
    )