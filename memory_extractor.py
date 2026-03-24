import json

from config import MODEL_DECISION
from db import (
    get_recent_memories_for_dedup,
    save_memory,
    update_memory,
)

ALLOWED_MEMORY_TYPES = {"preference", "trait", "event", "pattern"}

IMPORTANCE_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.70
DEDUP_LOOKBACK_LIMIT = 20

# 基础工具
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
    content = (item.get("content") or "").strip()
    if not content:
        return None

    return {
        "memory_type": _normalize_memory_type(item.get("memory_type")),
        "content": content,
        "importance": _clamp_score(item.get("importance", 0.5)),
        "confidence": _clamp_score(item.get("confidence", 0.5)),
    }


def _is_memory_score_passed(importance: float, confidence: float) -> bool:
    return (
        float(importance) >= IMPORTANCE_THRESHOLD
        and float(confidence) >= CONFIDENCE_THRESHOLD
    )

# 模型调用：提取候选记忆
def extract_memory_candidate(client, user_text: str, recent_messages: list[dict]) -> dict | None:
    system_prompt = """
    You are a conversational memory extractor for a long-term companion AI.
    你是一个长期陪伴型 AI 的“对话记忆提取器”。

    Your task is to decide whether the user's latest input contains information worth storing as long-term memory.
    你的任务是判断：用户这次输入中，是否包含值得纳入“长期记忆”的信息。

    You must be precise, conservative for ordinary chat, but decisive for explicit memory instructions.
    对于普通闲聊要保守，但对于“用户明确要求记住”的内容要果断。

    --------------------------------
    [What counts as long-term memory / 哪些内容值得长期记忆]

    Store information only if it belongs to one of these categories:
    仅当信息属于以下类别之一时，才值得长期记忆：

    1. Stable preferences
       稳定偏好
       - likes/dislikes about style, tone, habits, ways of interaction
       - 对风格、语气、习惯、互动方式的长期喜欢或讨厌

    2. Stable traits or patterns
       稳定特征或长期模式
       - recurring behavior patterns, communication style, long-term tendencies
       - 长期行为模式、沟通特点、持续倾向

    3. Important personal facts or important events
       重要个人事实或重要事件
       - facts that will matter in future interactions
       - 在未来互动中会持续有用的重要事实或事件

    4. Persistent needs or requirements
       持续性需求或要求
       - things the assistant should remember and follow in future replies
       - 以后回复时需要长期参考和遵守的要求

    5. Explicit user instruction to remember something
       用户明确要求你记住的内容
       - if the user explicitly says “remember this”, “please remember”, “don't forget”...”
       - 如果用户明确说“记住”“请记住”“别忘了”
       - this is a strong signal and should normally be stored
       - 这是强信号，通常应被记住

    --------------------------------
    [What should NOT be stored / 哪些内容不应记忆]

    Do NOT store:
    不要记忆以下内容：

    1. one-off small talk
       一次性的闲聊

    2. temporary emotional states with no lasting value
       没有长期价值的瞬时情绪

    3. vague, generic, repetitive, or low-value statements
       空泛、普通、重复、低价值的信息

    4. greetings or filler messages
       寒暄、铺垫、口头填充内容

    --------------------------------
    [Critical product rule / 关键产品规则]

    If the user explicitly asks you to remember something, treat that as a high-priority memory signal.
    如果用户明确要求你“记住某件事”，这本身就是高优先级记忆信号。

    For explicit remember-instructions:
    对于“明确要求记住”的输入：

    - should_store should usually be true
    - should_store 通常应为 true

    - importance should usually be high
    - importance 通常应较高

    - do NOT undervalue short but important facts
    - 不要因为句子短就低估它的重要性

    Examples include:
    例如：

    - “请记住我的名字叫 cappy”
    - “我叫 cappy”
    - “以后叫我 cappy”
    - “记住我不喜欢被打断”
    - “请记住我更喜欢你直接一点”

    For these cases, importance normally should NOT be below 0.75 unless the content is ambiguous or unusable.
    对于这些情况，除非内容本身含糊不清或不可用，否则 importance 一般不应低于 0.75。

    --------------------------------
    [How to write content / content 字段怎么写]

    The "content" field must be:
    content 字段必须满足：

    1. one sentence
       一句话

    2. clear, compressed, future-reusable
       清晰、压缩、可面向未来复用

    3. written as a memory statement, not as raw user quote
       写成“记忆陈述句”，不要简单复读原话

    Good:
    好的例子：
    - 用户希望助手记住其名字是...。
    - 用户不喜欢被打断。
    - 用户更偏好直接、简洁的回应方式。

    Bad:
    不好的例子：
    - 请记住我的名字叫...
    - 我不喜欢这个
    - 他说他想让我记住

    --------------------------------
    [How to choose memory_type / memory_type 选择规则]

    Use only one of:
    只能从以下选项中选一个：

    - preference
    - trait
    - event
    - pattern
    - null

    Guidance:
    规则：

    - preference: likes/dislikes, preferred ways of interaction
      preference：偏好、厌恶、互动方式偏好

    - trait: relatively stable personal characteristic
      trait：相对稳定的人格特征或个人特点

    - event: important fact, identity fact, important event, explicit named fact
      event：重要事实、身份信息、重要事件、明确事实（如名字）

    - pattern: recurring tendency or repeated long-term behavioral pattern
      pattern：反复出现的长期模式

    - null: use only when should_store is false
      null：仅在 should_store 为 false 时使用

    For names, identity-like facts,instructions, prefer "event".
    对于名字、身份事实 这类信息，优先使用 event。

    --------------------------------
    [Scoring rules / 打分规则]

    importance:
    - 0.0 ~ 0.4 = trivial / no long-term value
    - 0.4 ~ 0.6 = somewhat useful but not strong
    - 0.75 ~ 1.0 = clearly important long-term memory
    - 0.75 ~ 1.0 = 明确重要、应长期保留

    confidence:
    - score how confident you are in your own extraction
    - 表示你对提取结果本身的把握程度
    - use high confidence when the meaning is explicit
    - 当用户表达很明确时，confidence 应较高

    --------------------------------
    [Output format / 输出格式]

    You must output valid JSON only, with no extra text.
    你必须只输出合法 JSON，不能输出任何额外解释文字。

    {
      "should_store": true,
      "memory_type": "preference|trait|event|pattern|null",
      "content": "一句简洁明确的话",
      "importance": 0.0,
      "confidence": 0.0
    }
    """.strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_messages[-6:])
    messages.append({
        "role": "user",
        "content": f"请判断下面这句话是否值得纳入长期记忆，并按要求输出 JSON：\n\n{user_text}"
    })

    resp = client.chat.completions.create(
        model=MODEL_DECISION,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)

    if not data.get("should_store"):
        return None

    return _normalize_memory_item(data)


# 模型调用：会话级总结

def summarize_session_memories(
    client,
    session_messages: list[dict],
) -> list[dict]:
    if not session_messages:
        return []
    system_prompt = """
    You are a memory summarizer for a long-term memory system.
    你是一个长期记忆系统里的“记忆总结器”。

    Your task is to read a conversation segment and summarize the parts that may be worth keeping as long-term memory.
    你的任务是阅读一段对话，并总结其中那些可能值得保留为长期记忆的内容。

    Focus on what is worth remembering from this conversation, not on summarizing every line.
    重点是提炼“这段对话里有什么值得记住”，而不是逐句概括整段聊天。

    The output should feel like a distilled memory summary of the session.
    输出应当像这段会话沉淀下来的“记忆摘要”。

    Do not force a fixed number of memories.
    不要强行凑固定数量。

    If there is little worth remembering, output fewer items.
    如果值得记住的内容很少，就输出更少的条目。

    If there is nothing worth keeping as long-term memory, output an empty list.
    如果没有适合进入长期记忆的内容，就输出空列表。

    Prefer concise, reusable memory statements.
    优先输出简洁、可复用的记忆句子。

    Try to preserve things that may matter in future conversations, such as:
    尽量保留那些在未来对话中可能仍然有价值的内容，例如：

    - preferences or dislikes
    - 偏好或厌恶

    - stable tendencies or personal characteristics
    - 稳定倾向或个人特征

    - facts or events
    - 事实或事件

    - ongoing needs, boundaries, or expectations
    - 持续性的需求、边界或期待

    - things the user explicitly wanted remembered
    - 用户明确希望被记住的内容

    Do not keep filler, or details that only matter for the current turn.
    不要保留铺垫语、或只对当前轮次有用的细节。

    Each memory should be written as a clean memory statement, not as raw dialogue.
    每条记忆都应该写成干净的“记忆句子”，而不是原始对话复读。

    Use these memory types when appropriate:
    需要时可使用以下记忆类型：

    - preference
    - trait
    - event
    - pattern

    Output valid JSON only:

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
        model=MODEL_DECISION,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "以下是一段完整对话，请提取值得长期保存的记忆：\n\n"
                + json.dumps(session_messages, ensure_ascii=False)
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)
    raw_memories = data.get("memories", [])

    result = []
    for item in raw_memories:
        normalized = _normalize_memory_item(item)
        if normalized is not None:
            result.append(normalized)

    return result


# 模型调用：去重与合并决策
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
        model=MODEL_DECISION,
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
        kind = (action.get("action") or "").strip()
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

# 流程入口：每轮在线记忆
def process_memory(
    client,
    user_id: str,
    user_text: str,
    recent_messages: list[dict],
):
    """
    每轮调用一次：
    1. 提取候选记忆
    2. 与已有记忆去重/合并
    3. 根据阈值决定 insert / update / skip
    """

    print(f"[MEMORY] input -> {user_text}")

    candidate = extract_memory_candidate(
        client=client,
        user_text=user_text,
        recent_messages=recent_messages,
    )

    if not candidate:
        print("[MEMORY] no candidate")
        return

    print("[MEMORY CANDIDATE]", candidate)

    existing_memories = get_recent_memories_for_dedup(
        user_id,
        limit=DEDUP_LOOKBACK_LIMIT,
    )

    actions = resolve_memory_actions(
        client=client,
        candidates=[candidate],
        existing_memories=existing_memories,
    )

    if not actions:
        print("[MEMORY] no actions")
        return

    print("[MEMORY ACTIONS]", actions)

    for action in actions:
        kind = action.get("action")

        if kind == "skip":
            print(f"[MEMORY] skip -> candidate_index={action.get('candidate_index')}")
            continue

        content = (action.get("content") or "").strip()
        if not content:
            print("[MEMORY] empty content after resolve")
            continue

        memory_type = _normalize_memory_type(action.get("memory_type"))
        importance = _clamp_score(action.get("importance", 0.5))
        confidence = _clamp_score(action.get("confidence", 0.5))

        if not _is_memory_score_passed(importance, confidence):
            print(
                f"[MEMORY] below threshold -> "
                f"importance={importance}, confidence={confidence}, content={content}"
            )
            continue

        if kind == "insert":
            memory_id = save_memory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                importance=importance,
                confidence=confidence,
                source_message_id=None,
            )
            print(f"[MEMORY] insert #{memory_id} -> {content}")

        elif kind == "update_existing":
            existing_id = action.get("existing_memory_id")
            if existing_id:
                update_memory(
                    memory_id=int(existing_id),
                    content=content,
                    importance=importance,
                    confidence=confidence,
                )
                print(f"[MEMORY] update #{existing_id} -> {content}")
            else:
                print("[MEMORY] update_existing but missing existing_memory_id")