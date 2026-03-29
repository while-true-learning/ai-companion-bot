# memory_extractor.py

import json

from config import MODEL_DECIDER, CONTEXT_LIMIT
from db import (
    get_recent_memories_for_dedup,
    save_memory,
    update_memory,
)

ALLOWED_MEMORY_TYPES = {"preference", "trait", "event", "pattern", "identity"}

ALLOWED_IDENTITY_KEYS = {
    "user_name",
    "user_age",
    "user_birthday",
    "user_birthdate",
    "user_pronouns",
    "user_gender_identity",
    "user_sexual_orientation",
    "user_relationship_status",
    "user_nationality",
    "user_ethnicity",
    "user_religion",
    "user_language",
    "user_location",
    "user_hometown",
    "user_education_status",
    "user_school",
    "user_major",
    "user_degree_goal",
    "user_job",
    "user_occupation",
    "user_career_stage",
    "user_student_status",
    "user_family_role",
    "ai_name",
    "ai_persona",
    "ai_role",
    "ai_relationship_frame",
    "ai_style",
}

IMPORTANCE_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.70
DEDUP_LOOKBACK_LIMIT = CONTEXT_LIMIT


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


def _normalize_identity_key(identity_key: str | None, memory_type: str | None = None) -> str | None:
    identity_key = str(identity_key or "").strip()
    if memory_type != "identity":
        return None
    if identity_key not in ALLOWED_IDENTITY_KEYS:
        return None
    return identity_key

def _normalize_memory_item(item: dict) -> dict | None:
    content = str(item.get("content", "") or "").strip()
    if not content:
        return None

    memory_type = _normalize_memory_type(item.get("memory_type"))

    normalized = {
        "memory_type": memory_type,
        "content": content,
        "importance": _clamp_score(item.get("importance", 0.5)),
        "confidence": _clamp_score(item.get("confidence", 0.5)),
    }

    if memory_type == "identity":
        identity_key = _normalize_identity_key(
            item.get("identity_key"),
            memory_type=memory_type,
        )
        if not identity_key:
            return None
        normalized["identity_key"] = identity_key

    return normalized


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


def _find_existing_identity_memory(existing_memories: list[dict], identity_key: str) -> dict | None:
    identity_key = str(identity_key or "").strip()
    if not identity_key:
        return None

    for mem in existing_memories:
        if (
            str(mem.get("memory_type", "")).strip() == "identity"
            and str(mem.get("identity_key", "")).strip() == identity_key
        ):
            return mem
    return None


def _build_identity_correction_event(old_content: str, new_content: str, identity_key: str) -> str:
    mapping = {
        "user_name": "用户姓名信息",
        "user_age": "用户年龄信息",
        "user_birthday": "用户生日信息",
        "user_birthdate": "用户出生日期信息",
        "user_pronouns": "用户代称信息",
        "user_gender_identity": "用户性别认同信息",
        "user_sexual_orientation": "用户性取向信息",
        "user_relationship_status": "用户关系状态信息",
        "user_nationality": "用户国籍信息",
        "user_ethnicity": "用户族裔/民族信息",
        "user_religion": "用户宗教信息",
        "user_language": "用户语言信息",
        "user_location": "用户所在地信息",
        "user_hometown": "用户家乡信息",
        "user_education_status": "用户学业身份信息",
        "user_school": "用户学校信息",
        "user_major": "用户专业信息",
        "user_degree_goal": "用户学位目标信息",
        "user_job": "用户工作信息",
        "user_occupation": "用户职业信息",
        "user_career_stage": "用户职业阶段信息",
        "user_student_status": "用户学生状态信息",
        "user_family_role": "用户家庭角色信息",
        "ai_name": "AI 名称信息",
        "ai_persona": "AI 人格设定信息",
        "ai_role": "AI 身份设定信息",
        "ai_relationship_frame": "AI 关系框架信息",
        "ai_style": "AI 风格设定信息",
        "other_identity": "一条身份信息",
    }
    label = mapping.get(identity_key, "一条身份信息")
    return f"{label}已从“{old_content}”修正为“{new_content}”。"


def _build_identity_merge_event(old_content: str, new_content: str, merged_content: str, identity_key: str) -> str:
    mapping = {
        "user_name": "用户姓名信息",
        "user_age": "用户年龄信息",
        "user_birthday": "用户生日信息",
        "user_birthdate": "用户出生日期信息",
        "user_pronouns": "用户代称信息",
        "user_gender_identity": "用户性别认同信息",
        "user_sexual_orientation": "用户性取向信息",
        "user_relationship_status": "用户关系状态信息",
        "user_nationality": "用户国籍信息",
        "user_ethnicity": "用户族裔/民族信息",
        "user_religion": "用户宗教信息",
        "user_language": "用户语言信息",
        "user_location": "用户所在地信息",
        "user_hometown": "用户家乡信息",
        "user_education_status": "用户学业身份信息",
        "user_school": "用户学校信息",
        "user_major": "用户专业信息",
        "user_degree_goal": "用户学位目标信息",
        "user_job": "用户工作信息",
        "user_occupation": "用户职业信息",
        "user_career_stage": "用户职业阶段信息",
        "user_student_status": "用户学生状态信息",
        "user_family_role": "用户家庭角色信息",
        "ai_name": "AI 名称信息",
        "ai_persona": "AI 人格设定信息",
        "ai_role": "AI 身份设定信息",
        "ai_relationship_frame": "AI 关系框架信息",
        "ai_style": "AI 风格设定信息",
        "other_identity": "一条身份信息",
    }
    label = mapping.get(identity_key, "一条身份信息")
    return f"{label}已补充合并：原为“{old_content}”，新增“{new_content}”，现整理为“{merged_content}”。"


def _merge_identity_contents_with_llm(
    client,
    identity_key: str,
    old_content: str,
    new_content: str,
) -> dict:
    system_prompt = """
你是 identity 记忆合并器。

你的任务是判断：一条新的 identity 信息，与旧的同类 identity 信息之间是什么关系。

你只能输出三种 mode：
1. same
- 两者语义相同，仅表达不同，不需要更新

2. merge
- 新信息是在补充旧信息，应合并成一条更完整的 identity

3. replace
- 新信息与旧信息冲突，旧信息应被新信息替换

规则：
- 如果新信息提供了额外细节，但不否定旧信息，优先 merge
- 如果新信息明显修正了旧信息，使用 replace
- merged_content 必须是一句数据库风格的完整陈述
- 不要拆成多句
- 不要输出解释文字，只输出 JSON

输出 JSON：
{
  "mode": "same|merge|replace",
  "merged_content": "合并后或替换后的完整句子",
  "reason": "简短原因"
}
""".strip()

    user_prompt = json.dumps(
        {
            "identity_key": identity_key,
            "old_content": old_content,
            "new_content": new_content,
        },
        ensure_ascii=False,
    )

    resp = client.chat.completions.create(
        model=MODEL_DECIDER,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)

    mode = str(data.get("mode", "") or "").strip()
    if mode not in {"same", "merge", "replace"}:
        mode = "replace"

    merged_content = str(data.get("merged_content", "") or "").strip()
    if not merged_content:
        merged_content = new_content.strip()

    return {
        "mode": mode,
        "merged_content": merged_content,
        "reason": str(data.get("reason", "") or "").strip(),
    }


def _build_identity_actions(
    client,
    candidate_index: int,
    candidate: dict,
    existing_memories: list[dict],
) -> list[dict]:
    candidate = _normalize_memory_item(candidate)
    if not candidate:
        return []

    identity_key = candidate.get("identity_key")
    if not identity_key:
        return []

    existing = _find_existing_identity_memory(existing_memories, identity_key)

    if existing is None:
        return [{
            "action": "insert",
            "candidate_index": candidate_index,
            "existing_memory_id": None,
            "memory_type": "identity",
            "identity_key": identity_key,
            "content": candidate["content"],
            "importance": candidate["importance"],
            "confidence": candidate["confidence"],
        }]

    old_content = str(existing.get("content", "") or "").strip()
    new_content = str(candidate.get("content", "") or "").strip()

    if old_content == new_content:
        return [{
            "action": "skip",
            "candidate_index": candidate_index,
            "existing_memory_id": int(existing["id"]),
        }]

    merge_result = _merge_identity_contents_with_llm(
        client=client,
        identity_key=identity_key,
        old_content=old_content,
        new_content=new_content,
    )

    mode = merge_result["mode"]
    merged_content = merge_result["merged_content"]

    if mode == "same":
        return [{
            "action": "skip",
            "candidate_index": candidate_index,
            "existing_memory_id": int(existing["id"]),
        }]

    if mode == "merge":
        return [
            {
                "action": "update_existing",
                "candidate_index": candidate_index,
                "existing_memory_id": int(existing["id"]),
                "memory_type": "identity",
                "identity_key": identity_key,
                "content": merged_content,
                "importance": max(float(existing.get("importance", 0.5)), candidate["importance"]),
                "confidence": max(float(existing.get("confidence", 0.5)), candidate["confidence"]),
            },
            {
                "action": "insert",
                "candidate_index": candidate_index,
                "existing_memory_id": None,
                "memory_type": "event",
                "content": _build_identity_merge_event(
                    old_content=old_content,
                    new_content=new_content,
                    merged_content=merged_content,
                    identity_key=identity_key,
                ),
                "importance": max(0.6, candidate["importance"]),
                "confidence": max(0.8, candidate["confidence"]),
            }
        ]

    return [
        {
            "action": "update_existing",
            "candidate_index": candidate_index,
            "existing_memory_id": int(existing["id"]),
            "memory_type": "identity",
            "identity_key": identity_key,
            "content": merged_content,
            "importance": candidate["importance"],
            "confidence": candidate["confidence"],
        },
        {
            "action": "insert",
            "candidate_index": candidate_index,
            "existing_memory_id": None,
            "memory_type": "event",
            "content": _build_identity_correction_event(
                old_content=old_content,
                new_content=merged_content,
                identity_key=identity_key,
            ),
            "importance": max(0.65, candidate["importance"]),
            "confidence": max(0.85, candidate["confidence"]),
        }
    ]

def extract_memories(
    client,
    pending_messages: list[dict],
) -> list[dict]:
    normalized_pending = _normalize_pending_messages(pending_messages)
    pending_text = _merge_messages_text(normalized_pending)

    if not pending_text:
        return []

    pending_context_text = _format_messages(normalized_pending, max_items=12)

    system_prompt = """
你是一个长期陪伴型 AI 的“多记忆提取器”。

你的任务：
读取当前这一段用户输入，并提取其中所有值得保留的记忆。
你必须自动判断每条记忆属于哪一类，并输出标准 JSON。

--------------------------------
[提取目标]

你应该提取这类信息：
1. preference
- 喜欢、不喜欢、偏好、口味、习惯性选择
- 例如：喜欢猫、讨厌太吵、偏爱冷色调

2. trait
- 相对稳定的人格特征、思维风格、性格倾向
- 例如：倾向理性思考、不擅长表达情绪、容易多想

3. event
- 一次性的经历、刚发生的事情、具体事件
- 例如：今天和朋友出去吃饭、昨天搬家了

4. pattern
- 重复出现的行为模式、长期倾向、经常性的状态
- 例如：总是熬夜、经常打游戏缓解压力、习惯把事情想得很结构化

5. identity
- 用户是谁，或者 AI 被如何定义
- 例如：名字、学校、专业、身份、关系框架、AI 名称、人设设定等

--------------------------------
[不要提取的内容]

以下内容通常不要提取：
- 纯寒暄
- 无信息量的短句
- 单纯语气词、感叹词
- 没有稳定价值的碎片噪音
- 不能复用的空泛表达

--------------------------------
[identity 规则]

如果内容涉及身份、稳定自我描述、基础背景信息，优先归类为 identity，不要归类成 event。

例如：
- 我叫 X
- 我在 X 工作
- 我学 X 专业
- 你是我的...
- 你的名字叫...

这些都应优先考虑 identity。

特别是姓名

通常应输出：
memory_type = "identity"
identity_key = "user_name"

--------------------------------
[identity_key 可选值]

如果 memory_type = "identity"，还必须输出 identity_key，值只能是：

- user_name
- user_age
- user_birthday
- user_birthdate
- user_pronouns
- user_gender_identity
- user_sexual_orientation
- user_relationship_status
- user_nationality
- user_ethnicity
- user_religion
- user_language
- user_location
- user_hometown
- user_education_status
- user_school
- user_major
- user_degree_goal
- user_job
- user_occupation
- user_career_stage
- user_student_status
- user_family_role
- ai_name
- ai_persona
- ai_role
- ai_relationship_frame
- ai_style

如果不是 identity，就不要输出 identity_key，或者输出 null。

--------------------------------
[输出要求]

- 输出 0~N 条 memories
- 每条 memory 必须原子化，只表达一条独立信息
- content 必须改写成“可存数据库”的记忆句子
- content 必须简洁、明确、可复用
- 不要照搬原话
- 不要解释
- 不要输出 JSON 以外的任何文字

--------------------------------
[评分要求]

importance:
这条信息本身有多值得长期保存

confidence:
你对提取和分类的把握有多高

都用 0~1 之间的小数

--------------------------------
[输出格式]

{
  "memories": [
    {
      "memory_type": "preference|trait|event|pattern|identity",
      "identity_key": "user_name|user_age|user_birthday|user_birthdate|user_pronouns|user_gender_identity|user_sexual_orientation|user_relationship_status|user_nationality|user_ethnicity|user_religion|user_language|user_location|user_hometown|user_education_status|user_school|user_major|user_degree_goal|user_job|user_occupation|user_career_stage|user_student_status|user_family_role|ai_name|ai_persona|ai_role|ai_relationship_frame|ai_style|null",
      "content": "数据库风格记忆句",
      "importance": 0.0,
      "confidence": 0.0
    }
  ]
}
""".strip()

    user_prompt = f"""
当前输入内容：
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

    results = []
    for item in data.get("memories", []):
        normalized = _normalize_memory_item(item)
        if normalized:
            results.append(normalized)

    return results

def resolve_memory_actions(
    client,
    candidates: list[dict],
    existing_memories: list[dict],
) -> list[dict]:
    if not candidates:
        return []

    actions: list[dict] = []

    for idx, candidate in enumerate(candidates):
        candidate = _normalize_memory_item(candidate)
        if not candidate:
            continue

        if candidate["memory_type"] == "identity":
            actions.extend(_build_identity_actions(
                client=client,
                candidate_index=idx,
                candidate=candidate,
                existing_memories=existing_memories,
            ))
            continue

        system_prompt = """
你是一个记忆去重与合并决策器。
你会看到“一个新记忆候选”和“已有记忆”。

你只能输出以下动作之一：
1. insert: 这是新的记忆，应插入
2. update_existing: 这是已有记忆的更完整版本，应更新旧记忆
3. skip: 这和已有记忆重复，跳过

要求：
- 如果语义基本相同，不要重复插入
- 如果只是旧记忆的更清晰或更完整版本，优先 update_existing
- 只有明确是新信息时才 insert
- existing_memory_id 只有在 update_existing 时填写
- 不要修改 memory_type
- 不处理 identity 的唯一逻辑，identity 已由代码层单独处理

输出 JSON：
{
  "action": "insert|update_existing|skip",
  "existing_memory_id": 12,
  "memory_type": "preference|trait|event|pattern",
  "content": "更新后或插入时要保存的内容",
  "importance": 0.0,
  "confidence": 0.0
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
                            "candidate": candidate,
                            "existing_memories": [
                                m for m in existing_memories
                                if str(m.get("memory_type", "")).strip() != "identity"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        data = json.loads(resp.choices[0].message.content)
        kind = str(data.get("action", "") or "").strip()

        if kind not in {"insert", "update_existing", "skip"}:
            continue

        item = {
            "action": kind,
            "candidate_index": idx,
            "existing_memory_id": data.get("existing_memory_id"),
        }

        if kind != "skip":
            normalized = _normalize_memory_item(data)
            if normalized is None:
                normalized = candidate

            if normalized["memory_type"] == "identity":
                normalized["memory_type"] = "pattern"
                normalized.pop("identity_key", None)

            item.update(normalized)

        actions.append(item)

    return actions


def apply_memory_actions(
    user_id: str,
    actions: list[dict],
    source_message_id: int | None = None,
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
        identity_key = _normalize_identity_key(
            action.get("identity_key"),
            memory_type=memory_type,
        )

        if kind == "insert":
            memory_id = save_memory(
                user_id=user_id,
                memory_type=memory_type,
                identity_key=identity_key,
                content=content,
                importance=importance,
                confidence=confidence,
                source_message_id=source_message_id,
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
                identity_key=identity_key,
            )
            print(f"{log_prefix} update #{existing_id} -> {content}")


def process_memory(
    client,
    user_id: str,
    pending_messages: list[dict],
) -> list[dict]:
    normalized_pending = _normalize_pending_messages(pending_messages)
    merged_text = _merge_messages_text(normalized_pending)

    print(f"[MEMORY] input -> {merged_text or '空'}")

    memory_items = extract_memories(
        client=client,
        pending_messages=normalized_pending,
    )

    if not memory_items:
        print("[MEMORY] no memory extracted")
        return []

    print("[MEMORY EXTRACTED]", memory_items)

    existing_memories = get_recent_memories_for_dedup(
        user_id=user_id,
        limit=DEDUP_LOOKBACK_LIMIT,
    )

    actions = resolve_memory_actions(
        client=client,
        candidates=memory_items,
        existing_memories=existing_memories,
    )

    print("[MEMORY ACTIONS]", actions)
    return actions