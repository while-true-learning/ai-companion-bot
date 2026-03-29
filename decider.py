import json
import re
from typing import Optional
from config import MODEL_DECIDER, CONTEXT_LIMIT


def _default_result(flag_name: str, reason: str = "") -> dict:
    return {
        flag_name: False,
        "confidence": 0.0,
        "reason": reason or "默认不调用"
    }

def _format_recent_rows(recent_rows: Optional[list], max_items: int = CONTEXT_LIMIT) -> str: #通过最近x条历史做出绝对
    if not recent_rows:
        return "无"

    lines = []
    for row in recent_rows[-max_items:]:
        try:
            if isinstance(row, dict):
                role = row.get("role", "unknown")
                content = row.get("content", "")
                created_at = row.get("created_at", "")
            elif isinstance(row, (list, tuple)) and len(row) >= 4:
                _, role, content, created_at = row[:4]
            else:
                continue

            lines.append(f"[{created_at}] {role}: {content}")
        except Exception:
            continue

    return "\n".join(lines) if lines else "无"


def _format_pending_messages(pending_messages: Optional[list]) -> str:
    if not pending_messages:
        return "无"

    lines = []
    for msg in pending_messages:
        try:
            role = str(msg.get("role", "user")).strip()
            content = str(msg.get("content", "") or "").strip()
            created_at = str(msg.get("created_at", "") or "").strip()

            if not content:
                continue

            if created_at:
                lines.append(f"[{created_at}] {role}: {content}")
            else:
                lines.append(f"{role}: {content}")
        except Exception:
            continue

    return "\n".join(lines) if lines else "无"


def _merge_pending_text(pending_messages: Optional[list]) -> str:
    if not pending_messages:
        return ""

    parts = []
    for msg in pending_messages:
        try:
            content = str(msg.get("content", "") or "").strip()
            if content:
                parts.append(content)
        except Exception:
            continue

    return " ".join(parts).strip()


def _extract_json_text(raw: str) -> str:
    raw = (raw or "").strip()

    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("未找到 JSON 对象")


def _run_binary_decider(
    client,
    *,
    pending_messages: list[dict],
    recent_rows: Optional[list],
    model_name: Optional[str],
    flag_name: str,
    target_name: str,
    rules_text: str,
    empty_reason: str = "当前轮 pending 为空",
    low_confidence_threshold: float = 0.35,
) -> dict:
    model_name = model_name or MODEL_DECIDER

    pending_text = _merge_pending_text(pending_messages)
    pending_context_text = _format_pending_messages(pending_messages)
    history_text = _format_recent_rows(recent_rows)

    if not pending_text:
        return _default_result(flag_name, empty_reason)

    system_prompt = f"""
你是一个对话前置决策器。
你的唯一任务是判断：这次回复用户之前，是否需要额外读取“{target_name}”。

你必须只输出 JSON，不能输出任何额外文字。

输出格式严格为：
{{
  "{flag_name}": true,
  "confidence": 0.0,
  "reason": "一句简短中文解释"
}}

判断原则：
{rules_text}
""".strip()

    user_prompt = f"""
数据库中的历史对话（不包括本轮）：
{history_text}

当前 pending 轮内容（本轮重点）：
{pending_context_text}

当前轮合并后的核心文本：
{pending_text}
""".strip()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        return _default_result(flag_name, f"调用失败: {e}")

    try:
        data = json.loads(_extract_json_text(raw))
    except Exception:
        return _default_result(flag_name, f"模型输出不是合法 JSON: {raw[:120]}")

    use_flag = bool(data.get(flag_name, False))

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "") or "").strip()[:120]

    if confidence < low_confidence_threshold:
        use_flag = False

    return {
        flag_name: use_flag,
        "confidence": confidence,
        "reason": reason or "无"
    }


EMOTION_RULES = """
1. 默认应当保守，默认 false，不要轻易调用情绪总结。
2. 只有在“当前 pending 轮内容本身不足以自然回复”，而且“近期情绪状态可能明显帮助理解和回应”时，才输出 true。
3. 以下情况更倾向于 true：
   - 用户在谈最近状态、持续状态、重复模式、情绪趋势
   - 用户表达模糊但带明显情绪负载，例如“还是这样”“我又开始了”“我不开心”
   - 用户在问自己最近怎么了、为什么总这样、是不是一直如此
   - 只看当前轮内容不够，结合近期情绪总结会明显改善回复质量
4. 以下情况更倾向于 false：
   - 当前轮上下文已经足够清楚
   - 普通闲聊、事实陈述、信息问答、代码/作业/翻译类问题
   - 单次具体事件本身已足够理解，不需要额外历史
   - 调用情绪总结不会显著改善回复
5. 不要因为用户出现单个情绪词就轻易输出 true。
6. 以当前 pending 轮为主，历史上下文只辅助判断。
7. reason 必须简短。
""".strip()


LONG_TERM_MEMORY_RULES = """
1. 如果用户在询问“你是否记得某个信息”（如偏好、身份、过去说过的话），倾向 true。
2. 如果用户引用过去内容，而当前 pending + recent history 无法解析，这意味着需要在数据库里查找 倾向 true。
3. 如果当前上下文已经足够回答，返回 false。
4. reason 简短。
""".strip()


def decide_use_emotion_summary(
    client,
    pending_messages: list[dict],
    recent_rows: Optional[list] = None,
    model_name: Optional[str] = None,
) -> dict:
    return _run_binary_decider(
        client,
        pending_messages=pending_messages,
        recent_rows=recent_rows,
        model_name=model_name,
        flag_name="use_emotion_summary",
        target_name="用户近期情绪总结",
        rules_text=EMOTION_RULES,
    )


def decide_use_long_term_memory(
    client,
    pending_messages: list[dict],
    recent_rows: Optional[list] = None,
    model_name: Optional[str] = None,
) -> dict:
    return _run_binary_decider(
        client,
        pending_messages=pending_messages,
        recent_rows=recent_rows,
        model_name=model_name,
        flag_name="use_long_term_memory",
        target_name="长期记忆",
        rules_text=LONG_TERM_MEMORY_RULES,
    )