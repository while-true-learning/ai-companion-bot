import json
import re
from typing import Optional

from config import MODEL_SUMMARIZE, MODEL_DECIDER
from db import (
    save_emotion_event,
    get_recent_emotion_events,
    get_latest_emotion_event,
    get_emotion_summary_row,
    save_or_update_emotion_summary,
    emotion_summary_is_stale,
)


ALLOWED_PRIMARY = {
    "sad", "anxious", "angry", "fear",
    "calm", "positive", "mixed", "neutral", "unknown"
}
ALLOWED_SECONDARY = ALLOWED_PRIMARY | {"none"}


def default_emotion_result(reason: str = "") -> dict:
    return {
        "primary_emotion": "unknown",
        "secondary_emotion": "none",
        "fine_grained": [],
        "intensity": 0.0,
        "confidence": 0.0,
        "reason": reason
    }


def format_recent_rows(recent_rows: Optional[list], max_items: int = 6) -> str:
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


def format_pending_messages(pending_messages: Optional[list]) -> str:
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


def merge_pending_text(pending_messages: Optional[list]) -> str:
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


def extract_json_text(raw: str) -> str:
    raw = raw.strip()
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("未找到 JSON 对象")


def detect_current_emotion(
    client,
    pending_messages: list[dict],
    recent_rows: Optional[list] = None,
) -> dict:
    model_name = MODEL_DECIDER

    pending_text = merge_pending_text(pending_messages)
    if not pending_text:
        return default_emotion_result("当前轮 pending 为空")

    pending_context_text = format_pending_messages(pending_messages)
    history_text = format_recent_rows(recent_rows)

    system_prompt = """
你是一个情绪分析器。
你的任务是根据用户当前这一轮 pending 输入，并结合少量数据库历史对话，判断用户当前最主要的情绪状态。

判断时必须以“当前 pending 轮内容”为主，数据库历史只用于辅助，不能因为历史情绪而夸大当前情绪。

你必须只输出 JSON，不能输出任何额外文字。

输出格式必须严格为：
{
  "primary_emotion": "sad|anxious|angry|fear|calm|positive|mixed|neutral|unknown",
  "secondary_emotion": "sad|anxious|angry|fear|calm|positive|mixed|neutral|unknown|none",
  "fine_grained": ["..."],
  "intensity": 0.0,
  "confidence": 0.0,
  "reason": "一句简短中文解释"
}

规则：
1. primary_emotion 只能从固定选项中选一个。
2. secondary_emotion 没有就填 "none"。
3. fine_grained 最多 3 个，用更细的自然词。
4. intensity 范围 0 到 1，表示当前这次表达的情绪强度。
5. confidence 范围 0 到 1，表示判断可靠性。
6. 如果信息不足，要保守，输出 neutral 或 unknown。
7. 不要做医学诊断，不要输出抑郁症、躁狂症等临床结论。
8. 只分析当前状态，不给建议。
9. reason 必须简短。
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
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        return default_emotion_result(f"情绪识别调用失败: {e}")

    try:
        data = json.loads(extract_json_text(raw))
    except Exception:
        return default_emotion_result(f"模型输出不是合法 JSON: {raw[:120]}")

    primary = data.get("primary_emotion", "unknown")
    secondary = data.get("secondary_emotion", "none")
    fine_grained = data.get("fine_grained", [])
    intensity = data.get("intensity", 0.0)
    confidence = data.get("confidence", 0.0)
    reason = data.get("reason", "")

    if primary not in ALLOWED_PRIMARY:
        primary = "unknown"

    if secondary not in ALLOWED_SECONDARY:
        secondary = "none"

    if not isinstance(fine_grained, list):
        fine_grained = []
    fine_grained = [
        str(x).strip() for x in fine_grained
        if str(x).strip() and len(str(x).strip()) <= 30
    ][:3]

    try:
        intensity = float(intensity)
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(1.0, intensity))

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if not isinstance(reason, str):
        reason = ""
    reason = reason.strip()[:120]

    return {
        "primary_emotion": primary,
        "secondary_emotion": secondary,
        "fine_grained": fine_grained,
        "intensity": intensity,
        "confidence": confidence,
        "reason": reason
    }


def process_emotion(
    client,
    user_id: str,
    pending_messages: list[dict],
    recent_rows: Optional[list] = None,
) -> dict:
    result = detect_current_emotion(
        client=client,
        pending_messages=pending_messages,
        recent_rows=recent_rows,
    )
    return result

def build_emotion_summary(
    client,
    user_id: str,
    limit: int = 30,
) -> dict:
    model_name = MODEL_SUMMARIZE
    events = get_recent_emotion_events(user_id, limit=limit)

    if not events:
        summary = {
            "summary_text": "暂无足够的情绪记录。",
            "current_primary_emotion": "neutral",
            "possible_causes": [],
            "long_term": [],
            "short_term": [],
            "last_significant_emotion": {}
        }
        save_or_update_emotion_summary(
            user_id=user_id,
            summary_text=summary["summary_text"],
            current_primary_emotion=summary["current_primary_emotion"],
            possible_causes=summary["possible_causes"],
            long_term=summary["long_term"],
            short_term=summary["short_term"],
            last_significant_emotion=summary["last_significant_emotion"],
            source_event_count=0,
            source_last_event_id=0
        )
        return summary

    lines = []
    for item in reversed(events):
        fine_grained = item.get("fine_grained", [])
        fine_grained_text = ", ".join(fine_grained) if fine_grained else "无"
        lines.append(
            f"[{item['created_at']}] "
            f"primary={item['primary_emotion']}, "
            f"secondary={item['secondary_emotion']}, "
            f"intensity={item['intensity']}, "
            f"confidence={item['confidence']}, "
            f"reason={item['reason_summary']}, "
            f"fine_grained={fine_grained_text}, "
            f"source={item['source_text']}"
        )

    events_text = "\n".join(lines)

    system_prompt = """
你是一个情绪总结器。
你会读取最近若干条 emotion event 记录，输出一个高层总结。

你必须只输出 JSON，不能输出任何额外文字。

输出格式严格为：
{
  "summary_text": "一段简短中文总结",
  "current_primary_emotion": "sad|anxious|angry|fear|calm|positive|mixed|neutral|unknown",
  "possible_causes": [
    {"cause": "原因", "score": 0.0}
  ],
  "long_term": [
    {"topic": "主题", "emotion": "anxious", "score": 0.0, "reason_summary": "简短说明"}
  ],
  "short_term": [
    {"topic": "主题", "emotion": "positive", "score": 0.0, "reason_summary": "简短说明"}
  ],
  "last_significant_emotion": {
    "emotion": "sad",
    "created_at": "",
    "reason_summary": ""
  }
}

规则：
1. 这是高层总结，不要逐条复述。
2. long_term 表示更持续、重复或跨时间出现的主题。
3. short_term 表示更短时、瞬时、局部的波动。
4. possible_causes 按可能性从高到低排序，最多 5 条。
5. 所有 score 都在 0 到 1。
6. 如果证据不足，要保守。
7. 不要做医学诊断。
8. summary_text 要简短、自然、中文。
""".strip()

    user_prompt = f"""
以下是用户最近 {len(events)} 条情绪记录：
{events_text}
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
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(extract_json_text(raw))
    except Exception:
        latest = get_latest_emotion_event(user_id)
        data = {
            "summary_text": "近期有一定情绪波动，但总结结果暂不稳定。",
            "current_primary_emotion": latest["primary_emotion"] if latest else "neutral",
            "possible_causes": [],
            "long_term": [],
            "short_term": [],
            "last_significant_emotion": {}
        }

    current_primary_emotion = str(data.get("current_primary_emotion", "neutral")).strip().lower()
    if current_primary_emotion not in ALLOWED_PRIMARY:
        current_primary_emotion = "neutral"

    possible_causes = data.get("possible_causes", [])
    long_term = data.get("long_term", [])
    short_term = data.get("short_term", [])
    last_significant_emotion = data.get("last_significant_emotion", {})
    summary_text = str(data.get("summary_text", "")).strip()

    if not isinstance(possible_causes, list):
        possible_causes = []
    if not isinstance(long_term, list):
        long_term = []
    if not isinstance(short_term, list):
        short_term = []
    if not isinstance(last_significant_emotion, dict):
        last_significant_emotion = {}

    source_last_event_id = max(int(item["id"]) for item in events) if events else 0

    save_or_update_emotion_summary(
        user_id=user_id,
        summary_text=summary_text,
        current_primary_emotion=current_primary_emotion,
        possible_causes=possible_causes[:5],
        long_term=long_term[:5],
        short_term=short_term[:5],
        last_significant_emotion=last_significant_emotion,
        source_event_count=len(events),
        source_last_event_id=source_last_event_id
    )

    return {
        "summary_text": summary_text,
        "current_primary_emotion": current_primary_emotion,
        "possible_causes": possible_causes[:5],
        "long_term": long_term[:5],
        "short_term": short_term[:5],
        "last_significant_emotion": last_significant_emotion
    }


def get_emotion_summary(
    client,
    user_id: str,
    limit: int = 30,
    force_refresh: bool = False,
) -> dict:
    cached = get_emotion_summary_row(user_id)

    if force_refresh or cached is None or emotion_summary_is_stale(user_id):
        return build_emotion_summary(
            client=client,
            user_id=user_id,
            limit=limit,
        )

    return {
        "summary_text": cached.get("summary_text", ""),
        "current_primary_emotion": cached.get("current_primary_emotion", "neutral"),
        "possible_causes": cached.get("possible_causes", []),
        "long_term": cached.get("long_term", []),
        "short_term": cached.get("short_term", []),
        "last_significant_emotion": cached.get("last_significant_emotion", {})
    }