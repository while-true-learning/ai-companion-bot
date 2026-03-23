import json
from typing import Optional


def detect_current_emotion(client, model_name: str, user_text: str, recent_rows: Optional[list] = None) -> dict:
    """
    用模型判断用户当前情绪。

    参数:
        client: OpenAI client
        model_name: 模型名，比如 "gpt-5.4-mini"
        user_text: 当前用户输入
        recent_rows: 最近几条聊天记录，格式通常是:
            [
                (id, role, content, created_at),
                ...
            ]

    返回:
        {
            "primary_emotion": "sad|anxious|angry|fear|calm|positive|mixed|neutral|unknown",
            "secondary_emotion": "sad|anxious|angry|fear|calm|positive|mixed|neutral|unknown|none",
            "fine_grained": ["..."],
            "intensity": 0.0,
            "confidence": 0.0,
            "reason": "一句简短中文解释"
        }
    """

    context_text = ""
    if recent_rows:
        lines = []
        for row in recent_rows[-6:]:
            # 兼容 tuple: (id, role, content, created_at)
            if isinstance(row, (list, tuple)) and len(row) >= 4:
                _, role, content, created_at = row[:4]
                lines.append(f"[{created_at}] {role}: {content}")
        context_text = "\n".join(lines)

    system_prompt = """
你是一个情绪分析器。
你的任务是根据用户当前输入，并结合最近少量对话上下文，判断用户当前最主要的情绪状态。

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
3. fine_grained 最多 3 个，使用更自然细致的情绪词，例如：
   overwhelmed, lonely, frustrated, tired, numb, confused, relieved, hopeful, guilty, empty, ashamed
4. intensity 范围 0 到 1，表示情绪强度。
5. confidence 范围 0 到 1，表示判断可靠性。
6. 如果信息不足，要保守，输出 neutral 或 unknown。
7. 不要做医学诊断，不要输出抑郁症、躁狂症等临床结论。
8. 只分析“当前状态”，不要展开建议。
""".strip()

    user_prompt = f"""
最近对话上下文：
{context_text if context_text else "无"}

当前用户输入：
{user_text}
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

    except Exception as e:
        return {
            "primary_emotion": "unknown",
            "secondary_emotion": "none",
            "fine_grained": [],
            "intensity": 0.0,
            "confidence": 0.0,
            "reason": f"情绪识别调用失败: {e}"
        }

    # 尝试解析 JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "primary_emotion": "unknown",
            "secondary_emotion": "none",
            "fine_grained": [],
            "intensity": 0.0,
            "confidence": 0.0,
            "reason": f"模型输出不是合法 JSON: {raw[:120]}"
        }

    # 兜底校验
    allowed_primary = {
        "sad", "anxious", "angry", "fear",
        "calm", "positive", "mixed", "neutral", "unknown"
    }
    allowed_secondary = allowed_primary | {"none"}

    primary = data.get("primary_emotion", "unknown")
    secondary = data.get("secondary_emotion", "none")
    fine_grained = data.get("fine_grained", [])
    intensity = data.get("intensity", 0.0)
    confidence = data.get("confidence", 0.0)
    reason = data.get("reason", "")

    if primary not in allowed_primary:
        primary = "unknown"

    if secondary not in allowed_secondary:
        secondary = "none"

    if not isinstance(fine_grained, list):
        fine_grained = []
    fine_grained = [str(x) for x in fine_grained[:3]]

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

    return {
        "primary_emotion": primary,
        "secondary_emotion": secondary,
        "fine_grained": fine_grained,
        "intensity": intensity,
        "confidence": confidence,
        "reason": reason
    }