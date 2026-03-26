import json
import sqlite3
from datetime import datetime
from config import DB_PATH


ALLOWED_MEMORY_TYPES = {"preference", "trait", "event", "pattern"}
ALLOWED_PRIMARY_EMOTIONS = {
    "sad", "anxious", "angry", "fear",
    "calm", "positive", "mixed", "neutral", "unknown"
}
ALLOWED_SECONDARY_EMOTIONS = ALLOWED_PRIMARY_EMOTIONS | {"none"}

# 获取当前时间（ISO格式）
# 用于所有数据库时间字段统一格式
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

# 获取数据库连接
# 统一设置 row_factory，方便后续转 dict
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# 初始化数据库结构（所有表 + 索引）
# 只在程序启动时调用一次
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # 原始消息表
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_messages_user_id_id
    ON messages(user_id, id)
    """)

    # 结构化记忆表
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        content TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.5,
        confidence REAL NOT NULL DEFAULT 0.5,
        source_message_id INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(source_message_id) REFERENCES messages(id)
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_memories_user_id_updated_at
    ON memories(user_id, updated_at DESC)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_memories_user_id_type
    ON memories(user_id, memory_type)
    """)

    #interaction_signals 这次互动是什么性质
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interaction_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        trigger_message_id INTEGER,
        openness REAL NOT NULL DEFAULT 0.0,
        warmth REAL NOT NULL DEFAULT 0.0,
        engagement REAL NOT NULL DEFAULT 0.0,
        reliance REAL NOT NULL DEFAULT 0.0,
        respect REAL NOT NULL DEFAULT 0.0,
        rejection REAL NOT NULL DEFAULT 0.0,
        confidence REAL NOT NULL DEFAULT 0.0,
        reason_summary TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY(trigger_message_id) REFERENCES messages(id)
    )
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_interaction_signals_user_id_id
    ON interaction_signals(user_id, id DESC)
    """)

    # 关系状态表：每个 user 一行
    cur.execute("""
    CREATE TABLE IF NOT EXISTS relationship_state (
        user_id TEXT PRIMARY KEY,
        familiarity REAL NOT NULL DEFAULT 0.0,
        trust REAL NOT NULL DEFAULT 0.0,
        affection REAL NOT NULL DEFAULT 0.0,
        dependency REAL NOT NULL DEFAULT 0.0,
        updated_at TEXT NOT NULL
    )
    """)

    # 记忆总结游标：记录上次总结做到哪条 message
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memory_summary_state (
        user_id TEXT PRIMARY KEY,
        last_summarized_message_id INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )
    """)

    # 情绪事件表：每轮 pending 聚合后的快照
    cur.execute("""
    CREATE TABLE IF NOT EXISTS emotion_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        trigger_message_id INTEGER,
        source_text TEXT NOT NULL,
        primary_emotion TEXT NOT NULL,
        secondary_emotion TEXT NOT NULL DEFAULT 'none',
        fine_grained_json TEXT NOT NULL DEFAULT '[]',
        intensity REAL NOT NULL DEFAULT 0.0,
        confidence REAL NOT NULL DEFAULT 0.0,
        reason_summary TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY(trigger_message_id) REFERENCES messages(id)
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_emotion_events_user_id_id
    ON emotion_events(user_id, id DESC)
    """)

    # 情绪总结缓存表：按需生成，可覆盖
    cur.execute("""
    CREATE TABLE IF NOT EXISTS emotion_summaries (
        user_id TEXT PRIMARY KEY,
        summary_text TEXT NOT NULL DEFAULT '',
        current_primary_emotion TEXT NOT NULL DEFAULT 'neutral',
        possible_causes_json TEXT NOT NULL DEFAULT '[]',
        long_term_json TEXT NOT NULL DEFAULT '[]',
        short_term_json TEXT NOT NULL DEFAULT '[]',
        last_significant_emotion_json TEXT NOT NULL DEFAULT '{}',
        source_event_count INTEGER NOT NULL DEFAULT 0,
        source_last_event_id INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )
    """)

    #会话总结
    cur.execute("""
    CREATE TABLE IF NOT EXISTS session_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        summary TEXT NOT NULL,
        topics TEXT NOT NULL,
        emotional_tone TEXT,
        importance REAL NOT NULL DEFAULT 0.5,
        start_message_id INTEGER,
        end_message_id INTEGER,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

# 保存一条对话消息（user / assistant / system）
# 返回 message_id（后续可用于关联 memory / emotion / signals）
def save_message(user_id: str, role: str, content: str) -> int:
    allowed_roles = {"user", "assistant", "system"}
    if role not in allowed_roles:
        raise ValueError(f"invalid role: {role}")

    content = content.strip()
    if not content:
        raise ValueError("message content cannot be empty")

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO messages (user_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """, (
            user_id,
            role,
            content,
            now_iso()
        ))

        message_id = cur.lastrowid
        conn.commit()
        return message_id
    finally:
        conn.close()

# 获取最近 N 条消息（用于上下文）
# 返回 list[(id, role, content, created_at)]
def get_recent_messages(user_id: str, limit: int = 200):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT id, role, content, created_at
        FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """, (user_id, limit))

        rows = cur.fetchall()
        rows = [tuple(row) for row in rows]
        rows.reverse()
        return rows
    finally:
        conn.close()

# 获取某条消息之后的新消息（用于增量处理）
def get_messages_after_id(user_id: str, last_message_id: int):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT id, role, content, created_at
        FROM messages
        WHERE user_id = ? AND id > ?
        ORDER BY id ASC
        """, (user_id, last_message_id))
        rows = cur.fetchall()
        return [tuple(row) for row in rows]
    finally:
        conn.close()

# 转换为 LLM 输入格式（chat messages）
def rows_to_chat_messages(rows):
    messages = []
    for _id, role, content, created_at in rows:
        if role in {"user", "assistant", "system"}:
            messages.append({
                "role": role,
                "content": content
            })
    return messages

# 保存一条结构化记忆
# memory_type: preference / trait / event / pattern
def save_memory(
    user_id: str,
    memory_type: str,
    content: str,
    importance: float = 0.5,
    confidence: float = 0.5,
    source_message_id: int | None = None
) -> int:
    if memory_type not in ALLOWED_MEMORY_TYPES:
        raise ValueError(f"invalid memory_type: {memory_type}")

    content = content.strip()
    if not content:
        raise ValueError("memory content cannot be empty")

    importance = max(0.0, min(1.0, float(importance)))
    confidence = max(0.0, min(1.0, float(confidence)))
    ts = now_iso()

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO memories (
            user_id, memory_type, content,
            importance, confidence, source_message_id,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, memory_type, content,
            importance, confidence, source_message_id,
            ts, ts
        ))
        memory_id = cur.lastrowid
        conn.commit()
        return memory_id
    finally:
        conn.close()

# 更新已有记忆（支持部分字段更新）
def update_memory(
    memory_id: int,
    content: str,
    importance: float | None = None,
    confidence: float | None = None
):
    content = content.strip()
    if not content:
        raise ValueError("memory content cannot be empty")

    conn = get_conn()
    try:
        cur = conn.cursor()

        if importance is None and confidence is None:
            cur.execute("""
            UPDATE memories
            SET content = ?, updated_at = ?
            WHERE id = ?
            """, (content, now_iso(), memory_id))
        else:
            if importance is None:
                cur.execute("SELECT importance FROM memories WHERE id = ?", (memory_id,))
                row = cur.fetchone()
                importance = float(row["importance"]) if row else 0.5
            if confidence is None:
                cur.execute("SELECT confidence FROM memories WHERE id = ?", (memory_id,))
                row = cur.fetchone()
                confidence = float(row["confidence"]) if row else 0.5

            importance = max(0.0, min(1.0, float(importance)))
            confidence = max(0.0, min(1.0, float(confidence)))

            cur.execute("""
            UPDATE memories
            SET content = ?, importance = ?, confidence = ?, updated_at = ?
            WHERE id = ?
            """, (content, importance, confidence, now_iso(), memory_id))

        conn.commit()
    finally:
        conn.close()

# 获取重要记忆（用于 prompt）
def get_memories(user_id: str, limit: int = 10):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT id, user_id, memory_type, content, importance, confidence,
               source_message_id, created_at, updated_at
        FROM memories
        WHERE user_id = ?
        ORDER BY importance DESC, updated_at DESC
        LIMIT ?
        """, (user_id, limit))

        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

# 获取最近记忆（用于去重 / merge）
def get_recent_memories_for_dedup(user_id: str, limit: int = 30):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT id, user_id, memory_type, content, importance, confidence,
               source_message_id, created_at, updated_at
        FROM memories
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT ?
        """, (user_id, limit))
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

# 获取用户当前关系状态
def get_relationship_state(user_id: str) -> dict:
    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute("""
        SELECT user_id, familiarity, trust, affection, dependency, updated_at
        FROM relationship_state
        WHERE user_id = ?
        """, (user_id,))
        row = cur.fetchone()

        if row is None:
            state = {
                "user_id": user_id,
                "familiarity": 0.0,
                "trust": 0.0,
                "affection": 0.0,
                "dependency": 0.0,
                "updated_at": now_iso()
            }
            cur.execute("""
            INSERT INTO relationship_state (
                user_id, familiarity, trust, affection, dependency, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                state["user_id"],
                state["familiarity"],
                state["trust"],
                state["affection"],
                state["dependency"],
                state["updated_at"]
            ))
            conn.commit()
        else:
            state = dict(row)

        return state
    finally:
        conn.close()

# 插入或更新关系状态
def upsert_relationship_state(
    user_id: str,
    familiarity: float,
    trust: float,
    affection: float,
    dependency: float
):
    familiarity = max(0.0, min(100.0, float(familiarity)))
    trust = max(0.0, min(100.0, float(trust)))
    affection = max(0.0, min(100.0, float(affection)))
    dependency = max(0.0, min(100.0, float(dependency)))

    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO relationship_state (
            user_id, familiarity, trust, affection, dependency, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            familiarity = excluded.familiarity,
            trust = excluded.trust,
            affection = excluded.affection,
            dependency = excluded.dependency,
            updated_at = excluded.updated_at
        """, (
            user_id,
            familiarity,
            trust,
            affection,
            dependency,
            now_iso()
        ))

        conn.commit()
    finally:
        conn.close()


def get_last_summarized_message_id(user_id: str) -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT last_summarized_message_id
        FROM memory_summary_state
        WHERE user_id = ?
        """, (user_id,))
        row = cur.fetchone()

        if row is None:
            cur.execute("""
            INSERT INTO memory_summary_state (user_id, last_summarized_message_id, updated_at)
            VALUES (?, ?, ?)
            """, (user_id, 0, now_iso()))
            conn.commit()
            return 0

        return int(row["last_summarized_message_id"])
    finally:
        conn.close()


def set_last_summarized_message_id(user_id: str, message_id: int):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO memory_summary_state (user_id, last_summarized_message_id, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            last_summarized_message_id = excluded.last_summarized_message_id,
            updated_at = excluded.updated_at
        """, (user_id, int(message_id), now_iso()))
        conn.commit()
    finally:
        conn.close()


def save_emotion_event(
    user_id: str,
    source_text: str,
    primary_emotion: str,
    secondary_emotion: str = "none",
    fine_grained: list[str] | None = None,
    intensity: float = 0.0,
    confidence: float = 0.0,
    reason_summary: str = "",
    trigger_message_id: int | None = None
) -> int:
    source_text = (source_text or "").strip()
    if not source_text:
        raise ValueError("emotion source_text cannot be empty")

    primary_emotion = str(primary_emotion or "unknown").strip().lower()
    secondary_emotion = str(secondary_emotion or "none").strip().lower()

    if primary_emotion not in ALLOWED_PRIMARY_EMOTIONS:
        primary_emotion = "unknown"

    if secondary_emotion not in ALLOWED_SECONDARY_EMOTIONS:
        secondary_emotion = "none"

    if not isinstance(fine_grained, list):
        fine_grained = []

    cleaned_fine_grained = []
    for item in fine_grained[:3]:
        s = str(item).strip()
        if s:
            cleaned_fine_grained.append(s[:30])

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

    reason_summary = str(reason_summary or "").strip()[:120]

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO emotion_events (
            user_id,
            trigger_message_id,
            source_text,
            primary_emotion,
            secondary_emotion,
            fine_grained_json,
            intensity,
            confidence,
            reason_summary,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            trigger_message_id,
            source_text,
            primary_emotion,
            secondary_emotion,
            json.dumps(cleaned_fine_grained, ensure_ascii=False),
            intensity,
            confidence,
            reason_summary,
            now_iso()
        ))
        emotion_event_id = cur.lastrowid
        conn.commit()
        return emotion_event_id
    finally:
        conn.close()


def get_recent_emotion_events(user_id: str, limit: int = 30):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT id, user_id, trigger_message_id, source_text,
               primary_emotion, secondary_emotion, fine_grained_json,
               intensity, confidence, reason_summary, created_at
        FROM emotion_events
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """, (user_id, limit))

        rows = cur.fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["fine_grained"] = json.loads(item["fine_grained_json"] or "[]")
            except json.JSONDecodeError:
                item["fine_grained"] = []
            result.append(item)
        return result
    finally:
        conn.close()


def get_latest_emotion_event(user_id: str) -> dict | None:
    rows = get_recent_emotion_events(user_id, limit=1)
    return rows[0] if rows else None


def save_or_update_emotion_summary(
    user_id: str,
    summary_text: str,
    current_primary_emotion: str = "neutral",
    possible_causes: list[dict] | None = None,
    long_term: list[dict] | None = None,
    short_term: list[dict] | None = None,
    last_significant_emotion: dict | None = None,
    source_event_count: int = 0,
    source_last_event_id: int = 0
):
    current_primary_emotion = str(current_primary_emotion or "neutral").strip().lower()
    if current_primary_emotion not in ALLOWED_PRIMARY_EMOTIONS:
        current_primary_emotion = "neutral"

    summary_text = str(summary_text or "").strip()
    possible_causes = possible_causes if isinstance(possible_causes, list) else []
    long_term = long_term if isinstance(long_term, list) else []
    short_term = short_term if isinstance(short_term, list) else []
    last_significant_emotion = last_significant_emotion if isinstance(last_significant_emotion, dict) else {}

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO emotion_summaries (
            user_id,
            summary_text,
            current_primary_emotion,
            possible_causes_json,
            long_term_json,
            short_term_json,
            last_significant_emotion_json,
            source_event_count,
            source_last_event_id,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            summary_text = excluded.summary_text,
            current_primary_emotion = excluded.current_primary_emotion,
            possible_causes_json = excluded.possible_causes_json,
            long_term_json = excluded.long_term_json,
            short_term_json = excluded.short_term_json,
            last_significant_emotion_json = excluded.last_significant_emotion_json,
            source_event_count = excluded.source_event_count,
            source_last_event_id = excluded.source_last_event_id,
            updated_at = excluded.updated_at
        """, (
            user_id,
            summary_text,
            current_primary_emotion,
            json.dumps(possible_causes, ensure_ascii=False),
            json.dumps(long_term, ensure_ascii=False),
            json.dumps(short_term, ensure_ascii=False),
            json.dumps(last_significant_emotion, ensure_ascii=False),
            int(source_event_count),
            int(source_last_event_id),
            now_iso()
        ))
        conn.commit()
    finally:
        conn.close()


def get_emotion_summary_row(user_id: str) -> dict | None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT user_id, summary_text, current_primary_emotion,
               possible_causes_json, long_term_json, short_term_json,
               last_significant_emotion_json, source_event_count,
               source_last_event_id, updated_at
        FROM emotion_summaries
        WHERE user_id = ?
        """, (user_id,))
        row = cur.fetchone()

        if row is None:
            return None

        item = dict(row)
        try:
            item["possible_causes"] = json.loads(item["possible_causes_json"] or "[]")
        except json.JSONDecodeError:
            item["possible_causes"] = []

        try:
            item["long_term"] = json.loads(item["long_term_json"] or "[]")
        except json.JSONDecodeError:
            item["long_term"] = []

        try:
            item["short_term"] = json.loads(item["short_term_json"] or "[]")
        except json.JSONDecodeError:
            item["short_term"] = []

        try:
            item["last_significant_emotion"] = json.loads(item["last_significant_emotion_json"] or "{}")
        except json.JSONDecodeError:
            item["last_significant_emotion"] = {}

        return item
    finally:
        conn.close()

# 判断情绪总结是否过期
def emotion_summary_is_stale(user_id: str) -> bool:
    summary = get_emotion_summary_row(user_id)
    latest_event = get_latest_emotion_event(user_id)

    if summary is None:
        return True

    if latest_event is None:
        return False

    return int(latest_event["id"]) > int(summary.get("source_last_event_id", 0))

# 工具函数：限制 float 在范围内
def _clamp_float(value, min_value: float, max_value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))

# 保存一次互动信号（核心输入层）
def save_interaction_signal(
    user_id: str,
    trigger_message_id: int | None = None,
    openness: float = 0.0,      # openness: 用户是否开放表达
    warmth: float = 0.0,        # warmth: 是否有情感温度
    engagement: float = 0.0,    # engagement: 是否持续互动
    reliance: float = 0.0,      # reliance: 是否依赖AI
    respect: float = 0.0,       # respect: 是否尊重AI
    rejection: float = 0.0,     # rejection: 是否拒绝/疏远
    confidence: float = 0.0,
    reason_summary: str = ""
) -> int:
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")

    openness = _clamp_float(openness, 0.0, 1.0)
    warmth = _clamp_float(warmth, 0.0, 1.0)
    engagement = _clamp_float(engagement, 0.0, 1.0)
    reliance = _clamp_float(reliance, 0.0, 1.0)
    respect = _clamp_float(respect, 0.0, 1.0)
    rejection = _clamp_float(rejection, 0.0, 1.0)
    confidence = _clamp_float(confidence, 0.0, 1.0)

    if trigger_message_id is not None:
        trigger_message_id = int(trigger_message_id)

    reason_summary = str(reason_summary or "").strip()[:120]

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO interaction_signals (
            user_id,
            trigger_message_id,
            openness,
            warmth,
            engagement,
            reliance,
            respect,
            rejection,
            confidence,
            reason_summary,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            trigger_message_id,
            openness,
            warmth,
            engagement,
            reliance,
            respect,
            rejection,
            confidence,
            reason_summary,
            now_iso()
        ))
        signal_id = cur.lastrowid
        conn.commit()
        return signal_id
    finally:
        conn.close()

# 获取最近20条互动信号
def get_recent_interaction_signals(user_id: str, limit: int = 20):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT id, user_id, trigger_message_id,
               openness, warmth, engagement,
               reliance, respect, rejection,
               confidence, reason_summary, created_at
        FROM interaction_signals
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """, (user_id, limit))

        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

# 获取最新一条互动信号
def get_latest_interaction_signal(user_id: str) -> dict | None:
    rows = get_recent_interaction_signals(user_id, limit=1)
    return rows[0] if rows else None

def save_session_summary(
    user_id: str,
    summary: str,
    topics: list[str] | None = None,
    emotional_tone: str = "",
    importance: float = 0.5,
    start_message_id: int | None = None,
    end_message_id: int | None = None,
) -> int:
    topics = topics or []
    importance = max(0.0, min(1.0, float(importance)))
    ts = now_iso()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO session_summaries (
            user_id,
            summary,
            topics,
            emotional_tone,
            importance,
            start_message_id,
            end_message_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            summary.strip(),
            json.dumps(topics, ensure_ascii=False),
            emotional_tone.strip(),
            importance,
            start_message_id,
            end_message_id,
            ts,
        ),
    )

    conn.commit()
    summary_id = cur.lastrowid
    conn.close()
    return summary_id


def get_context_bundle(
    user_id: str,
    message_limit: int = 12,
    memory_limit: int = 10,
    emotion_limit: int = 10
) -> dict:
    return {
        "recent_messages": get_recent_messages(user_id, limit=message_limit),
        "memories": get_memories(user_id, limit=memory_limit),
        "relationship_state": get_relationship_state(user_id),
        "recent_emotions": get_recent_emotion_events(user_id, limit=emotion_limit)
    }