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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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

    # 互动事件表：记录为什么变动
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interaction_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        trigger_message_id INTEGER,
        event_type TEXT NOT NULL,
        payload_json TEXT,
        delta_familiarity REAL NOT NULL DEFAULT 0.0,
        delta_trust REAL NOT NULL DEFAULT 0.0,
        delta_affection REAL NOT NULL DEFAULT 0.0,
        delta_dependency REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(trigger_message_id) REFERENCES messages(id)
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_interaction_events_user_id_id
    ON interaction_events(user_id, id DESC)
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

    conn.commit()
    conn.close()


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


def get_recent_messages(user_id: str, limit: int = 12):
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


def rows_to_chat_messages(rows):
    messages = []
    for _id, role, content, created_at in rows:
        if role in {"user", "assistant", "system"}:
            messages.append({
                "role": role,
                "content": content
            })
    return messages


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


def save_interaction_event(
    user_id: str,
    event_type: str,
    trigger_message_id: int | None = None,
    payload: dict | None = None,
    delta_familiarity: float = 0.0,
    delta_trust: float = 0.0,
    delta_affection: float = 0.0,
    delta_dependency: float = 0.0
) -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO interaction_events (
            user_id,
            trigger_message_id,
            event_type,
            payload_json,
            delta_familiarity,
            delta_trust,
            delta_affection,
            delta_dependency,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            trigger_message_id,
            event_type,
            json.dumps(payload, ensure_ascii=False) if payload is not None else None,
            float(delta_familiarity),
            float(delta_trust),
            float(delta_affection),
            float(delta_dependency),
            now_iso()
        ))

        event_id = cur.lastrowid
        conn.commit()
        return event_id
    finally:
        conn.close()


def get_recent_interaction_events(user_id: str, limit: int = 20):
    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute("""
        SELECT id, user_id, trigger_message_id, event_type, payload_json,
               delta_familiarity, delta_trust, delta_affection, delta_dependency,
               created_at
        FROM interaction_events
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """, (user_id, limit))

        rows = cur.fetchall()

        result = []
        for row in rows:
            item = dict(row)
            if item["payload_json"]:
                try:
                    item["payload"] = json.loads(item["payload_json"])
                except json.JSONDecodeError:
                    item["payload"] = None
            else:
                item["payload"] = None
            result.append(item)

        return result
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


def emotion_summary_is_stale(user_id: str) -> bool:
    summary = get_emotion_summary_row(user_id)
    latest_event = get_latest_emotion_event(user_id)

    if summary is None:
        return True

    if latest_event is None:
        return False

    return int(latest_event["id"]) > int(summary.get("source_last_event_id", 0))


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