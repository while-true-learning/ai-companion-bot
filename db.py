import json
import sqlite3
from datetime import datetime
from config import DB_PATH


ALLOWED_MEMORY_TYPES = {"preference", "trait", "event", "pattern"}


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
    importance: float,
    confidence: float
):
    content = content.strip()
    if not content:
        raise ValueError("memory content cannot be empty")

    importance = max(0.0, min(1.0, float(importance)))
    confidence = max(0.0, min(1.0, float(confidence)))

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        UPDATE memories
        SET content = ?, importance = ?, confidence = ?, updated_at = ?
        WHERE id = ?
        """, (content, importance, confidence, now_iso(), memory_id))
        conn.commit()
    finally:
        conn.close()


def get_memories(user_id: str, limit: int = 20, memory_type: str | None = None):
    conn = get_conn()
    try:
        cur = conn.cursor()

        if memory_type:
            cur.execute("""
            SELECT id, user_id, memory_type, content, importance, confidence,
                   source_message_id, created_at, updated_at
            FROM memories
            WHERE user_id = ? AND memory_type = ?
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """, (user_id, memory_type, limit))
        else:
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


def get_context_bundle(user_id: str, message_limit: int = 12, memory_limit: int = 10) -> dict:
    return {
        "recent_messages": get_recent_messages(user_id, limit=message_limit),
        "memories": get_memories(user_id, limit=memory_limit),
        "relationship_state": get_relationship_state(user_id)
    }