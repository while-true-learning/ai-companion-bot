import sqlite3
from datetime import datetime
from config import DB_PATH


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

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

    conn.commit()
    conn.close()


def save_message(user_id: str, role: str, content: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO messages (user_id, role, content, created_at)
    VALUES (?, ?, ?, ?)
    """, (
        user_id,
        role,
        content,
        datetime.now().isoformat(timespec="seconds")
    ))

    conn.commit()
    conn.close()


def get_recent_messages(user_id: str, limit: int = 12):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, role, content, created_at
    FROM messages
    WHERE user_id = ?
    ORDER BY id DESC
    LIMIT ?
    """, (user_id, limit))

    rows = cur.fetchall()
    conn.close()

    rows.reverse()
    return rows


def rows_to_chat_messages(rows):
    messages = []
    for _id, role, content, created_at in rows:
        if role in {"user", "assistant", "system"}:
            messages.append({
                "role": role,
                "content": content
            })
    return messages