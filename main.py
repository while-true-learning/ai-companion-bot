from config import OPENAI_API_KEY, USER_ID
from emotion import process_emotion, get_emotion_summary
from decider import decide_use_emotion_summary
from db import (
    init_db,
    save_message,
    get_recent_memories_for_dedup,
    save_memory,
    update_memory,
    get_last_summarized_message_id,
    set_last_summarized_message_id,
    get_messages_after_id,
    rows_to_chat_messages,
    get_recent_messages,
)
from idle_manager import IdleManager
from ai_client import client, generate_reply
from memory_extractor import (
    summarize_session_memories,
    resolve_memory_actions,
    process_memory,
)


idle_manager = IdleManager(
    force_reply_after_seconds=15,
    summary_after_seconds=5 * 60,
)


def _send_assistant_reply(user_id: str, reply: str):
    print(f"\nAI: {reply}")
    save_message(user_id, "assistant", reply)

def merge_pending_messages(pending_messages: list[dict]) -> str:
    return " ".join(
        (msg.get("content", "") or "").strip()
        for msg in pending_messages
    ).strip()

def on_force_reply(user_id: str, pending_messages: list[dict], meta: dict):
    print(f"\n[FORCE REPLY] 用户 {user_id} 已达到 {idle_manager.force_reply_after_seconds} 秒等待上限")
    print("[PENDING MESSAGES]")
    for msg in pending_messages:
        print("-", msg["content"])

    try:
        # 1. 短期记忆（recent history）
        recent_rows = get_recent_messages(user_id, limit=12)
        recent_history = rows_to_chat_messages(recent_rows)

        # 2. 合并用户输入
        merged_user_text = merge_pending_messages(pending_messages)

        # 3. 情绪记录（每轮一次）
        if merged_user_text:
            emotion_result = process_emotion(
                client=client,
                user_id=user_id,
                user_text=merged_user_text,
                recent_rows=recent_rows,
                trigger_message_id=recent_rows[-1][0] if recent_rows else None,
            )
            print(f"[EMOTION] {emotion_result}")

        # 4. 记忆提取并记录（每轮一次）
        if merged_user_text:
            process_memory(
                client=client,
                user_id=user_id,
                user_text=merged_user_text,
                recent_messages=recent_history,
            )
        # 5. 判断是否需要更复杂的情绪
        emotion_summary = None
        if merged_user_text:
            decision = decide_use_emotion_summary(
                client=client,
                user_text=merged_user_text,
                recent_rows=recent_rows,
            )
            print(f"[EMOTION SUMMARY DECIDER] {decision}")

            if decision["use_emotion_summary"]:
                emotion_summary = get_emotion_summary(
                    client=client,
                    user_id=user_id,
                    limit=30,
                    force_refresh=False,
                )
                print(f"[EMOTION SUMMARY] loaded -> {emotion_summary.get('summary_text', '')}")

        # 6. 生成回复
        reply = generate_reply(user_id, emotion_summary=emotion_summary)

        # 7. 输出 + 入库
        _send_assistant_reply(user_id, reply)

    except Exception as e:
        print(f"\n[ERROR] 强制回复失败: {e}")

    finally:
        idle_manager.clear_pending(user_id)


def on_summary_due(user_id: str, pending_messages: list[dict], meta: dict):
    print(f"\n[SUMMARY] 用户 {user_id} 已静默 {idle_manager.summary_after_seconds} 秒，开始会话级记忆总结")

    try:
        last_id = get_last_summarized_message_id(user_id)
        rows = get_messages_after_id(user_id, last_id)

        if not rows:
            print("[SUMMARY] 没有新的消息需要总结")
            return

        session_messages = rows_to_chat_messages(rows)
        if not session_messages:
            last_message_id = rows[-1][0]
            set_last_summarized_message_id(user_id, last_message_id)
            print("[SUMMARY] 没有可用于总结的 chat messages")
            return

        candidates = summarize_session_memories(
            client=client,
            session_messages=session_messages,
        )

        if not candidates:
            last_message_id = rows[-1][0]
            set_last_summarized_message_id(user_id, last_message_id)
            print("[SUMMARY] AI 未提取到新的长期记忆")
            return

        existing_memories = get_recent_memories_for_dedup(user_id, limit=30)
        actions = resolve_memory_actions(
            client=client,
            candidates=candidates,
            existing_memories=existing_memories,
        )

        for action in actions:
            kind = action.get("action")

            if kind == "skip":
                print(f"[MEMORY] skip -> candidate_index={action.get('candidate_index')}")
                continue

            content = (action.get("content") or "").strip()
            if not content:
                continue

            memory_type = action.get("memory_type") or "pattern"
            importance = float(action.get("importance", 0.5))
            confidence = float(action.get("confidence", 0.5))

            # 入库阈值，避免垃圾记忆
            if importance < 0.55 or confidence < 0.60:
                print(f"[MEMORY] below threshold -> {content}")
                continue

            if kind == "insert":
                memory_id = save_memory(
                    user_id=user_id,
                    memory_type=memory_type,
                    content=content,
                    importance=importance,
                    confidence=confidence,
                    source_message_id=rows[-1][0],
                )
                print(f"[MEMORY] insert #{memory_id} -> {content}")

            elif kind == "update_existing":
                existing_memory_id = action.get("existing_memory_id")
                if existing_memory_id:
                    update_memory(
                        memory_id=int(existing_memory_id),
                        content=content,
                        importance=importance,
                        confidence=confidence,
                    )
                    print(f"[MEMORY] update #{existing_memory_id} -> {content}")

        last_message_id = rows[-1][0]
        set_last_summarized_message_id(user_id, last_message_id)
        print(f"[SUMMARY] 完成，总结边界更新到 message_id={last_message_id}")

    except Exception as e:
        print(f"\n[ERROR] 会话总结失败: {e}")


def main():
    init_db()

    if not OPENAI_API_KEY:
        print("错误：没有检测到 OPENAI_API_KEY 环境变量。")
        return

    print("AI Companion started. 输入 exit 退出。")

    while True:
        user_text = input("\nYou: ").strip()

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            idle_manager.cancel_timer(USER_ID)
            print("Bye.")
            break
        save_message(USER_ID, "user", user_text)

        idle_manager.add_user_message(
            USER_ID,
            user_text,
            on_force_reply,
            on_summary_due,
        )


if __name__ == "__main__":
    main()