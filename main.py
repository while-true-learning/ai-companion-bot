from config import OPENAI_API_KEY, USER_ID, DECIDER_MODEL
from db import (
    init_db,
    save_message,
    get_recent_messages,
    rows_to_chat_messages,
    get_memories,
    get_recent_memories_for_dedup,
    save_memory,
    update_memory,
    get_last_summarized_message_id,
    set_last_summarized_message_id,
    get_messages_after_id,
)
from idle_manager import IdleManager
from ai_client import generate_reply, client, generate_nudge
from memory_extractor import summarize_session_memories, resolve_memory_actions
from reply_decider import should_reply_now

idle_manager = IdleManager(
    idle_seconds=15,
    force_reply_after_seconds=60,
    summary_after_seconds=5 * 60
)


def on_user_idle(user_id: str, pending_messages: list[dict], meta: dict):
    print(f"\n[IDLE] 用户 {user_id} 已静默 {idle_manager.idle_seconds} 秒")
    print("[PENDING MESSAGES]")
    for msg in pending_messages:
        print("-", msg["content"])

    try:
        recent_rows = get_recent_messages(user_id, limit=12)
        recent_history = rows_to_chat_messages(recent_rows)

        decision = should_reply_now(
            client=client,
            model_name=DECIDER_MODEL,
            pending_messages=pending_messages,
            recent_history=recent_history
        )

        print(f"\n[DECISION] {decision}")

        action = decision["action"]
        if action == "reply":
            reply = generate_reply(user_id)
            print(f"\nAI: {reply}")
            save_message(user_id, "assistant", reply)
            idle_manager.clear_pending(user_id)

        elif action == "nudge":
            reply = generate_nudge(user_id)
            print(f"\nAI: {reply}")
            save_message(user_id, "assistant", reply)
            idle_manager.clear_pending(user_id)

        else:
            print(f"\n[WAIT] 暂不回复，等待用户继续输入或 {idle_manager.force_reply_after_seconds} 秒强制回复")

    except Exception as e:
        print(f"\n[ERROR] 判断失败，默认直接回复: {e}")
        try:
            reply = generate_reply(user_id)
            print(f"\nAI: {reply}")
            save_message(user_id, "assistant", reply)
        finally:
            idle_manager.clear_pending(user_id)


def on_force_reply(user_id: str, pending_messages: list[dict], meta: dict):
    print(f"\n[FORCE REPLY] 用户 {user_id} 已达到 {idle_manager.force_reply_after_seconds} 秒等待上限")
    print("[PENDING MESSAGES]")
    for msg in pending_messages:
        print("-", msg["content"])

    try:
        reply = generate_reply(user_id)
        print(f"\nAI: {reply}")
        save_message(user_id, "assistant", reply)
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
            model_name=DECIDER_MODEL,
            session_messages=session_messages
        )

        if not candidates:
            last_message_id = rows[-1][0]
            set_last_summarized_message_id(user_id, last_message_id)
            print("[SUMMARY] AI 未提取到新的长期记忆")
            return

        existing_memories = get_recent_memories_for_dedup(user_id, limit=30)
        actions = resolve_memory_actions(
            client=client,
            model_name=DECIDER_MODEL,
            candidates=candidates,
            existing_memories=existing_memories
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
                    source_message_id=rows[-1][0]
                )
                print(f"[MEMORY] insert #{memory_id} -> {content}")

            elif kind == "update_existing":
                existing_memory_id = action.get("existing_memory_id")
                if existing_memory_id:
                    update_memory(
                        memory_id=int(existing_memory_id),
                        content=content,
                        importance=importance,
                        confidence=confidence
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
            on_user_idle,
            on_force_reply,
            on_summary_due
        )


if __name__ == "__main__":
    main()