from config import OPENAI_API_KEY, USER_ID
from db import init_db, save_message, get_recent_messages, rows_to_chat_messages
from idle_manager import IdleManager
from ai_client import generate_reply, client, generate_nudge
from reply_decider import should_reply_now

DECIDER_MODEL = "gpt-5.4-nano"

idle_manager = IdleManager(
    idle_seconds=8,
    force_reply_after_seconds=40
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
            print("\n[WAIT] 暂不回复，等待用户继续输入或 40 秒强制回复")

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
            on_force_reply
        )


if __name__ == "__main__":
    main()