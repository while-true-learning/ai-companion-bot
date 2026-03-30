from datetime import datetime
from config import OPENAI_API_KEY, USER_ID ,CONTEXT_LIMIT
from emotion import process_emotion, get_emotion_summary
from decider import decide_use_emotion_summary, decide_use_long_term_memory
from relation import process_interaction_signal
from db import (
    init_db,
    save_message,
    get_last_summarized_message_id,
    set_last_summarized_message_id,
    get_messages_after_id,
    save_interaction_signal,
    save_emotion_event,
    get_recent_messages,
    save_session_summary,
    get_memories,
    save_relationship_update
)
from idle_manager import IdleManager
from ai_client import client, generate_reply
from summarize import summarize_conversation_session
from memory_extractor import (process_memory,apply_memory_actions)


idle_manager = IdleManager(
    force_reply_after_seconds=15,
    summary_after_seconds=5*60,
)


# 输出 assistant 回复，并写入数据库
def _send_assistant_reply(user_id: str, reply: str):
    print(f"\nAI: {reply}")
    save_message(
        user_id,
        "assistant",
        reply,
        created_at=datetime.now().isoformat(timespec="seconds")
    )

# 把本轮 pending 用户消息拼成一段文本
# 用于 emotion / memory 这类按“本轮整体内容”分析的模块
def merge_pending_messages(pending_messages: list[dict]) -> dict:
    contents = []
    times = []

    for msg in pending_messages:
        content = (msg.get("content", "") or "").strip()
        created_at = msg.get("created_at")

        if content:
            contents.append(content)

        if created_at:
            try:
                times.append(datetime.fromisoformat(created_at))
            except Exception:
                pass

    merged_text = " ".join(contents).strip()

    if times:
        duration = (max(times) - min(times)).total_seconds()
        last_gap = (datetime.now() - max(times)).total_seconds()
    else:
        duration = 0
        last_gap = 0

    return {
        "text": merged_text,
        "duration": duration,
        "last_gap": last_gap,
        "first_created_at": times[0].isoformat() if times else None,
        "last_created_at": times[-1].isoformat() if times else None,
    }

# 将本轮 pending 用户消息逐条写入数据库
# 返回最后一条 user message 的 message_id；如果没有则返回 None
def save_pending_user_messages(user_id: str, pending_messages: list[dict]) -> int | None:
    last_message_id = None

    for msg in pending_messages:
        role = str(msg.get("role", "user")).strip()
        content = (msg.get("content", "") or "").strip()
        created_at = msg.get("created_at")

        if role != "user" or not content:
            continue

        last_message_id = save_message(
            user_id=user_id,
            role="user",
            content=content,
            created_at=created_at,
        )

    return last_message_id

def on_force_reply(user_id: str, pending_messages: list[dict], meta: dict):
    print(f"\n[FORCE REPLY] 用户 {user_id} 已达到 {idle_manager.force_reply_after_seconds} 秒等待上限")
    print("[PENDING MESSAGES]")
    for msg in pending_messages:
        print(f"- [{msg.get('created_at')}] {msg.get('content')}")

    try:
        merged = merge_pending_messages(pending_messages)

        merged_user_text = merged["text"]
        input_duration = merged["duration"]
        last_gap = merged["last_gap"]
        now = datetime.now()
        current_time_text = now.strftime("%Y-%m-%d %H:%M:%S")
        if not merged_user_text:
            return

        recent_rows = get_recent_messages(user_id, limit=CONTEXT_LIMIT)
        print(f"[MERGED TEXT] {merged_user_text}")
        # 1. 先分析，不写库
        relation_result = process_interaction_signal(
            client=client,
            user_id=user_id,
            pending_messages=pending_messages,
            input_duration=input_duration,
            last_gap=last_gap,
        )
        print(f"[INTERACTION SIGNAL] {relation_result['signal']}")
        print(f"[RELATIONSHIP UPDATE] {relation_result.get('relationship_update')}")
        print(f"[RELATIONSHIP STATE] {relation_result['relationship_state']}")

        emotion_result = process_emotion(
            client=client,
            user_id=user_id,
            pending_messages=pending_messages,
            recent_rows=recent_rows,
            input_duration=input_duration,
        )
        print(f"[EMOTION] {emotion_result}")

        memory_actions = process_memory(
            client=client,
            user_id=user_id,
            pending_messages=pending_messages,
        )
        #print(f"[MEMORY ACTIONS] {memory_actions}")

        emotion_summary = None
        long_term_memories = None

        decision = decide_use_emotion_summary(
            client=client,
            pending_messages=pending_messages,
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

        memory_decision = decide_use_long_term_memory(
            client=client,
            pending_messages=pending_messages,
            recent_rows=recent_rows,
        )
        print(f"[LONG TERM MEMORY DECIDER] {memory_decision}")

        if memory_decision["use_long_term_memory"]:
            long_term_memories = get_memories(user_id, limit=CONTEXT_LIMIT)
            print(f"[LONG TERM MEMORIES] loaded -> {len(long_term_memories)}")

        # 2. 再把 pending 写入 messages
        last_user_message_id = save_pending_user_messages(user_id, pending_messages)
        print(f"[PENDING SAVED] last_user_message_id={last_user_message_id}")

        # 3. 现在才正式写 interaction_signal / emotion_event / memory
        if last_user_message_id is not None:
            signal = relation_result["signal"]

            save_interaction_signal(
                user_id=user_id,
                trigger_message_id=last_user_message_id,
                openness=signal["openness"],
                warmth=signal["warmth"],
                engagement=signal["engagement"],
                reliance=signal["reliance"],
                respect=signal["respect"],
                rejection=signal["rejection"],
                confidence=signal["confidence"],
                reason_summary=signal["reason_summary"],
            )

            save_emotion_event(
                user_id=user_id,
                trigger_message_id=last_user_message_id,
                source_text=merged_user_text,
                primary_emotion=emotion_result["primary_emotion"],
                secondary_emotion=emotion_result["secondary_emotion"],
                fine_grained=emotion_result["fine_grained"],
                intensity=emotion_result["intensity"],
                confidence=emotion_result["confidence"],
                reason_summary=emotion_result["reason"],
            )

            apply_memory_actions(
                user_id=user_id,
                actions=memory_actions,
                source_message_id=last_user_message_id,
                log_prefix="[MEMORY]",
            )

            relationship_update = relation_result.get("relationship_update")
            if relationship_update:
                save_relationship_update(
                    user_id=user_id,
                    trigger_message_id=last_user_message_id,
                    familiarity_delta=relationship_update["familiarity_delta"],
                    trust_delta=relationship_update["trust_delta"],
                    affection_delta=relationship_update["affection_delta"],
                    dependency_delta=relationship_update["dependency_delta"],
                    confidence=relationship_update["confidence"],
                    reason_summary=relationship_update["reason_summary"],
                )
        else:
            print("[WARN] pending 已分析，但未成功写入 messages，跳过 signal/emotion/memory 入库")

        # 4. 生成回复
        reply = generate_reply(
            user_id,
            emotion_summary=emotion_summary,
            long_term_memories=long_term_memories,
            current_time_text=current_time_text,
        )
        _send_assistant_reply(user_id, reply)

    except Exception as e:
        print(f"\n[ERROR] 强制回复失败: {e}")

    finally:
        idle_manager.clear_pending(user_id)


# 会话级记忆总结：
# 这里只处理已经入库的 messages，不读 pending
def on_summary_due(user_id: str, pending_messages: list[dict], meta: dict):
    print(
        f"\n[SUMMARY] 用户 {user_id} 已静默 {idle_manager.summary_after_seconds} 秒，开始会话级记忆总结"
    )

    try:
        last_summarized_id = get_last_summarized_message_id(user_id)
        rows = get_messages_after_id(user_id, last_summarized_id)

        if not rows:
            print("[SUMMARY] 没有新增消息，跳过")
            return

        session_messages = []
        for row in rows:
            message_id, role, content, created_at = row
            session_messages.append(
                {
                    "role": role,
                    "content": content,
                    "created_at": created_at,
                }
            )

        summary_record = summarize_conversation_session(
            client=client,
            session_messages=session_messages,
        )

        if not summary_record:
            print("[SUMMARY] 未生成有效总结")
            return

        print("[SUMMARY RECORD]", summary_record)
        save_session_summary(
            user_id=user_id,
            summary=summary_record["summary"],
            topics=summary_record["topics"],
            emotional_tone=summary_record["emotional_tone"],
            importance=summary_record["importance"],
            start_message_id=rows[0][0],
            end_message_id=rows[-1][0],
        )

        last_message_id = rows[-1][0]
        set_last_summarized_message_id(user_id, last_message_id)
        print(f"[SUMMARY] 完成，总结边界更新到 message_id={last_message_id}")

        emotion_summary = get_emotion_summary(
            client=client,
            user_id=user_id,
            limit=30,  #情绪总结边界
            force_refresh=True,
        )
        print("[EMOTION SUMMARY @ 5MIN]", emotion_summary)

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

        # 注意：这里不再立刻 save_message
        # 当前输入先进入 pending，等 on_force_reply 时再统一分析并入库
        idle_manager.add_user_message(
            USER_ID,
            user_text,
            on_force_reply,
            on_summary_due,
        )


if __name__ == "__main__":
    main()