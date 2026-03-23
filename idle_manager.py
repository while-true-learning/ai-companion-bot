import threading
from datetime import datetime


class IdleManager:
    def __init__(
        self,
        idle_seconds: int = 8,
        force_reply_after_seconds: int = 40,
        summary_after_seconds: int = 1200
    ):
        self.idle_seconds = idle_seconds
        self.force_reply_after_seconds = force_reply_after_seconds
        self.summary_after_seconds = summary_after_seconds

        self.pending_buffers = {}
        self.idle_timers = {}
        self.force_timers = {}
        self.summary_timers = {}
        self.pending_meta = {}
        self.lock = threading.Lock()

    def add_user_message(
        self,
        user_id: str,
        content: str,
        on_idle_callback,
        on_force_callback,
        on_summary_callback
    ):
        with self.lock:
            if user_id not in self.pending_buffers:
                self.pending_buffers[user_id] = []

            now = datetime.now()
            if user_id not in self.pending_meta or not self.pending_buffers[user_id]:
                self.pending_meta[user_id] = {
                    "first_pending_at": now,
                    "last_user_message_at": now,
                }
            else:
                self.pending_meta[user_id]["last_user_message_at"] = now

            self.pending_buffers[user_id].append({
                "role": "user",
                "content": content,
                "created_at": now.isoformat(timespec="seconds")
            })

            # 重启 idle timer
            old_idle_timer = self.idle_timers.get(user_id)
            if old_idle_timer is not None:
                old_idle_timer.cancel()

            idle_timer = threading.Timer(
                self.idle_seconds,
                self._handle_idle,
                args=(user_id, on_idle_callback)
            )
            self.idle_timers[user_id] = idle_timer
            idle_timer.start()

            # 重启 force reply timer
            old_force_timer = self.force_timers.get(user_id)
            if old_force_timer is not None:
                old_force_timer.cancel()

            force_timer = threading.Timer(
                self.force_reply_after_seconds,
                self._handle_force_reply,
                args=(user_id, on_force_callback)
            )
            self.force_timers[user_id] = force_timer
            force_timer.start()

            # 重启 summary timer
            old_summary_timer = self.summary_timers.get(user_id)
            if old_summary_timer is not None:
                old_summary_timer.cancel()

            summary_timer = threading.Timer(
                self.summary_after_seconds,
                self._handle_summary,
                args=(user_id, on_summary_callback)
            )
            self.summary_timers[user_id] = summary_timer
            summary_timer.start()

    def _handle_idle(self, user_id: str, on_idle_callback):
        with self.lock:
            pending = list(self.pending_buffers.get(user_id, []))
            meta = dict(self.pending_meta.get(user_id, {}))
            self.idle_timers[user_id] = None

        if not pending:
            return

        on_idle_callback(user_id, pending, meta)

    def _handle_force_reply(self, user_id: str, on_force_callback):
        with self.lock:
            pending = list(self.pending_buffers.get(user_id, []))
            meta = dict(self.pending_meta.get(user_id, {}))
            self.force_timers[user_id] = None

        if not pending:
            return

        on_force_callback(user_id, pending, meta)

    def _handle_summary(self, user_id: str, on_summary_callback):
        with self.lock:
            pending = list(self.pending_buffers.get(user_id, []))
            meta = dict(self.pending_meta.get(user_id, {}))
            self.summary_timers[user_id] = None

        on_summary_callback(user_id, pending, meta)

    def get_state(self, user_id: str):
        with self.lock:
            pending = list(self.pending_buffers.get(user_id, []))
            meta = dict(self.pending_meta.get(user_id, {}))
        return pending, meta

    def clear_pending(self, user_id: str):
        with self.lock:
            self.pending_buffers[user_id] = []
            self.pending_meta.pop(user_id, None)

            idle_timer = self.idle_timers.get(user_id)
            if idle_timer is not None:
                idle_timer.cancel()
            self.idle_timers[user_id] = None

            force_timer = self.force_timers.get(user_id)
            if force_timer is not None:
                force_timer.cancel()
            self.force_timers[user_id] = None

            # 注意：不要清掉 summary timer
            # 因为 20 分钟总结需要跨越一次回复继续等待触发

    def cancel_timer(self, user_id: str):
        with self.lock:
            idle_timer = self.idle_timers.get(user_id)
            if idle_timer is not None:
                idle_timer.cancel()
            self.idle_timers[user_id] = None

            force_timer = self.force_timers.get(user_id)
            if force_timer is not None:
                force_timer.cancel()
            self.force_timers[user_id] = None

            summary_timer = self.summary_timers.get(user_id)
            if summary_timer is not None:
                summary_timer.cancel()
            self.summary_timers[user_id] = None