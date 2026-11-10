from collections import OrderedDict, deque
from datetime import UTC, datetime
from threading import RLock
from time import monotonic
from typing import Literal


class ChatMemoryService:
    """Thread-safe, process-local memory with bounded history and idle expiry."""

    def __init__(
        self,
        max_messages: int = 6,
        max_sessions: int = 1000,
        ttl_seconds: float = 3600,
    ):
        if max_messages < 1 or max_sessions < 1 or ttl_seconds <= 0:
            raise ValueError("Memory limits and TTL must be positive")
        self.max_messages = max_messages
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: OrderedDict[str, tuple[float, deque]] = OrderedDict()
        self._lock = RLock()

    def _prune(self, now: float) -> None:
        while self._sessions:
            _, (touched, _) = next(iter(self._sessions.items()))
            if now - touched < self.ttl_seconds:
                break
            self._sessions.popitem(last=False)

    def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ) -> None:
        if not session_id.strip() or role not in ("user", "assistant"):
            raise ValueError("A session ID and a valid message role are required")
        with self._lock:
            now = monotonic()
            self._prune(now)
            entry = self._sessions.pop(session_id, None)
            messages = entry[1] if entry else deque(maxlen=self.max_messages)
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            self._sessions[session_id] = (now, messages)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def add_exchange(self, session_id: str, user: str, assistant: str) -> None:
        """Commit both sides together, only after a successful answer."""
        with self._lock:
            self.add_message(session_id, "user", user)
            self.add_message(session_id, "assistant", assistant)

    def get_history(self, session_id: str, max_messages: int = 6) -> list[dict]:
        if max_messages < 0:
            raise ValueError("max_messages cannot be negative")
        with self._lock:
            now = monotonic()
            self._prune(now)
            entry = self._sessions.pop(session_id, None)
            if entry is None:
                return []
            self._sessions[session_id] = (now, entry[1])
            return (
                [dict(item) for item in list(entry[1])[-max_messages:]]
                if max_messages
                else []
            )

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
