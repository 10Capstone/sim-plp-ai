from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

import services.memory as memory_module
from services.memory import ChatMemoryService


def test_history_is_ordered_bounded_and_isolated():
    memory = ChatMemoryService()
    for i in range(10):
        memory.add_message("s1", "user" if i % 2 == 0 else "assistant", str(i))
    memory.add_message("s2", "user", "separate")
    history = memory.get_history("s1")
    assert [m["content"] for m in history] == [str(i) for i in range(4, 10)]
    assert history[0]["role"] == "user"
    assert datetime.fromisoformat(history[0]["timestamp"]).tzinfo is not None
    assert memory.get_history("s1", 2) == history[-2:]
    assert memory.get_history("s1", 0) == []
    assert memory.get_history("unknown") == []
    assert memory.get_history("s2")[0]["content"] == "separate"


def test_history_returns_copies_and_clear_is_idempotent():
    memory = ChatMemoryService()
    memory.add_exchange("s", "question", "answer")
    memory.get_history("s")[0]["content"] = "modified"
    assert memory.get_history("s")[0]["content"] == "question"
    memory.clear("s")
    memory.clear("s")
    assert memory.get_history("s") == []


def test_idle_ttl_and_lru_capacity(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(memory_module, "monotonic", lambda: now[0])
    memory = ChatMemoryService(max_sessions=2, ttl_seconds=10)
    memory.add_message("s1", "user", "a")
    memory.add_message("s2", "user", "b")
    now[0] = 5
    assert memory.get_history("s1")
    memory.add_message("s3", "user", "c")
    assert memory.get_history("s2") == []
    now[0] = 12
    assert memory.get_history("s1")  # A read extends the idle TTL.
    now[0] = 15
    assert memory.get_history("s3") == []
    now[0] = 22
    assert memory.get_history("s1") == []


def test_concurrent_exchanges_remain_atomic():
    memory = ChatMemoryService(max_messages=200)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(lambda i: memory.add_exchange("s", str(i), str(i)), range(100))
        )
    history = memory.get_history("s", 200)
    assert len(history) == 200
    for user, assistant in zip(history[::2], history[1::2]):
        assert user["role"] == "user" and assistant["role"] == "assistant"
        assert user["content"] == assistant["content"]


@pytest.mark.parametrize(
    "kwargs", [{"max_messages": 0}, {"max_sessions": 0}, {"ttl_seconds": 0}]
)
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        ChatMemoryService(**kwargs)


def test_invalid_role_session_and_history_limit():
    memory = ChatMemoryService()
    with pytest.raises(ValueError):
        memory.add_message("s", "system", "bad")
    with pytest.raises(ValueError):
        memory.add_message(" ", "user", "bad")
    with pytest.raises(ValueError):
        memory.get_history("s", -1)
