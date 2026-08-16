from agy_proxy.presets import Preset
from agy_proxy.sessions import SessionStore


class FakeAgent:
    def __init__(self, preset_name: str):
        self.preset_name = preset_name
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False


class FakeRunner:
    def __init__(self):
        self.created: list[FakeAgent] = []

    def new_agent(self, preset: Preset, system_extra=None) -> FakeAgent:
        a = FakeAgent(preset.name)
        self.created.append(a)
        return a


def _preset(name: str) -> Preset:
    return Preset(name=name)


async def test_reuses_agent_for_same_session():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=10)
    p = _preset("agy-coder")
    async with store.acquire("s1", p, None) as a1:
        pass
    async with store.acquire("s1", p, None) as a2:
        pass
    assert a1 is a2
    assert store.size() == 1


async def test_preset_change_rebuilds():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=10)
    async with store.acquire("s1", _preset("agy-coder"), None) as a1:
        pass
    async with store.acquire("s1", _preset("agy-fast"), None) as a2:
        pass
    assert a1 is not a2
    assert a1.closed is True
    assert a2.preset_name == "agy-fast"


async def test_lru_eviction_closes_oldest():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=2)
    async with store.acquire("s1", _preset("p"), None) as a1:
        pass
    async with store.acquire("s2", _preset("p"), None):
        pass
    async with store.acquire("s3", _preset("p"), None):
        pass
    assert store.size() == 2
    assert a1.closed is True  # oldest evicted


async def test_ttl_expiry(monkeypatch):
    import agy_proxy.sessions as s

    t = {"now": 1000.0}
    monkeypatch.setattr(s.time, "monotonic", lambda: t["now"])
    store = SessionStore(FakeRunner(), ttl_seconds=10, max_size=10)
    async with store.acquire("s1", _preset("p"), None) as a1:
        pass
    t["now"] = 1020.0  # advance past ttl
    async with store.acquire("s2", _preset("p"), None):
        pass
    assert a1.closed is True
    assert store.size() == 1


async def test_close_all():
    store = SessionStore(FakeRunner(), ttl_seconds=1000, max_size=10)
    async with store.acquire("s1", _preset("p"), None) as a1:
        pass
    await store.close_all()
    assert a1.closed is True
    assert store.size() == 0
