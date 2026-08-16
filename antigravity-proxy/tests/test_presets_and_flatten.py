from pathlib import Path

import pytest

from agy_proxy.presets import PresetRegistry
from agy_proxy.runner import flatten_messages, latest_user_text

PRESETS = str(Path(__file__).resolve().parent.parent / "presets.yaml")


def _registry() -> PresetRegistry:
    return PresetRegistry.load(PRESETS, default_model="gemini-3-pro")


def test_resolve_known_and_fallback():
    reg = _registry()
    assert "agy-coder" in reg.names()
    assert reg.resolve("agy-researcher").name == "agy-researcher"
    # Unknown model name falls back to the default preset.
    assert reg.resolve("gpt-4o").name == reg.default_name
    assert reg.resolve(None).name == reg.default_name


def test_effective_model_default():
    reg = _registry()
    p = reg.resolve("agy-coder")
    # Preset does not pin a model -> falls back to registry default.
    assert reg.effective_model(p) == "gemini-3-pro"


def test_researcher_disables_edit_tools():
    reg = _registry()
    p = reg.resolve("agy-researcher")
    assert "EDIT_FILE" in (p.disabled_tools or [])
    assert p.policy == "safe_defaults"


def test_flatten_single_user():
    system, prompt = flatten_messages([{"role": "user", "content": "hi"}])
    assert system is None
    assert prompt == "hi"


def test_flatten_system_extracted():
    system, prompt = flatten_messages(
        [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
    )
    assert system == "be nice"
    assert prompt == "hi"


def test_flatten_multiturn_transcript():
    msgs = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    _, prompt = flatten_messages(msgs)
    assert "User: one" in prompt
    assert "Assistant: two" in prompt
    assert "User: three" in prompt


def test_latest_user_text():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert latest_user_text(msgs) == "second"


def test_bad_preset_rejected(tmp_path):
    bad = tmp_path / "p.yaml"
    bad.write_text("default: x\npresets:\n  x:\n    policy: nonsense\n")
    with pytest.raises(ValueError):
        PresetRegistry.load(str(bad), "gemini-3-pro")
