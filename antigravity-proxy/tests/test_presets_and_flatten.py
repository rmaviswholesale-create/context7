from pathlib import Path

import pytest

from agy_proxy.errors import UnknownModelError
from agy_proxy.messages import flatten_messages, latest_user_text
from agy_proxy.presets import PresetRegistry

PRESETS = str(Path(__file__).resolve().parent.parent / "presets.yaml")


def _registry(tmp_path=None, **kw) -> PresetRegistry:
    ws = str(tmp_path) if tmp_path else None
    return PresetRegistry.load(PRESETS, default_model="gemini-3-pro", workspace=ws, **kw)


def test_resolve_known_unknown_and_default():
    reg = _registry()
    assert "agy-researcher" in reg.names()
    assert reg.resolve("agy-researcher").name == "agy-researcher"
    # Default preset only when the model is absent.
    assert reg.resolve(None).name == reg.default_name == "agy-researcher"
    assert reg.resolve("").name == reg.default_name
    # Unknown names are an error, never a silent fallback.
    with pytest.raises(UnknownModelError) as ei:
        reg.resolve("gpt-4o")
    assert ei.value.status == 404


def test_effective_model_default():
    reg = _registry()
    p = reg.resolve("agy-researcher")
    # Preset does not pin a model -> falls back to registry default.
    assert reg.effective_model(p) == "gemini-3-pro"


def test_researcher_is_read_only_allowlist():
    p = _registry().resolve("agy-researcher")
    assert p.enabled_tools == ["LIST_DIR", "SEARCH_DIR", "FIND_FILE", "VIEW_FILE"]
    assert not p.needs_workspace
    assert p.enable_subagents is False
    assert p.allow_client_system is False


def test_coder_is_opt_in_and_needs_workspace(tmp_path):
    assert "agy-coder" not in _registry(tmp_path).names()
    assert "agy-coder" in _registry(tmp_path).disabled
    # Opted in but no workspace: still disabled.
    reg = _registry(enable_coder=True)
    assert "agy-coder" not in reg.names()
    assert "AGY_WORKSPACE" in reg.disabled["agy-coder"]
    with pytest.raises(UnknownModelError):
        reg.resolve("agy-coder")
    # Opted in with a workspace: enabled.
    assert "agy-coder" in _registry(tmp_path, enable_coder=True).names()


def _write(tmp_path, body: str) -> str:
    f = tmp_path / "p.yaml"
    f.write_text(body)
    return str(f)


@pytest.mark.parametrize(
    "body, needle",
    [
        ("  x:\n    disabled_tools: [EDIT_FILE]\n    enabled_tools: [VIEW_FILE]\n", "enabled_tools"),
        ("  x:\n    policy: workspace_only\n    enabled_tools: [VIEW_FILE]\n", "workspace_only"),
        ("  x:\n    policy: allow_all\n    enabled_tools: [NOPE]\n", "unknown builtin tool"),
        ("  x:\n    policy: safe_defaults\n", "enabled_tools is required"),
        ("  x:\n    enabled_tools: [VIEW_FILE]\n    bogus: 1\n", "unknown keys"),
        ("  x:\n    policy: nonsense\n    enabled_tools: []\n", "unknown policy"),
    ],
)
def test_invalid_presets_rejected(tmp_path, body, needle):
    path = _write(tmp_path, "default: x\npresets:\n" + body)
    with pytest.raises(ValueError, match=needle):
        PresetRegistry.load(path, "gemini-3-pro", workspace=str(tmp_path))


def test_allow_all_without_tool_list_needs_workspace(tmp_path):
    path = _write(tmp_path, "default: y\npresets:\n  x:\n    policy: allow_all\n  y:\n"
                  "    enabled_tools: []\n    policy: deny_all\n")
    reg = PresetRegistry.load(path, "gemini-3-pro")
    assert reg.names() == ["y"] and "x" in reg.disabled


def test_disabled_default_rejected(tmp_path):
    path = _write(tmp_path, "default: x\npresets:\n  x:\n    policy: allow_all\n")
    with pytest.raises(ValueError, match="no enabled presets"):
        PresetRegistry.load(path, "gemini-3-pro")


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
    assert "User:\n    one" in prompt
    assert "Assistant:\n    two" in prompt
    assert "User:\n    three" in prompt


def test_latest_user_text():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert latest_user_text(msgs) == "second"
