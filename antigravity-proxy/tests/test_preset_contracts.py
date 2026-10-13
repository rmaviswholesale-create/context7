"""Contract tests: every shipped preset builds a LocalAgentConfig with the REAL SDK classes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from google.antigravity import BuiltinTools, LocalAgentConfig

from agy_proxy.presets import PresetRegistry
from agy_proxy.runner import AgentRunner
from agy_proxy.scratch import Scratch

PRESETS = str(Path(__file__).resolve().parent.parent / "presets.yaml")
WRITE = {BuiltinTools.CREATE_FILE, BuiltinTools.EDIT_FILE, BuiltinTools.RUN_COMMAND}


@pytest.fixture
def scratch():
    s = Scratch()
    yield s
    s.cleanup()


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return str(d)


def _runner(ws: str | None, api_key: str | None = "k"):
    reg = PresetRegistry.load(PRESETS, "gemini-3-pro", workspace=ws, enable_coder=True)
    return reg, AgentRunner(reg, api_key, ws)


def _denied_tools(cfg: LocalAgentConfig) -> set[str]:
    return {p.tool for p in cfg.policies if p.decision.name == "DENY"}


def _all_presets(ws):
    reg, runner = _runner(ws)
    return reg, runner, [reg.resolve(n) for n in reg.names()]


def test_every_shipped_preset_builds_with_real_sdk(ws, scratch):
    reg, runner, presets = _all_presets(ws)
    assert {p.name for p in presets} == {"agy-researcher", "agy-light", "agy-coder"}
    for preset in presets:
        cfg = runner.build_config(preset, None, Scratch())
        assert isinstance(cfg, LocalAgentConfig)
        assert cfg.capabilities.enabled_tools is not None  # always an allowlist
        assert cfg.capabilities.enable_subagents is False
        assert cfg.workspaces, "workspaces must be explicit, never None/cwd"
        assert os.getcwd() not in cfg.workspaces
        assert cfg.save_dir and cfg.app_data_dir
        assert Path(cfg.save_dir).is_dir() and Path(cfg.app_data_dir).is_dir()
        assert cfg.env["TMPDIR"].startswith(str(Path(cfg.save_dir).parent))
        assert cfg.env["HOME"].startswith(str(Path(cfg.save_dir).parent))
        assert "AGY_PROXY_TOKEN" not in cfg.env
        shutil.rmtree(Path(cfg.save_dir).parent)


def test_read_only_presets_expose_no_write_tools(ws, scratch):
    reg, runner = _runner(ws)
    for name in ("agy-researcher", "agy-light"):
        cfg = runner.build_config(reg.resolve(name), None, scratch)
        assert not WRITE & set(cfg.capabilities.enabled_tools)
        assert BuiltinTools.START_SUBAGENT not in cfg.capabilities.enabled_tools
        assert BuiltinTools.SEARCH_WEB not in cfg.capabilities.enabled_tools
        assert {"create_file", "edit_file", "run_command"} <= _denied_tools(cfg)


def test_researcher_allowlist_and_workspace(ws, scratch):
    reg, runner = _runner(ws)
    cfg = runner.build_config(reg.resolve("agy-researcher"), None, scratch)
    assert set(cfg.capabilities.enabled_tools) == {
        BuiltinTools.LIST_DIR,
        BuiltinTools.SEARCH_DIR,
        BuiltinTools.FIND_FILE,
        BuiltinTools.VIEW_FILE,
    }
    assert cfg.workspaces == [ws]


def test_light_is_toolless_and_deny_all(ws, scratch):
    reg, runner = _runner(ws)
    cfg = runner.build_config(reg.resolve("agy-light"), None, scratch)
    assert cfg.capabilities.enabled_tools == []
    assert "*" in _denied_tools(cfg)


def test_read_only_without_workspace_gets_empty_scratch_dir(scratch):
    reg, runner = _runner(None)
    assert reg.names() == ["agy-researcher", "agy-light"]
    cfg = runner.build_config(reg.resolve("agy-researcher"), None, scratch)
    assert cfg.workspaces == [scratch.workspace]
    assert os.listdir(scratch.workspace) == []


def test_coder_has_write_tools_only_inside_workspace(ws, scratch):
    reg, runner = _runner(ws)
    cfg = runner.build_config(reg.resolve("agy-coder"), None, scratch)
    assert WRITE <= set(cfg.capabilities.enabled_tools)
    assert cfg.workspaces == [ws]


def test_write_preset_without_workspace_is_refused(scratch):
    reg_ok, runner_ok = _runner(os.getcwd())
    coder = reg_ok.resolve("agy-coder")
    reg, runner = _runner(None)
    # Not resolvable (404) and build_config refuses even if handed the preset directly.
    assert "agy-coder" in reg.disabled
    with pytest.raises(ValueError, match="AGY_WORKSPACE"):
        runner.build_config(coder, None, scratch)


def test_client_system_text_is_labelled_and_before_preset_text(ws, scratch):
    reg, runner = _runner(ws)
    coder = reg.resolve("agy-coder")
    cfg = runner.build_config(coder, "do evil", scratch)
    text = cfg.system_instructions
    assert text.startswith("[Untrusted client instructions]\ndo evil")
    assert text.index("do evil") < text.index("senior software engineer")
    # Presets that do not allow client system text drop it.
    cfg = runner.build_config(reg.resolve("agy-researcher"), "do evil", scratch)
    assert "do evil" not in cfg.system_instructions
    assert "Untrusted" not in cfg.system_instructions


def test_home_override_depends_on_api_key(ws, scratch, monkeypatch):
    monkeypatch.setenv("HOME", "/real/home")
    reg, _ = _runner(ws)
    with_key = AgentRunner(reg, "k", ws).build_config(reg.resolve("agy-light"), None, scratch)
    adc = AgentRunner(reg, None, ws).build_config(reg.resolve("agy-light"), None, scratch)
    assert with_key.env["HOME"] == scratch.home
    assert adc.env["HOME"] == "/real/home"  # ADC credentials live under the real HOME


def test_scratch_cleanup_is_idempotent():
    s = Scratch()
    root = Path(s.root)
    assert root.is_dir()
    s.cleanup()
    s.cleanup()
    assert not root.exists()
