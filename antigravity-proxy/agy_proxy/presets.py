"""Agent presets -- your 'personal preference of agents'.

A preset maps a client-requested model name (e.g. ``agy-researcher``) to a concrete
Antigravity agent configuration: system instructions, the underlying Gemini
model, an explicit tool allowlist, autonomy policy, and thinking depth.

Presets are declared in ``presets.yaml`` and resolved by name at request time.
Security-relevant rules enforced here (see README "Security"):

* tools are an ALLOWLIST (``enabled_tools``); the SDK's ``confirm_run_command`` policy
  allows file writes, so the allowlist is the real barrier for read-only presets;
* a preset that can write files / run commands needs an explicit workspace, otherwise
  it is loaded *disabled*;
* unknown model names are an error (404), never a silent fallback to a powerful agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import broad_workspace_reason
from .errors import UnknownModelError

logger = logging.getLogger("agy_proxy.presets")

# ``workspace_only`` is intentionally absent: in SDK 0.1.21 it denies *all* file
# tools and does not restrict paths, so it cannot express "stay inside workspace".
_POLICY_NAMES = {"allow_all", "deny_all", "safe_defaults", "confirm_run_command"}
_BEHAVIORS = {"autonomous", "interactive"}
_THINKING = {"minimal", "low", "medium", "high", "extra_high"}

# Tools whose presence means the agent can change the host or act beyond reading.
WRITE_CAPABLE_TOOLS = frozenset(
    {"CREATE_FILE", "EDIT_FILE", "RUN_COMMAND", "START_SUBAGENT", "SCHEDULE"}
)
# Preset that is only enabled when the operator opts in (AGY_ENABLE_CODER=1).
CODER_PRESET = "agy-coder"

_KNOWN_KEYS = {
    "description",
    "system_instructions",
    "model",
    "policy",
    "behavior",
    "thinking_level",
    "enabled_tools",
    "enable_subagents",
    "allow_client_system",
    "workspaces",
    "enabled",
}


def valid_tool_names() -> set[str]:
    from google.antigravity import BuiltinTools

    return {t.name for t in BuiltinTools}


@dataclass(frozen=True)
class Preset:
    """A named agent configuration selectable by clients via the model field."""

    name: str
    description: str = ""
    system_instructions: str | None = None
    model: str | None = None
    policy: str = "safe_defaults"  # one of _POLICY_NAMES
    behavior: str = "autonomous"  # autonomous | interactive
    thinking_level: str | None = None  # None | minimal..extra_high
    # Allowlist of BuiltinTools names. ``None`` means "every default tool" and is
    # only accepted together with policy allow_all. ``[]`` means no tools at all.
    enabled_tools: list[str] | None = None
    enable_subagents: bool = False
    # Whether client-supplied system text is passed to the agent (labelled untrusted).
    allow_client_system: bool = False
    workspaces: list[str] = field(default_factory=list)

    @property
    def needs_workspace(self) -> bool:
        """True when the agent may write files, run commands, or spawn actors."""
        if self.enabled_tools is None:
            return True
        return bool(WRITE_CAPABLE_TOOLS & {t.upper() for t in self.enabled_tools})

    def validate(self) -> None:
        if self.policy == "workspace_only":
            raise ValueError(
                f"preset {self.name!r}: policy 'workspace_only' is not supported (the SDK's "
                "workspace_only denies every file tool and does not restrict paths); use "
                "an enabled_tools allowlist plus an explicit workspace"
            )
        if self.policy not in _POLICY_NAMES:
            raise ValueError(f"preset {self.name!r}: unknown policy {self.policy!r}")
        if self.behavior not in _BEHAVIORS:
            raise ValueError(f"preset {self.name!r}: unknown behavior {self.behavior!r}")
        if self.thinking_level is not None and self.thinking_level not in _THINKING:
            raise ValueError(
                f"preset {self.name!r}: unknown thinking_level {self.thinking_level!r}"
            )
        if self.enabled_tools is None:
            if self.policy != "allow_all":
                raise ValueError(
                    f"preset {self.name!r}: enabled_tools is required (omit it only with "
                    "policy allow_all); list the tools the agent may use"
                )
        else:
            known = valid_tool_names()
            for tool in self.enabled_tools:
                if str(tool).upper() not in known:
                    raise ValueError(f"preset {self.name!r}: unknown builtin tool {tool!r}")
        for ws in self.workspaces:
            p = Path(ws)
            if not p.is_absolute() or not p.is_dir():
                raise ValueError(
                    f"preset {self.name!r}: workspace {ws!r} must be an existing absolute directory"
                )
            broad = broad_workspace_reason(p)
            if broad:
                raise ValueError(f"preset {self.name!r}: workspace {ws!r} is {broad}")


class PresetRegistry:
    """Loads presets from YAML and resolves requested model names to presets."""

    def __init__(
        self,
        presets: dict[str, Preset],
        default_name: str,
        default_model: str,
        disabled: dict[str, str] | None = None,
    ):
        if not presets:
            raise ValueError(
                f"no enabled presets (disabled: {sorted(disabled or {})}); "
                "check presets.yaml, AGY_WORKSPACE and AGY_ENABLE_CODER"
            )
        if default_name not in presets:
            raise ValueError(
                f"default preset {default_name!r} is not an enabled preset "
                f"(enabled: {sorted(presets)}; disabled: {sorted(disabled or {})})"
            )
        self._presets = presets
        self._default_name = default_name
        self._default_model = default_model
        self._disabled = dict(disabled or {})

    @classmethod
    def load(
        cls,
        path: str,
        default_model: str,
        *,
        workspace: str | None = None,
        enable_coder: bool = False,
    ) -> PresetRegistry:
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        default_name = data.get("default", "")
        raw = data.get("presets", {}) or {}
        presets: dict[str, Preset] = {}
        disabled: dict[str, str] = {}
        for name, cfg in raw.items():
            cfg = cfg or {}
            if "disabled_tools" in cfg:
                raise ValueError(
                    f"preset {name!r}: 'disabled_tools' was removed; list the tools the agent "
                    "may use in 'enabled_tools' (an allowlist) instead"
                )
            unknown = set(cfg) - _KNOWN_KEYS
            if unknown:
                raise ValueError(f"preset {name!r}: unknown keys {sorted(unknown)}")
            tools = cfg.get("enabled_tools")
            if tools is not None and not isinstance(tools, list):
                raise ValueError(f"preset {name!r}: enabled_tools must be a list")
            preset = Preset(
                name=name,
                description=cfg.get("description", ""),
                system_instructions=cfg.get("system_instructions"),
                model=cfg.get("model"),
                policy=cfg.get("policy", "safe_defaults"),
                behavior=cfg.get("behavior", "autonomous"),
                thinking_level=cfg.get("thinking_level"),
                enabled_tools=[str(t) for t in tools] if tools is not None else None,
                enable_subagents=bool(cfg.get("enable_subagents", False)),
                allow_client_system=bool(cfg.get("allow_client_system", False)),
                workspaces=[str(w) for w in (cfg.get("workspaces") or [])],
            )
            preset.validate()
            enabled = bool(cfg.get("enabled", True))
            if preset.needs_workspace:
                # AGY_ENABLE_CODER is the single opt-in for EVERY write-capable preset,
                # whatever its name and whatever `enabled:` says in the yaml.
                enabled = enable_coder and (enabled or name == CODER_PRESET)
            if not enabled:
                disabled[name] = "disabled (set AGY_ENABLE_CODER=1 or edit presets.yaml)"
            elif preset.needs_workspace and not (workspace or preset.workspaces):
                disabled[name] = "needs AGY_WORKSPACE (can write files / run commands)"
                logger.warning("preset %r disabled: %s", name, disabled[name])
            else:
                presets[name] = preset
        if not default_name:
            default_name = next(iter(presets), "")
        return cls(presets, default_name, default_model, disabled)

    def resolve(self, model_name: str | None) -> Preset:
        """Resolve a requested model name to an enabled preset.

        The default preset is used only when ``model`` is absent/empty. Any other
        unknown (or disabled) name raises :class:`UnknownModelError` (HTTP 404).
        """
        if not model_name:
            return self._presets[self._default_name]
        preset = self._presets.get(model_name)
        if preset is None:
            raise UnknownModelError(f"model {model_name!r} not found", param="model")
        return preset

    def effective_model(self, preset: Preset) -> str:
        return preset.model or self._default_model

    def names(self) -> list[str]:
        return list(self._presets)

    @property
    def disabled(self) -> dict[str, str]:
        return dict(self._disabled)

    @property
    def default_name(self) -> str:
        return self._default_name
