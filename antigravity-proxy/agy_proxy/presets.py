"""Agent presets -- your 'personal preference of agents'.

A preset maps a client-requested model name (e.g. ``agy-coder``) to a concrete
Antigravity agent configuration: system instructions, the underlying Gemini
model, capability/tool profile, autonomy policy, and thinking depth.

Presets are declared in ``presets.yaml`` and resolved by name at request time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_POLICY_NAMES = {"allow_all", "deny_all", "safe_defaults", "workspace_only", "confirm_run_command"}
_BEHAVIORS = {"autonomous", "interactive"}
_THINKING = {"minimal", "low", "medium", "high", "extra_high"}


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
    enabled_tools: list[str] | None = None  # BuiltinTools names; None = all
    disabled_tools: list[str] | None = None
    enable_subagents: bool = True
    workspaces: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.policy not in _POLICY_NAMES:
            raise ValueError(f"preset {self.name!r}: unknown policy {self.policy!r}")
        if self.behavior not in _BEHAVIORS:
            raise ValueError(f"preset {self.name!r}: unknown behavior {self.behavior!r}")
        if self.thinking_level is not None and self.thinking_level not in _THINKING:
            raise ValueError(
                f"preset {self.name!r}: unknown thinking_level {self.thinking_level!r}"
            )


class PresetRegistry:
    """Loads presets from YAML and resolves requested model names to presets."""

    def __init__(self, presets: dict[str, Preset], default_name: str, default_model: str):
        if not presets:
            raise ValueError("no presets loaded")
        if default_name not in presets:
            raise ValueError(f"default preset {default_name!r} not in presets")
        self._presets = presets
        self._default_name = default_name
        self._default_model = default_model

    @classmethod
    def load(cls, path: str, default_model: str) -> PresetRegistry:
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        default_name = data.get("default", "")
        raw = data.get("presets", {}) or {}
        presets: dict[str, Preset] = {}
        for name, cfg in raw.items():
            cfg = cfg or {}
            preset = Preset(
                name=name,
                description=cfg.get("description", ""),
                system_instructions=cfg.get("system_instructions"),
                model=cfg.get("model"),
                policy=cfg.get("policy", "safe_defaults"),
                behavior=cfg.get("behavior", "autonomous"),
                thinking_level=cfg.get("thinking_level"),
                enabled_tools=cfg.get("enabled_tools"),
                disabled_tools=cfg.get("disabled_tools"),
                enable_subagents=cfg.get("enable_subagents", True),
                workspaces=cfg.get("workspaces", []) or [],
            )
            preset.validate()
            presets[name] = preset
        if not default_name:
            default_name = next(iter(presets))
        return cls(presets, default_name, default_model)

    def resolve(self, model_name: str | None) -> Preset:
        """Resolve a requested model name to a preset.

        Falls back to the configured default when the name is unknown, so a
        client asking for e.g. ``gpt-4o`` still gets a working agent.
        """
        if model_name and model_name in self._presets:
            return self._presets[model_name]
        return self._presets[self._default_name]

    def effective_model(self, preset: Preset) -> str:
        return preset.model or self._default_model

    def names(self) -> list[str]:
        return list(self._presets)

    @property
    def default_name(self) -> str:
        return self._default_name
