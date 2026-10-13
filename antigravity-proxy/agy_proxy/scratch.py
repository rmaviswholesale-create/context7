"""Proxy-owned scratch directories handed to the SDK (save_dir / app_data_dir / HOME).

The SDK leaves ``mkdtemp`` directories behind forever when ``save_dir`` is None, so the
proxy creates and removes them itself.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class Scratch:
    """One temp tree per agent (save/appdata/home/tmp/workspace). Cleanup is idempotent."""

    def __init__(self, prefix: str = "agy-proxy-"):
        self.root = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
        self.save_dir = self._sub("save")
        self.app_data_dir = self._sub("appdata")
        self.home = self._sub("home")
        self.tmp = self._sub("tmp")
        self.workspace = self._sub("workspace")  # empty dir for presets with no workspace
        self._closed = False

    def _sub(self, name: str) -> str:
        path = self.root / name
        path.mkdir(mode=0o700)
        return str(path)

    @property
    def closed(self) -> bool:
        return self._closed

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.root, ignore_errors=True)
