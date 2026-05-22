"""On-disk JSON preset store, keyed by (arm_id, name)."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PRESETS_PATH = Path.home() / ".haller" / "presets.json"


class PresetNotFound(Exception):
    pass


class PresetStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_PRESETS_PATH)
        self._data: dict[str, dict[str, dict[str, float]]] = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def save(self, name: str, arm_id: str, joints_deg: dict[str, float]) -> None:
        self._data.setdefault(arm_id, {})[name] = dict(joints_deg)
        self._write()

    def get(self, name: str, arm_id: str) -> dict[str, float]:
        try:
            return dict(self._data[arm_id][name])
        except KeyError as e:
            raise PresetNotFound(f"no preset {name!r} for arm {arm_id!r}") from e

    def list(self, arm_id: str) -> list[str]:
        return list(self._data.get(arm_id, {}).keys())
