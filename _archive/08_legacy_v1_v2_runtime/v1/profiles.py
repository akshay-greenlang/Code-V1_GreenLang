# -*- coding: utf-8 -*-
"""Runtime profiles for GreenLang v1 app set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V1AppProfile:
    app_id: str
    key: str
    v1_dir: Path
    command_template: str


V1_APP_PROFILES: dict[str, V1AppProfile] = {
    "cbam": V1AppProfile(
        app_id="GL-CBAM-APP",
        key="cbam",
        v1_dir=Path("applications/GL-CBAM-APP/v1"),
        command_template="gl run cbam <config.yaml> <imports.csv> <output_dir>",
    ),
    "csrd": V1AppProfile(
        app_id="GL-CSRD-APP",
        key="csrd",
        v1_dir=Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1"),
        command_template="gl run csrd <input.csv|json> <output_dir>",
    ),
    "vcci": V1AppProfile(
        app_id="GL-VCCI-Carbon-APP",
        key="vcci",
        v1_dir=Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1"),
        command_template="gl run vcci <input.csv|json> <output_dir>",
    ),
}


def get_profile(key: str) -> V1AppProfile:
    lowered = key.lower()
    if lowered not in V1_APP_PROFILES:
        valid = ", ".join(sorted(V1_APP_PROFILES))
        raise ValueError(f"unknown profile '{key}', expected one of: {valid}")
    return V1_APP_PROFILES[lowered]

