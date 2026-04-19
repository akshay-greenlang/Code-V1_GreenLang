# -*- coding: utf-8 -*-
"""Runtime profiles for GreenLang v2 app set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V2AppProfile:
    app_id: str
    key: str
    v2_dir: Path
    command_template: str


V2_APP_PROFILES: dict[str, V2AppProfile] = {
    "cbam": V2AppProfile(
        app_id="GL-CBAM-APP",
        key="cbam",
        v2_dir=Path("applications/GL-CBAM-APP/v2"),
        command_template="gl run cbam <config.yaml> <imports.csv> <output_dir>",
    ),
    "csrd": V2AppProfile(
        app_id="GL-CSRD-APP",
        key="csrd",
        v2_dir=Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/v2"),
        command_template="gl run csrd <input.csv|json> <output_dir>",
    ),
    "vcci": V2AppProfile(
        app_id="GL-VCCI-Carbon-APP",
        key="vcci",
        v2_dir=Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v2"),
        command_template="gl run vcci <input.csv|json> <output_dir>",
    ),
    "eudr": V2AppProfile(
        app_id="GL-EUDR-APP",
        key="eudr",
        v2_dir=Path("applications/GL-EUDR-APP/v2"),
        command_template="gl run eudr <input.json> <output_dir>",
    ),
    "ghg": V2AppProfile(
        app_id="GL-GHG-APP",
        key="ghg",
        v2_dir=Path("applications/GL-GHG-APP/v2"),
        command_template="gl run ghg <input.json> <output_dir>",
    ),
    "iso14064": V2AppProfile(
        app_id="GL-ISO14064-APP",
        key="iso14064",
        v2_dir=Path("applications/GL-ISO14064-APP/v2"),
        command_template="gl run iso14064 <input.json> <output_dir>",
    ),
    "sb253": V2AppProfile(
        app_id="GL-SB253-APP",
        key="sb253",
        v2_dir=Path("applications/GL-SB253-APP/v2"),
        command_template="gl run sb253 <input.json> <output_dir>",
    ),
    "taxonomy": V2AppProfile(
        app_id="GL-Taxonomy-APP",
        key="taxonomy",
        v2_dir=Path("applications/GL-Taxonomy-APP/v2"),
        command_template="gl run taxonomy <input.json> <output_dir>",
    ),
}


def get_profile(key: str) -> V2AppProfile:
    lowered = key.lower()
    if lowered not in V2_APP_PROFILES:
        valid = ", ".join(sorted(V2_APP_PROFILES))
        raise ValueError(f"unknown profile '{key}', expected one of: {valid}")
    return V2_APP_PROFILES[lowered]

