from __future__ import annotations

import os
from pathlib import Path


def _candidate_shadow_dirs() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = os.environ.get("MINIMAL_GRASPQP_SHADOW_ASSETS")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            repo_root / "assets" / "shadow_hand",
            Path("/home/haegu/graspqp/graspqp/assets/shadow_hand"),
        ]
    )
    return candidates


def resolve_shadow_hand_asset_dir(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the Shadow Hand asset directory used by the minimal implementation."""

    required = {
        "shadow_hand.urdf",
        "contact_points.json",
        "penetration_points.json",
        "meshes",
        "contact_mesh",
    }
    candidates = [Path(explicit_path)] if explicit_path is not None else _candidate_shadow_dirs()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists() and required.issubset({p.name for p in candidate.iterdir()}):
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve Shadow Hand assets. Checked:\n{checked}")
