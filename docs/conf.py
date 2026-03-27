from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "minimal_graspqp"
author = "minimal_graspqp contributors"
copyright = "2026, minimal_graspqp contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = "minimal_graspqp"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "",
    "source_branch": "main",
    "navigation_with_keys": True,
}

os.environ.setdefault("MINIMAL_GRASPQP_SHADOW_ASSETS", "/home/haegu/graspqp/graspqp/assets/shadow_hand")
