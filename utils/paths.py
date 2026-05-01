"""Centralised path resolution for the project.

All other modules import get_path() from here instead of
defining PROJECT_ROOT locally.  dirname is called twice because
this file lives one level deeper (project/utils/paths.py), so
we need to step up twice to reach the project root.
"""

import os

# project/utils/paths.py  →  dirname once = project/utils/
#                          →  dirname twice = project/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path(*parts: str) -> str:
    """Return an absolute path by joining PROJECT_ROOT with *parts.

    Examples
    --------
    get_path("models", "rf_model.pkl")  →  /abs/path/to/project/models/rf_model.pkl
    get_path("data")                    →  /abs/path/to/project/data
    """
    return os.path.join(PROJECT_ROOT, *parts)
