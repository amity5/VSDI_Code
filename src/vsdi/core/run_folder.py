from __future__ import annotations

import json
import hashlib
from typing import Mapping, Any


def config_tag(config_dict: Mapping[str, Any], n_chars: int = 8) -> str:
    """
    Compute a stable short hash tag for a configuration dictionary.

    This is intentionally config-only, no filesystem side effects.
    Run folder creation/writing is handled by vsdi.paths.RunFolderManager.
    """
    if n_chars <= 0:
        raise ValueError("n_chars must be > 0")

    payload = json.dumps(dict(config_dict), sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:n_chars]
