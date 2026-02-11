from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class RunFolderManager:
    """
    Responsible for creating reproducible run directories
    and saving config.json.
    """

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)

    def create_run_folder(self, config, label: Optional[str] = None) -> Path:
        """
        Create a new run folder using config hash and timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = config.hash_tag()

        name = f"{timestamp}_{tag}"
        if label:
            name = f"{name}_{label}"

        run_dir = self.output_root / name
        run_dir.mkdir(parents=True, exist_ok=False)

        self._write_config(run_dir, config)

        return run_dir

    @staticmethod
    def _write_config(run_dir: Path, config) -> None:
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
