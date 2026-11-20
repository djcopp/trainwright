"""Project layout on disk.

::

    trainwright.yaml        # ProjectConfig
    experiments/*.yaml      # optional named experiment configs
    .trainwright/
        profile.json        # data profile (trainwright profile)
        split.json          # frozen split (trainwright split)
        test_ledger.jsonl   # every evaluation on the test split
        benchmark/<run>.json
    runs/<id>/              # one directory per run (see tracking.py)
    artifacts/<run>/        # exported, deployable bundles
    reports/REPORT.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ConfigError, ProjectConfig, format_validation_error

CONFIG_NAME = "trainwright.yaml"


class ProjectNotFound(FileNotFoundError):
    pass


@dataclass
class Project:
    root: Path
    config: ProjectConfig

    # ----------------------------------------------------------------- loading
    @classmethod
    def find(cls, start: Path | str | None = None) -> Project:
        start_path = Path(start or Path.cwd()).resolve()
        if start_path.is_file():
            start_path = start_path.parent
        for d in [start_path, *start_path.parents]:
            if (d / CONFIG_NAME).exists():
                return cls.load(d)
        raise ProjectNotFound(
            f"No {CONFIG_NAME} found in {start_path} or its parents. "
            "Run `trainwright init --data <path> --target <column>` first."
        )

    @classmethod
    def load(cls, root: Path | str) -> Project:
        root = Path(root).resolve()
        raw = yaml.safe_load((root / CONFIG_NAME).read_text()) or {}
        from pydantic import ValidationError

        try:
            cfg = ProjectConfig.model_validate(raw)
        except ValidationError as e:
            raise ConfigError(format_validation_error(e)) from e
        return cls(root=root, config=cfg)

    def save(self) -> None:
        header = (
            "# trainwright project config. Edit freely; validated on every command.\n"
            "# Changing data/split invalidates the frozen split: re-run `trainwright split`.\n"
        )
        (self.root / CONFIG_NAME).write_text(
            header + yaml.safe_dump(self.config.model_dump(), sort_keys=False)
        )

    # ----------------------------------------------------------------- paths
    @property
    def state_dir(self) -> Path:
        return self.root / ".trainwright"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def profile_path(self) -> Path:
        return self.state_dir / "profile.json"

    @property
    def split_path(self) -> Path:
        return self.state_dir / "split.json"

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / "test_ledger.jsonl"

    def benchmark_path(self, run_id: str) -> Path:
        return self.state_dir / "benchmark" / f"{run_id}.json"

    @property
    def data_path(self) -> Path:
        p = Path(self.config.data.path)
        return p if p.is_absolute() else (self.root / p).resolve()

    def ensure_dirs(self) -> None:
        for d in [self.state_dir, self.runs_dir, self.experiments_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def load_user_plugins(self) -> list[str]:
        """Import ``models.py`` / ``tw_plugins.py`` in the project root so custom
        ``@register_model`` definitions are available to the CLI."""
        import importlib.util
        import sys

        loaded = []
        for name in ("tw_plugins.py", "models.py"):
            path = self.root / name
            if path.exists():
                mod_name = f"_tw_user_{path.stem}"
                if mod_name in sys.modules:
                    continue
                spec = importlib.util.spec_from_file_location(mod_name, path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = mod
                    spec.loader.exec_module(mod)
                    loaded.append(str(path))
        return loaded
