"""trainwright: an agent-first PyTorch framework for training small, task-specific models.

Python API (Lightning-style)::

    import trainwright as tw

    task = tw.ClassificationTask(model, n_classes=3)
    trainer = tw.Trainer(max_epochs=20, monitor="val_accuracy", mode="max", patience=5)
    trainer.fit(task, train_loader, val_loader)

Project lifecycle (what agents should use): see ``trainwright guide`` / the CLI.
"""

__version__ = "0.5.4"

from ._utils import seed_everything
from .config import ExperimentConfig, ProjectConfig, default_experiment, resolve_experiment
from .data.modules import DataModule, TensorLoader
from .models import build_model, list_models, register_model
from .project import Project
from .task import ClassificationTask, RegressionTask, Task
from .tracking import Run
from .trainer import Callback, EarlyStopping, Trainer

__all__ = [
    "Callback",
    "ClassificationTask",
    "DataModule",
    "EarlyStopping",
    "ExperimentConfig",
    "Project",
    "ProjectConfig",
    "RegressionTask",
    "Run",
    "Task",
    "TensorLoader",
    "Trainer",
    "__version__",
    "build_model",
    "default_experiment",
    "list_models",
    "register_model",
    "resolve_experiment",
    "seed_everything",
]
