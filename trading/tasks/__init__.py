"""Trading Environment Tasks and Graders.

Provides 3 tasks with increasing difficulty and programmatic graders
that score agent performance on a 0.0–1.0 scale.
"""

from .task_definitions import TASKS, get_task, TaskConfig
from .graders import grade_trajectory

__all__ = [
    "TASKS",
    "get_task",
    "TaskConfig",
    "grade_trajectory",
]
