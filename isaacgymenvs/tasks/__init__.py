"""Task registry."""

from isaacgymenvs.tasks.go2_bridge import Go2Bridge

TASK_REGISTRY = {
    "Go2Bridge": Go2Bridge,
}


def make_task(task_name: str, cfg: dict):
    if task_name not in TASK_REGISTRY:
        valid = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {valid}")
    return TASK_REGISTRY[task_name](cfg)
