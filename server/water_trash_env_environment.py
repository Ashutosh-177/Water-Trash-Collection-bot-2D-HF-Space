"""
Water Surface Trash Collector — Environment Implementation.

A 2D kinematic robot navigates a 100×100 water surface to collect
floating trash items. Supports three difficulty levels (easy, medium, hard).
"""

import math
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State

    from ..models import WaterTrashAction, WaterTrashObservation
except ImportError:
    try:
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import State
    except ImportError:
        from openenv_core.env_server.interfaces import Environment
        from openenv_core.env_server.types import State

    from models import WaterTrashAction, WaterTrashObservation

# Required for OpenEnv Grader Support
from openenv.core.rubrics.base import Rubric

class WaterTrashGrader(Rubric):
    """Parses observation metadata to generate a valid strictly (0, 1) score."""
    def forward(self, action: Any, observation: Any) -> float:
        # Get score strictly from [0.0, 1.0] cumulative reward
        score = getattr(observation, 'reward', 0.0)
        if hasattr(observation, 'metadata') and 'cumulative_reward' in observation.metadata:
            score = observation.metadata['cumulative_reward']
            
        # Ensure it is STRICTLY between 0 and 1
        return max(0.01, min(0.99, float(score)))


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy": {
        "total_trash": 1,
        "trash_list": [{"x": 20.0, "y": 20.0}],
        "drift_velocity_x": 0.0,
        "description": "Single trash at (20, 20). Robot starts at origin.",
    },
    "medium": {
        "total_trash": 5,
        "trash_list": None,  # generated randomly at reset
        "drift_velocity_x": 0.0,
        "description": "5 trash items scattered in the upper-right quadrant.",
    },
    "hard": {
        "total_trash": 10,
        "trash_list": None,  # generated randomly at reset
        "drift_velocity_x": -0.2,
        "description": "10 trash items with a -0.2 x-drift water current.",
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Return the task configuration for the given *task_id*.

    Valid IDs: ``"easy"``, ``"medium"``, ``"hard"``.
    """
    task_id = task_id.lower()
    if task_id not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Choose from: {list(_TASK_REGISTRY)}"
        )
    return dict(_TASK_REGISTRY[task_id])  # shallow copy


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class WaterTrashEnvironment(Environment):
    """
    OpenEnv environment for a 2D water-surface trash collector.

    The robot is a circle-based agent in a 100×100 coordinate system.
    Trash items are (x, y) point objects; collection occurs when the
    robot comes within 2.0 units of a point.

    Kinematics
    ----------
    θ  += angular_velocity * 0.5          (max 0.5 rad / step)
    x  += linear_velocity  * 5.0 * cos(θ) + drift_x
    y  += linear_velocity  * 5.0 * sin(θ)

    Reward (per step, episodic sum ≤ 1.0)
    -------------------------------------
    +1.0 / total_trash   per trash collected
    +0.001               if distance to nearest trash decreased (shaping)

    Episode ends after 200 steps or when all trash is collected.
    """

    MAX_STEPS = 400
    COLLECTION_RADIUS = 3.5
    MAX_LINEAR_SPEED = 0.9
    MAX_ANGULAR_SPEED = 0.2

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_level: str = "easy"
        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.robot_theta: float = 0.0
        self.trash_list: List[Dict[str, float]] = []
        self.total_trash: int = 0
        self.collected_count: int = 0
        self.drift_velocity_x: float = 0.0
        self.prev_nearest_dist: float = 0.0
        self.cumulative_reward: float = 0.0
        
        # Attach OpenEnv compatible rubric grader for hackathon scoring
        self.rubric = WaterTrashGrader()

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WaterTrashObservation:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        episode_id : str, optional
            Custom episode identifier.
        **kwargs
            ``task_level`` (str): one of ``"easy"``, ``"medium"``, ``"hard"``.
        """
        if seed is not None:
            random.seed(seed)

        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)

        self.task_level = kwargs.get("task_level", "easy").lower()
        task = get_task(self.task_level)

        self.total_trash = task["total_trash"]
        self.drift_velocity_x = task["drift_velocity_x"]
        self.collected_count = 0
        self.cumulative_reward = 0.0

        # Robot always starts at the origin facing right
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # Trash placement
        if task["trash_list"] is not None:
            # Fixed placement (easy)
            self.trash_list = [dict(t) for t in task["trash_list"]]
        elif self.task_level == "medium":
            self.trash_list = [
                {"x": random.uniform(10, 40), "y": random.uniform(10, 40)}
                for _ in range(self.total_trash)
            ]
        elif self.task_level == "hard":
            self.trash_list = [
                {"x": random.uniform(-40, 40), "y": random.uniform(-40, 40)}
                for _ in range(self.total_trash)
            ]

        self.prev_nearest_dist = self._nearest_trash_dist()
        return self._make_obs(reward=0.0, done=False)

    # ------------------------------------------------------------------- step
    def step(
        self,
        action: WaterTrashAction,  # type: ignore[override]
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> WaterTrashObservation:
        """Execute one step in the environment."""
        self._state.step_count += 1

        # 1. Clamp & scale inputs
        lin_vel = max(-1.0, min(1.0, action.linear_velocity)) * self.MAX_LINEAR_SPEED
        ang_vel = max(-1.0, min(1.0, action.angular_velocity)) * self.MAX_ANGULAR_SPEED

        # 2. Kinematic update + environmental drift
        self.robot_theta += ang_vel
        self.robot_x += lin_vel * math.cos(self.robot_theta) + self.drift_velocity_x
        self.robot_y += lin_vel * math.sin(self.robot_theta)

        # 3. Trash collection
        step_reward = 0.0
        for trash in self.trash_list[:]:  # iterate over a copy
            dist = math.hypot(self.robot_x - trash["x"], self.robot_y - trash["y"])
            if dist <= self.COLLECTION_RADIUS:
                self.trash_list.remove(trash)
                self.collected_count += 1
                step_reward += 1.0 / self.total_trash

        # 4. Dense shaping reward (approach nearest trash)
        current_nearest = self._nearest_trash_dist()
        if current_nearest < self.prev_nearest_dist:
            step_reward += 0.001
        self.prev_nearest_dist = current_nearest

        # 5. Accumulate & clamp to [0, 1]
        self.cumulative_reward = min(self.cumulative_reward + step_reward, 1.0)
        step_reward = min(step_reward, 1.0)

        # 6. Termination
        done = len(self.trash_list) == 0 or self._state.step_count >= self.MAX_STEPS

        return self._make_obs(reward=step_reward, done=done)

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> State:
        """Return the current environment state."""
        return self._state

    # --------------------------------------------------------------- helpers
    def _nearest_trash_dist(self) -> float:
        if not self.trash_list:
            return 0.0
        return min(
            math.hypot(self.robot_x - t["x"], self.robot_y - t["y"])
            for t in self.trash_list
        )

    def _nearest_trash_angle(self) -> float:
        if not self.trash_list:
            return 0.0
        nearest = min(
            self.trash_list,
            key=lambda t: math.hypot(self.robot_x - t["x"], self.robot_y - t["y"]),
        )
        abs_angle = math.atan2(
            nearest["y"] - self.robot_y, nearest["x"] - self.robot_x
        )
        # Return relative angle w.r.t. robot heading
        rel = abs_angle - self.robot_theta
        # Normalise to (-π, π]
        rel = (rel + math.pi) % (2 * math.pi) - math.pi
        return rel

    def _make_obs(self, *, reward: float, done: bool) -> WaterTrashObservation:
        return WaterTrashObservation(
            robot_x=round(self.robot_x, 4),
            robot_y=round(self.robot_y, 4),
            robot_theta=round(self.robot_theta, 4),
            nearest_trash_dist=round(self._nearest_trash_dist(), 4),
            nearest_trash_angle=round(self._nearest_trash_angle(), 4),
            trash_count=len(self.trash_list),
            done=done,
            reward=round(reward, 6),
            metadata={
                "step": self._state.step_count,
                "collected": self.collected_count,
                "total_trash": self.total_trash,
                "cumulative_reward": round(self.cumulative_reward, 6),
                "task_level": self.task_level,
            },
        )
