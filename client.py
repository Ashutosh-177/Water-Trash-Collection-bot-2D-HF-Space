"""Water Surface Trash Collector Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import WaterTrashAction, WaterTrashObservation


class WaterTrashEnv(
    EnvClient[WaterTrashAction, WaterTrashObservation, State]
):
    """
    Client for the Water Surface Trash Collector Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with WaterTrashEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     result = client.step(WaterTrashAction(linear_velocity=1.0, angular_velocity=0.0))
        ...     print(result.observation.trash_count)
    """

    def _step_payload(self, action: WaterTrashAction) -> Dict:
        """Convert WaterTrashAction to JSON payload."""
        return {
            "linear_velocity": action.linear_velocity,
            "angular_velocity": action.angular_velocity,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WaterTrashObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = WaterTrashObservation(
            robot_x=obs_data.get("robot_x", 0.0),
            robot_y=obs_data.get("robot_y", 0.0),
            robot_theta=obs_data.get("robot_theta", 0.0),
            nearest_trash_dist=obs_data.get("nearest_trash_dist", 0.0),
            nearest_trash_angle=obs_data.get("nearest_trash_angle", 0.0),
            trash_count=obs_data.get("trash_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
