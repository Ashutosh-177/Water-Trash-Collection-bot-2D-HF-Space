"""
Data models for the Water Surface Trash Collector Environment.

Defines the Action and Observation types for a 2D kinematic robot
that collects trash floating on a water surface.
"""

from typing import List

from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv_core.env_server.types import Action, Observation


class WaterTrashAction(Action):
    """
    Action for the Water Trash Collector environment.

    The robot is controlled via two continuous signals:
      - linear_velocity:  forward/backward thrust, clamped to [-1.0, 1.0]
      - angular_velocity: turning rate, clamped to [-1.0, 1.0]
    """

    linear_velocity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Forward/backward thrust in [-1.0, 1.0]",
    )
    angular_velocity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Turning rate in [-1.0, 1.0]",
    )


class WaterTrashObservation(Observation):
    """
    Observation from the Water Trash Collector environment.

    The observation vector contains:
      [robot_x, robot_y, robot_theta,
       nearest_trash_dist, nearest_trash_angle,
       trash_count]
    """

    robot_x: float = Field(default=0.0, description="Robot X position")
    robot_y: float = Field(default=0.0, description="Robot Y position")
    robot_theta: float = Field(default=0.0, description="Robot heading (radians)")
    nearest_trash_dist: float = Field(
        default=0.0, description="Distance to the nearest trash item"
    )
    nearest_trash_angle: float = Field(
        default=0.0, description="Relative angle to the nearest trash item (radians)"
    )
    trash_count: int = Field(
        default=0, description="Number of trash items remaining"
    )
