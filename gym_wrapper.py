"""
Gymnasium-compatible wrapper for the Water Trash Collector environment.

Wraps the core WaterTrashEnvironment into a standard Gymnasium Env so it
can be used with stable-baselines3, RLlib, or any Gymnasium-compatible
training framework.

The agent's "camera" is represented by the observation vector:
  [robot_x, robot_y, robot_theta, nearest_trash_dist, nearest_trash_angle, trash_count]

This simulates what a real onboard camera + perception pipeline would provide:
the robot's position, heading, and where the nearest trash is relative to it.
"""

import math
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from server.water_trash_env_environment import WaterTrashEnvironment


class WaterTrashGymEnv(gym.Env):
    """
    Gymnasium wrapper for the Water Trash Collector.

    Observation Space (6-dim continuous):
        [robot_x, robot_y, robot_theta, nearest_trash_dist, nearest_trash_angle, trash_count_normalized]

    Action Space (2-dim continuous):
        [linear_velocity, angular_velocity]  both in [-1, 1]

    The 'camera' view is available via render() which produces an RGB array
    showing the water surface, robot, and trash from a top-down perspective.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, task_level="medium", render_mode=None, max_steps=200):
        super().__init__()
        self.task_level = task_level
        self.render_mode = render_mode
        self.max_steps = max_steps

        self._env = WaterTrashEnvironment()

        # Action: [linear_velocity, angular_velocity] in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation: [robot_x, robot_y, robot_theta, nearest_dist, nearest_angle, trash_count_norm]
        # Normalized to roughly [-1, 1] range for better training
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        self._pygame_screen = None
        self._clock = None

    def _obs_to_array(self, obs) -> np.ndarray:
        """Convert environment observation to normalized numpy array."""
        max_coord = 50.0  # Grid goes from -50 to +50
        max_dist = math.hypot(100, 100)  # Max possible distance ~141

        return np.array([
            obs.robot_x / max_coord,                         # Normalized X [-1, 1]
            obs.robot_y / max_coord,                         # Normalized Y [-1, 1]
            obs.robot_theta / math.pi,                       # Normalized theta [-1, 1]
            min(obs.nearest_trash_dist / max_dist, 1.0),     # Normalized distance [0, 1]
            obs.nearest_trash_angle / math.pi,               # Normalized angle [-1, 1]
            obs.trash_count / max(self._env.total_trash, 1), # Normalized count [0, 1]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            obs = self._env.reset(seed=seed, task_level=self.task_level)
        else:
            obs = self._env.reset(task_level=self.task_level)

        return self._obs_to_array(obs), {"task_level": self.task_level}

    def step(self, action):
        from models import WaterTrashAction

        lin_vel = float(np.clip(action[0], -1.0, 1.0))
        ang_vel = float(np.clip(action[1], -1.0, 1.0))

        act = WaterTrashAction(linear_velocity=lin_vel, angular_velocity=ang_vel)
        obs = self._env.step(act)

        reward = obs.reward if obs.reward else 0.0
        terminated = obs.done
        truncated = False
        info = obs.metadata if obs.metadata else {}

        return self._obs_to_array(obs), reward, terminated, truncated, info

    def render(self):
        """Render the environment as an RGB array (the agent's 'camera' view)."""
        import pygame

        WINDOW_SIZE = 400
        GRID_LIMIT = 50.0

        if self._pygame_screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._pygame_screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
                pygame.display.set_caption("Water Trash Collector — Camera View")
            else:
                self._pygame_screen = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            self._clock = pygame.time.Clock()

        def w2s(x, y):
            sx = int((x + GRID_LIMIT) / (2 * GRID_LIMIT) * WINDOW_SIZE)
            sy = int((GRID_LIMIT - y) / (2 * GRID_LIMIT) * WINDOW_SIZE)
            return sx, sy

        # Background — deep ocean blue
        self._pygame_screen.fill((15, 40, 70))

        # Grid lines
        for i in range(-int(GRID_LIMIT), int(GRID_LIMIT) + 1, 10):
            p1, p2 = w2s(i, -GRID_LIMIT), w2s(i, GRID_LIMIT)
            pygame.draw.line(self._pygame_screen, (25, 55, 90), p1, p2, 1)
            p1, p2 = w2s(-GRID_LIMIT, i), w2s(GRID_LIMIT, i)
            pygame.draw.line(self._pygame_screen, (25, 55, 90), p1, p2, 1)

        # Trash items — glowing green dots
        for t in self._env.trash_list:
            tx, ty = w2s(t["x"], t["y"])
            pygame.draw.circle(self._pygame_screen, (50, 200, 80), (tx, ty), 6)
            pygame.draw.circle(self._pygame_screen, (100, 255, 120), (tx, ty), 3)

        # Robot — white circle with red heading
        rx, ry = w2s(self._env.robot_x, self._env.robot_y)
        pygame.draw.circle(self._pygame_screen, (200, 200, 220), (rx, ry), 10)
        end_x = rx + 14 * math.cos(self._env.robot_theta)
        end_y = ry - 14 * math.sin(self._env.robot_theta)
        pygame.draw.line(self._pygame_screen, (255, 60, 60), (rx, ry), (end_x, end_y), 3)

        # Collection radius indicator
        col_r = int((self._env.COLLECTION_RADIUS / (2 * GRID_LIMIT)) * WINDOW_SIZE)
        pygame.draw.circle(self._pygame_screen, (100, 200, 255, 80), (rx, ry), col_r, 1)

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])

        # Return RGB array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._pygame_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self._pygame_screen is not None:
            import pygame
            pygame.quit()
            self._pygame_screen = None


# Register with Gymnasium
gym.register(
    id="WaterTrashCollector-v0",
    entry_point="gym_wrapper:WaterTrashGymEnv",
    kwargs={"task_level": "medium"},
)
