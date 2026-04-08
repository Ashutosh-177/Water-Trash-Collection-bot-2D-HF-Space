---
title: Water Trash Collector
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🌊 Water Surface Trash Collector

An OpenEnv-compliant reinforcement learning environment where an autonomous robot navigates a 2D water surface to collect floating trash.

## Environment Description

A circular cleaning drone operates on a 100×100 coordinate water surface. The robot uses its onboard camera (observation vector) to detect the nearest trash item's distance and relative angle, then navigates to collect it.

### Observation Space
| Field | Description |
|-------|-------------|
| `robot_x` | Robot X position |
| `robot_y` | Robot Y position |
| `robot_theta` | Robot heading (radians) |
| `nearest_trash_dist` | Distance to nearest trash |
| `nearest_trash_angle` | Relative angle to nearest trash |
| `trash_count` | Remaining trash items |

### Action Space
| Field | Range | Description |
|-------|-------|-------------|
| `linear_velocity` | [-1.0, 1.0] | Forward/backward thrust |
| `angular_velocity` | [-1.0, 1.0] | Turning rate |

### Difficulty Levels
| Level | Trash Count | Features |
|-------|-------------|----------|
| **Easy** | 1 | Fixed position at (20, 20) |
| **Medium** | 5 | Random positions in upper-right quadrant |
| **Hard** | 10 | Random positions across full map + water drift |

## Built With
- [OpenEnv](https://github.com/openenv) — Environment framework
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — RL training
- [Gradio](https://gradio.app/) — Web interface
