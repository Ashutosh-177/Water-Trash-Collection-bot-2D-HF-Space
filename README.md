<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0c2444,50:1a5276,100:2d8cf0&height=220&section=header&text=🌊%20Water%20Trash%20Collector%20Bot&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Autonomous%20RL%20Agent%20for%20Ocean%20Cleanup&descSize=18&descAlignY=55&descColor=a8d8ff" width="100%"/>

<!-- Typing Animation -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=2D8CF0&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=100&lines=🤖+Training+autonomous+trash+collectors;🌊+Reinforcement+Learning+on+water;📷+Camera-based+navigation+system;🧠+PPO+Agent+with+100%25+collection+rate" alt="Typing SVG" /></a>

<br/>

<!-- Badges -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-PPO-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-00C853?style=for-the-badge&logo=openaigym&logoColor=white)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗_Live_Demo-HF_Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/ashu-17/water-trash-collector)
[![Pygame](https://img.shields.io/badge/Pygame-GUI-00AA00?style=for-the-badge&logo=pygame&logoColor=white)](https://pygame.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

<!-- Star/Fork Badges -->
<img src="https://img.shields.io/github/stars/Ashutosh-177/Water-Trash-Collection-bot-2D-HF-Space?style=social" alt="Stars"/>
<img src="https://img.shields.io/github/forks/Ashutosh-177/Water-Trash-Collection-bot-2D-HF-Space?style=social" alt="Forks"/>
<img src="https://img.shields.io/github/watchers/Ashutosh-177/Water-Trash-Collection-bot-2D-HF-Space?style=social" alt="Watchers"/>

</div>

<!-- Wave Separator -->
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🎬 Demo

<div align="center">

```
    🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊
    🌊                                                🌊
    🌊    🤖 ──────────►  🗑️   COLLECTED! ✅         🌊
    🌊         robot        trash    +1.0 reward       🌊
    🌊                                                🌊
    🌊    📷 Camera ─► 🧠 PPO Agent ─► 🎮 Action      🌊
    🌊                                                🌊
    🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊
```

**▶ [Try Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/ashu-17/water-trash-collector)**

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🎯 What is this?

> **An end-to-end Reinforcement Learning project** where an autonomous robot learns to navigate a 2D water surface, detect floating trash using its camera, and collect every piece — trained entirely through self-play using PPO.

<div align="center">

```mermaid
graph LR
    A["🌊 Water\nEnvironment"] -->|observe| B["📷 Camera\nSensor"]
    B -->|6D vector| C["🧠 PPO\nAgent"]
    C -->|velocity + turn| D["🎮 Robot\nAction"]
    D -->|move| A
    A -->|"collect!"| E["✅ Reward\n+1.0"]
    
    style A fill:#0c2444,stroke:#2d8cf0,color:#fff
    style B fill:#1a3a5c,stroke:#2d8cf0,color:#fff
    style C fill:#1e4d2b,stroke:#50c850,color:#fff
    style D fill:#4d3a1e,stroke:#f0a02d,color:#fff
    style E fill:#1e4d1e,stroke:#50ff8a,color:#fff
```

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🏆 Training Results

<div align="center">

The agent achieves **100% collection rate** across ALL difficulty levels!

| | Difficulty | Trash Items | Special Features | Collection | Reward | Steps |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🟢 | **Easy** | 1 | Fixed position | **✅ 100%** | 1.059 | 60 |
| 🟡 | **Medium** | 5 | Random spawn | **✅ 100%** | 1.032 | 36 |
| 🔴 | **Hard** | 10 | Random + Drift 🌊 | **✅ 100%** | 1.317 | 330 |

</div>

### 📈 Training Progress
```
Timestep    Reward    Status
─────────────────────────────────────────
     0K     0.12     ❌ Random exploration
   100K     0.78     🔄 Learning to navigate  
   300K     1.02     ✅ Collecting all trash
   500K     1.03     ⚡ Path optimization
 1,000K     1.32     🏆 HARD MODE MASTERED!
─────────────────────────────────────────
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🏗️ Project Architecture

<div align="center">

```mermaid
graph TB
    subgraph "🌐 Deployment Layer"
        HF["🤗 HuggingFace Spaces"]
        DOCKER["🐳 Docker Container"]
    end
    
    subgraph "🎮 Interface Layer"
        PYGAME["🖥️ PyGame Desktop GUI"]
        GRADIO["🌐 Gradio Web GUI"]
        API["📡 OpenEnv REST API"]
    end
    
    subgraph "🧠 Intelligence Layer"  
        PPO["🏋️ PPO Agent (SB3)"]
        LLM["🤖 Gemini LLM Agent"]
        FALLBACK["📐 Math Fallback"]
    end
    
    subgraph "🌊 Environment Layer"
        ENV["🌊 Water Surface Env"]
        PHYSICS["⚙️ 2D Kinematics"]
        CAMERA["📷 Camera System"]
    end
    
    HF --> DOCKER
    DOCKER --> GRADIO
    DOCKER --> API
    PYGAME --> ENV
    GRADIO --> ENV
    API --> ENV
    PPO --> ENV
    LLM --> API
    FALLBACK --> ENV
    ENV --> PHYSICS
    ENV --> CAMERA
    
    style HF fill:#FFD21E,stroke:#333,color:#333
    style DOCKER fill:#2496ED,stroke:#333,color:#fff
    style PYGAME fill:#00AA00,stroke:#333,color:#fff
    style GRADIO fill:#FF7C00,stroke:#333,color:#fff
    style API fill:#009688,stroke:#333,color:#fff
    style PPO fill:#EE4C2C,stroke:#333,color:#fff
    style LLM fill:#4285F4,stroke:#333,color:#fff
    style ENV fill:#0c2444,stroke:#2d8cf0,color:#fff
```

</div>

### 📂 File Structure

```
🌊 water_trash_env/
│
├── 🧠 Core
│   ├── server/water_trash_env_environment.py   # Environment physics & logic
│   ├── models.py                               # OpenEnv Action/Observation models
│   └── gym_wrapper.py                          # Gymnasium wrapper for RL
│
├── 🎮 Interfaces
│   ├── gui.py                                  # PyGame desktop GUI + camera feed
│   ├── web_gui.py                              # Gradio web GUI (HF Spaces)
│   └── server/app.py                           # FastAPI OpenEnv server
│
├── 🤖 Agents
│   ├── train.py                                # PPO training pipeline (SB3)
│   ├── inference.py                            # Gemini LLM + fallback agent
│   └── client.py                               # Typed OpenEnv client
│
├── 📦 Config
│   ├── Dockerfile                              # HF Spaces deployment
│   ├── pyproject.toml                          # Python project config
│   └── openenv.yaml                            # OpenEnv manifest
│
└── 🏋️ Models
    └── trained_models/
        ├── best_model.zip                      # Best training checkpoint
        └── final_model.zip                     # Final trained model
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🎮 Environment Details

<div align="center">

### 📷 Observation Space (Camera Input)

</div>

The robot's onboard camera provides a **6-dimensional** observation:

```python
observation = {
    "robot_x":              # 📍 Robot X position (normalized to grid)
    "robot_y":              # 📍 Robot Y position (normalized to grid)  
    "robot_theta":          # 🧭 Heading angle (radians)
    "nearest_trash_dist":   # 📏 Distance to nearest trash (camera detection)
    "nearest_trash_angle":  # 🎯 Relative angle to nearest trash
    "trash_count":          # 🔢 Remaining targets detected by sensor
}
```

### 🎮 Action Space

```python
action = {
    "linear_velocity":  [-1.0, 1.0]    # ⬆️⬇️ Forward/backward thrust
    "angular_velocity": [-1.0, 1.0]    # ↩️↪️ Turning rate
}
```

### 🏅 Reward Design

| Event | Reward | Purpose |
|:-----:|:------:|:-------:|
| 🗑️ Collect trash | `+1.0 / total` | Per-item collection bonus |
| 📏 Get closer | `+0.001` | Shaping: encourage approach |
| ⏰ Time limit | 400 steps max | Encourage efficiency |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🚀 Quick Start

### 📥 1. Clone & Install

```bash
git clone https://github.com/Ashutosh-177/Water-Trash-Collection-bot-2D-HF-Space.git
cd Water-Trash-Collection-bot-2D-HF-Space
pip install -e ".[dev]"
```

### 🖥️ 2. Desktop GUI (PyGame)

```bash
# 🤖 Watch AI agent play
python gui.py --ai --task medium

# 🎮 Manual control
python gui.py --task hard
```

<div align="center">

| Key | Action | | Key | Action |
|:---:|:------:|:---:|:---:|:------:|
| `↑` `↓` `←` `→` | Move robot | | `A` | Toggle AI/Manual |
| `1` / `2` / `3` | Easy / Med / Hard | | `R` | Reset episode |
| `Q` | Quit | | `C` | Toggle camera |

</div>

### 🌐 3. Web GUI (Gradio)

```bash
python web_gui.py
# 🔗 Open http://localhost:7860
```

### 🧠 4. Train Your Own Agent

```bash
# Train on medium (5 trash items)
python train.py --task medium --steps 500000

# Train on hard (10 trash + water drift) 
python train.py --task hard --steps 1000000

# Evaluate trained model
python train.py --eval --task hard
```

### 📡 5. OpenEnv Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🌐 Live Demo & API

<div align="center">

### ▶️ [Launch Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/ashu-17/water-trash-collector)

</div>

Connect programmatically:

```python
from water_trash_env import WaterTrashAction, WaterTrashEnv

with WaterTrashEnv.from_env("ashu-17/water-trash-collector") as env:
    obs = await env.reset(task_level="hard")
    
    while not obs.done:
        action = WaterTrashAction(linear_velocity=1.0, angular_velocity=0.0)
        obs = await env.step(action)
        print(f"Trash remaining: {obs.trash_count}")
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 🔧 Tech Stack

<div align="center">

| | Technology | Purpose |
|:---:|:---:|:---:|
| 🧠 | **PPO (Stable-Baselines3)** | Reinforcement Learning Algorithm |
| 🏋️ | **Gymnasium** | Environment Wrapper & Interface |
| 📡 | **OpenEnv + FastAPI** | REST/WebSocket API Server |
| 🎮 | **Pygame** | Desktop GUI with Camera Feed |
| 🌐 | **Gradio** | Web Interface (HF Spaces) |
| 🤖 | **Google Gemini** | LLM Inference Agent |
| 🐳 | **Docker** | Containerized Deployment |
| 🤗 | **HuggingFace Spaces** | Cloud Hosting |
| 🐍 | **Pydantic** | Type-safe Data Models |

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

## 👤 Author

<div align="center">

**Ashutosh**

[![GitHub](https://img.shields.io/badge/GitHub-Ashutosh--177-181717?style=for-the-badge&logo=github)](https://github.com/Ashutosh-177)

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%"/>

<div align="center">

### ⭐ Star this repo if you found it interesting!

*Built with ❤️ for cleaner oceans, one pixel at a time* 🌊🤖

</div>

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0c2444,50:1a5276,100:2d8cf0&height=120&section=footer" width="100%"/>
