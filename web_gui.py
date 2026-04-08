#!/usr/bin/env python3
"""
Gradio-based Web GUI for the Water Trash Collector — runs on HuggingFace Spaces.
Renders the environment visually using PIL and provides interactive controls.
"""

import math
import os
import sys
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models import WaterTrashAction
from server.water_trash_env_environment import WaterTrashEnvironment

# ──────────── CONFIG ────────────
IMG_SIZE = 600
GRID_LIMIT = 50.0
TRASH_COLORS = [
    (220, 80, 80), (200, 180, 60), (180, 100, 220),
    (80, 200, 120), (240, 140, 60), (100, 180, 240),
]

# Global environment instance
env = WaterTrashEnvironment()
obs = None
frame_count = 0


def w2p(x, y):
    """World coords to pixel coords."""
    px = int((x + GRID_LIMIT) / (2 * GRID_LIMIT) * IMG_SIZE)
    py = int((GRID_LIMIT - y) / (2 * GRID_LIMIT) * IMG_SIZE)
    return px, py


def render_frame():
    """Render the current environment state as a PIL Image."""
    global frame_count
    frame_count += 1

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (8, 28, 56))
    draw = ImageDraw.Draw(img)

    # Grid
    for i in range(-int(GRID_LIMIT), int(GRID_LIMIT) + 1, 10):
        p1 = w2p(i, -GRID_LIMIT)
        p2 = w2p(i, GRID_LIMIT)
        draw.line([p1, p2], fill=(22, 50, 85), width=1)
        p1 = w2p(-GRID_LIMIT, i)
        p2 = w2p(GRID_LIMIT, i)
        draw.line([p1, p2], fill=(22, 50, 85), width=1)

    # Trash items
    for i, t in enumerate(env.trash_list):
        tx, ty = w2p(t["x"], t["y"])
        c = TRASH_COLORS[i % len(TRASH_COLORS)]

        # Glow ring
        draw.ellipse([tx - 14, ty - 14, tx + 14, ty + 14],
                      outline=(c[0] // 3, c[1] // 3, c[2] // 3), width=1)

        # Trash body — polygon
        n_sides = 5 + (i % 3)
        rot = frame_count * 0.02 + i * 1.2
        pts = []
        for j in range(n_sides):
            a = rot + 2 * math.pi * j / n_sides
            r = 8 * (0.7 + 0.3 * math.sin(a * 3 + i))
            pts.append((tx + int(r * math.cos(a)), ty + int(r * math.sin(a))))
        draw.polygon(pts, fill=c, outline=(min(c[0]+60, 255), min(c[1]+60, 255), min(c[2]+60, 255)))

        # Highlight dot
        draw.ellipse([tx - 2, ty - 4, tx + 1, ty - 1], fill=(255, 255, 255))

    # Robot
    rx, ry = w2p(env.robot_x, env.robot_y)
    th = env.robot_theta
    ct, st = math.cos(th), math.sin(th)

    # Scanning ring
    scan_r = int(20 + 4 * math.sin(frame_count * 0.15))
    draw.ellipse([rx - scan_r, ry - scan_r, rx + scan_r, ry + scan_r],
                  outline=(40, 160, 255, 60), width=1)

    # Collector arms
    for side in [1, -1]:
        arm_a = th + side * 0.5
        ax1 = rx + int(12 * math.cos(arm_a))
        ay1 = ry - int(12 * math.sin(arm_a))
        ax2 = rx + int(24 * math.cos(arm_a))
        ay2 = ry - int(24 * math.sin(arm_a))
        draw.line([(ax1, ay1), (ax2, ay2)], fill=(180, 200, 220), width=3)
        # Claw
        claw_a = arm_a + side * 0.2
        cx = ax2 + int(6 * math.cos(claw_a))
        cy = ay2 - int(6 * math.sin(claw_a))
        draw.line([(ax2, ay2), (cx, cy)], fill=(200, 220, 240), width=2)

    # Body
    draw.ellipse([rx - 14, ry - 14, rx + 14, ry + 14], fill=(50, 55, 70))
    draw.ellipse([rx - 12, ry - 12, rx + 12, ry + 12], fill=(70, 80, 100))
    draw.ellipse([rx - 9, ry - 9, rx + 9, ry + 9], fill=(90, 100, 120))

    # Eye
    eye_p = int(180 + 75 * math.sin(frame_count * 0.2))
    draw.ellipse([rx - 4, ry - 4, rx + 4, ry + 4], fill=(30, 40, 55))
    draw.ellipse([rx - 3, ry - 3, rx + 3, ry + 3], fill=(0, eye_p, 255))

    # Heading arrow
    ant_x = rx + int(22 * ct)
    ant_y = ry - int(22 * st)
    draw.line([(rx + int(12 * ct), ry - int(12 * st)), (ant_x, ant_y)],
              fill=(100, 255, 180), width=2)
    # Arrow tip
    for side in [-0.5, 0.5]:
        bx = ant_x - int(5 * math.cos(th + side))
        by = ant_y + int(5 * math.sin(th + side))
        draw.line([(ant_x, ant_y), (bx, by)], fill=(100, 255, 180), width=2)

    # Thruster glow
    rear_x = rx - int(14 * ct)
    rear_y = ry + int(14 * st)
    tr = int(3 + 2 * abs(math.sin(frame_count * 0.3)))
    draw.ellipse([rear_x - tr, rear_y - tr, rear_x + tr, rear_y + tr],
                  fill=(80, 180, 255))

    # HUD overlay text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_sm = font

    # HUD background
    draw.rectangle([4, 4, 220, 120], fill=(6, 14, 30, 200))

    y_pos = 8
    hud = [
        (f"Step: {env.state.step_count}/{env.MAX_STEPS}", (210, 225, 250)),
        (f"Level: {env.task_level.upper()}", (45, 160, 255)),
        (f"Trash: {obs.trash_count}/{env.total_trash}" if obs else "Trash: ?",
         (80, 255, 140) if obs and obs.trash_count == 0 else (255, 90, 80)),
        (f"Reward: {env.cumulative_reward:.3f}", (80, 255, 140)),
        (f"Dist: {obs.nearest_trash_dist:.1f}" if obs else "Dist: ?", (80, 105, 145)),
    ]
    for text, color in hud:
        draw.text((10, y_pos), text, fill=color, font=font)
        y_pos += 20

    # Done banner
    if obs and obs.done:
        draw.rectangle([0, IMG_SIZE // 2 - 25, IMG_SIZE, IMG_SIZE // 2 + 25],
                        fill=(0, 0, 0))
        if env.collected_count == env.total_trash:
            msg = f"ALL COLLECTED! Reward: {env.cumulative_reward:.3f}"
            color = (80, 255, 140)
        else:
            msg = f"TIME UP  {env.collected_count}/{env.total_trash} collected"
            color = (255, 90, 80)
        draw.text((IMG_SIZE // 2 - len(msg) * 4, IMG_SIZE // 2 - 8), msg,
                  fill=color, font=font)

    return img


def reset_env(task_level):
    """Reset the environment."""
    global obs
    obs = env.reset(task_level=task_level)
    status = f"Reset to {task_level.upper()} | Trash: {obs.trash_count}"
    return render_frame(), status


def step_env(linear_vel, angular_vel):
    """Step the environment with given action."""
    global obs
    if obs is None:
        obs = env.reset(task_level="medium")

    if not obs.done:
        action = WaterTrashAction(
            linear_velocity=float(linear_vel),
            angular_velocity=float(angular_vel)
        )
        obs = env.step(action)

    status = (f"Step {env.state.step_count} | "
              f"Trash: {obs.trash_count}/{env.total_trash} | "
              f"Reward: {env.cumulative_reward:.3f}")
    if obs.done:
        status += f" | DONE ({env.collected_count}/{env.total_trash} collected)"
    return render_frame(), status


def ai_step():
    """One step of the deterministic AI policy."""
    global obs
    if obs is None:
        obs = env.reset(task_level="medium")

    if obs.done:
        return render_frame(), "Episode done. Click Reset."

    angle = obs.nearest_trash_angle
    if abs(angle) < 0.15:
        move, turn = 1.0, 0.0
    elif angle > 0:
        move, turn = 0.4, 1.0
    else:
        move, turn = 0.4, -1.0

    return step_env(move, turn)


ai_running = False


def toggle_autoplay(task_level):
    """Start or stop AI auto-play."""
    global obs, ai_running
    if not ai_running:
        ai_running = True
        obs = env.reset(task_level=task_level)
        return "⏹ Stop AI", f"AI started | Level: {task_level.upper()}", render_frame()
    else:
        ai_running = False
        return "🚀 Auto-Play Episode", "AI stopped.", render_frame()


def timer_tick(task_level):
    """Called every 150 ms by gr.Timer. Steps the AI if running."""
    global obs
    if not ai_running:
        return render_frame(), gr.update()

    if obs is None:
        obs = env.reset(task_level=task_level)

    if obs.done:
        return render_frame(), f"DONE! Collected {env.collected_count}/{env.total_trash} | Reward: {env.cumulative_reward:.3f}"

    img, status = ai_step()
    return img, status


# ──────────── GRADIO UI ────────────

with gr.Blocks(
    title="🌊 Water Trash Collector",
    css="""
    .gradio-container { max-width: 1000px !important; }
    #env-image { border: 2px solid #1e3a5f; border-radius: 8px; }
    .status-bar { font-family: monospace; font-size: 14px; }
    """
) as demo:

    gr.Markdown("""
    # 🌊 Water Surface Trash Collector
    **An RL environment where an autonomous robot collects floating trash from water.**

    The robot (blue circle with arms) uses its camera sensors to detect trash (colored shapes) and navigates to collect them.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            env_image = gr.Image(
                label="Environment View",
                elem_id="env-image",
                type="pil",
                height=IMG_SIZE,
                width=IMG_SIZE,
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes="status-bar",
            )

        with gr.Column(scale=1):
            gr.Markdown("### Controls")

            task_dd = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="Difficulty",
            )
            reset_btn = gr.Button("🔄 Reset", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### Manual Control")

            lin_slider = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Linear Velocity")
            ang_slider = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Angular Velocity")
            step_btn = gr.Button("▶ Step", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### AI Agent")
            ai_step_btn = gr.Button("🤖 AI Step", variant="secondary")
            ai_play_btn = gr.Button("🚀 Auto-Play Episode", variant="primary")

    # Timer fires every 150 ms and drives animation when AI is running
    timer = gr.Timer(value=0.15, active=True)
    timer.tick(fn=timer_tick, inputs=[task_dd], outputs=[env_image, status_text])

    # Event handlers
    reset_btn.click(fn=reset_env, inputs=[task_dd], outputs=[env_image, status_text])
    step_btn.click(fn=step_env, inputs=[lin_slider, ang_slider], outputs=[env_image, status_text])
    ai_step_btn.click(fn=ai_step, inputs=[], outputs=[env_image, status_text])
    ai_play_btn.click(
        fn=toggle_autoplay,
        inputs=[task_dd],
        outputs=[ai_play_btn, status_text, env_image],
    )

    # Initialize on load
    demo.load(fn=lambda: reset_env("medium"), outputs=[env_image, status_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)