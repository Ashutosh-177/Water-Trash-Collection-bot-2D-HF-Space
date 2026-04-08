#!/usr/bin/env python3
"""
🌊 Water Surface Trash Collector — Gradio HF Spaces Version
Renders each frame as a numpy image via pygame offscreen surface.
No display required — works on headless HF Spaces servers.
"""

import math
import os
import sys
import random
import numpy as np

# ── Headless pygame setup (MUST be before pygame import) ──
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["SDL_AUDIODRIVER"] = "dummy"

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    raise ImportError("Run: pip install pygame")

try:
    import gradio as gr
except ImportError:
    raise ImportError("Run: pip install gradio")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models import WaterTrashAction
from server.water_trash_env_environment import WaterTrashEnvironment

# ──────────────── LAYOUT ────────────────
MAP_SIZE    = 600
PANEL_W     = 280
CAMERA_SIZE = 240
WIN_W       = MAP_SIZE + PANEL_W
WIN_H       = MAP_SIZE
GRID_LIMIT  = 50.0

# ──────────────── COLORS ────────────────
OCEAN_DEEP    = (8,  28,  56)
OCEAN_MID     = (14, 42,  78)
GRID_COL      = (22, 50,  85)
TRASH_PALETTE = [
    (220, 80,  80),  (200, 180, 60),  (180, 100, 220),
    (80,  200, 120), (240, 140, 60),  (100, 180, 240),
]
PANEL_BG      = (6,  14,  30)
PANEL_BORDER  = (30, 70,  130)
ACCENT        = (45, 160, 255)
TEXT_BRIGHT   = (210, 225, 250)
TEXT_DIM      = (80, 105, 145)
SUCCESS       = (80, 255, 140)
DANGER        = (255, 90,  80)
WARNING       = (255, 200, 60)
PROGRESS_BG   = (18, 35,  60)
PROGRESS_FIL  = (45, 200, 130)
CAM_WATER     = (10, 30,  60)
CAM_GRID      = (18, 40,  70)
CAM_BORDER    = (40, 90,  160)


# ──────────────── COORDINATE HELPERS ────────────────

def w2s(x, y, map_size=MAP_SIZE):
    sx = int((x + GRID_LIMIT) / (2 * GRID_LIMIT) * map_size)
    sy = int((GRID_LIMIT - y) / (2 * GRID_LIMIT) * map_size)
    return sx, sy


# ──────────────── DRAWING ────────────────

def draw_ocean(surf, frame):
    surf.fill(OCEAN_DEEP)
    for y in range(0, MAP_SIZE, 14):
        off = math.sin(frame * 0.025 + y * 0.012) * 6
        b   = 8 + int(5 * math.sin(frame * 0.018 + y * 0.008))
        c   = (OCEAN_DEEP[0]+b, OCEAN_DEEP[1]+b, OCEAN_DEEP[2]+b)
        pygame.draw.line(surf, c, (int(off), y), (MAP_SIZE + int(off), y))


def draw_grid(surf):
    for i in range(-int(GRID_LIMIT), int(GRID_LIMIT)+1, 10):
        pygame.draw.line(surf, GRID_COL, w2s(i, -GRID_LIMIT), w2s(i, GRID_LIMIT), 1)
        pygame.draw.line(surf, GRID_COL, w2s(-GRID_LIMIT, i), w2s(GRID_LIMIT, i), 1)


def draw_trash(surf, env, frame):
    for i, t in enumerate(env.trash_list):
        tx, ty = w2s(t["x"], t["y"])
        c      = TRASH_PALETTE[i % len(TRASH_PALETTE)]
        bob    = math.sin(frame * 0.07 + i * 1.3) * 3
        ty_b   = int(ty + bob)
        rr     = int(10 + math.sin(frame * 0.05 + i) * 3)
        try:
            pygame.gfxdraw.aacircle(surf, tx, ty_b, rr, (c[0]//3, c[1]//3, c[2]//3))
        except Exception:
            pass
        rot = frame * 0.015 + i * 1.2
        ns  = 5 + (i % 3)
        pts = []
        for j in range(ns):
            a = rot + 2 * math.pi * j / ns
            r = 7 * (0.7 + 0.3 * math.sin(a * 3 + i))
            pts.append((tx + int(r * math.cos(a)), ty_b + int(r * math.sin(a))))
        pygame.draw.polygon(surf, c, pts)
        pygame.draw.polygon(surf, (min(c[0]+60,255), min(c[1]+60,255), min(c[2]+60,255)), pts, 2)
        pygame.draw.circle(surf, (255,255,255), (tx-2, ty_b-3), 2)


def draw_robot(surf, env, frame):
    rx, ry = w2s(env.robot_x, env.robot_y)
    th     = env.robot_theta
    ct, st = math.cos(th), math.sin(th)
    sn     = -st  # screen-y inverted

    # Scanning ring
    scan_r = int(22 + 6 * math.sin(frame * 0.1))
    try:
        pygame.gfxdraw.aacircle(surf, rx, ry, scan_r, (40, 160, 255, 40))
    except Exception:
        pass

    # Glow
    try:
        pygame.gfxdraw.filled_circle(surf, rx, ry, 20, (30, 120, 200, 25))
    except Exception:
        pass

    # Collector arms
    arm_spread = 0.5
    claw_open  = 0.15 + 0.1 * math.sin(frame * 0.12)
    for side in [1, -1]:
        arm_angle = th + side * arm_spread
        ax1 = rx + int(12 * math.cos(arm_angle))
        ay1 = ry - int(12 * math.sin(arm_angle))
        ax2 = rx + int(26 * math.cos(arm_angle))
        ay2 = ry - int(26 * math.sin(arm_angle))
        pygame.draw.line(surf, (180, 200, 220), (ax1, ay1), (ax2, ay2), 3)
        claw_a = arm_angle + side * claw_open
        cx_tip = ax2 + int(6 * math.cos(claw_a))
        cy_tip = ay2 - int(6 * math.sin(claw_a))
        pygame.draw.line(surf, (200, 220, 240), (ax2, ay2), (cx_tip, cy_tip), 2)

    # Body
    pygame.draw.circle(surf, (50, 55, 70),   (rx, ry), 14)
    pygame.draw.circle(surf, (70, 80, 100),  (rx, ry), 12)
    pygame.draw.circle(surf, (90, 100, 120), (rx, ry), 10)
    pygame.draw.circle(surf, (120, 135, 160),(rx-2, ry-2), 5)

    # Eye
    pygame.draw.circle(surf, (30, 40, 55), (rx, ry), 5)
    eye_pulse = int(180 + 75 * math.sin(frame * 0.2))
    pygame.draw.circle(surf, (0, eye_pulse, 255), (rx, ry), 3)

    # Heading arrow
    ant_len = 20
    ant_x = rx + int(ant_len * ct)
    ant_y = ry + int(ant_len * sn)
    pygame.draw.line(surf, (100, 255, 180),
                     (rx + int(12 * ct), ry + int(12 * sn)), (ant_x, ant_y), 2)
    arr = 5
    for sign in [1, -1]:
        bx = ant_x - int(arr * math.cos(th - sign * 0.5))
        by = ant_y + int(arr * math.sin(th - sign * 0.5))
        pygame.draw.line(surf, (100, 255, 180), (ant_x, ant_y), (bx, by), 2)

    # Side LEDs
    for side in [1, -1]:
        lx = rx + int(side * 8 * (-sn))
        ly = ry + int(side * 8 * ct)
        blink = (frame + side * 5) % 16 < 10
        col   = (255,60,60) if blink and side==-1 else (60,255,60) if blink else (40,40,40)
        pygame.draw.circle(surf, col, (lx, ly), 2)

    # Rear thruster
    rear_x = rx - int(14 * ct)
    rear_y = ry - int(14 * sn)
    thrust_r = int(3 + 2 * abs(math.sin(frame * 0.3)))
    try:
        pygame.gfxdraw.filled_circle(surf, rear_x, rear_y, thrust_r, (80, 180, 255, 100))
    except Exception:
        pass


def draw_camera(surf, env, obs, frame, cam_x, cam_y, cam_size):
    cam_surf  = pygame.Surface((cam_size, cam_size))
    cam_surf.fill(CAM_WATER)
    cam_range = 15.0

    def cam_w2s(wx, wy):
        dx     = wx - env.robot_x
        dy     = wy - env.robot_y
        cos_t  = math.cos(-env.robot_theta + math.pi / 2)
        sin_t  = math.sin(-env.robot_theta + math.pi / 2)
        rx_loc = dx * cos_t - dy * sin_t
        ry_loc = dx * sin_t + dy * cos_t
        px = int((rx_loc + cam_range) / (2 * cam_range) * cam_size)
        py = int((cam_range - ry_loc) / (2 * cam_range) * cam_size)
        return px, py

    # Grid
    for i in range(-30, 31, 5):
        pygame.draw.line(cam_surf, CAM_GRID, cam_w2s(i, -30), cam_w2s(i, 30), 1)
        pygame.draw.line(cam_surf, CAM_GRID, cam_w2s(-30, i), cam_w2s(30, i), 1)

    # Trash
    for i, t in enumerate(env.trash_list):
        cx, cy = cam_w2s(t["x"], t["y"])
        if 0 <= cx < cam_size and 0 <= cy < cam_size:
            c    = TRASH_PALETTE[i % len(TRASH_PALETTE)]
            dist = math.hypot(t["x"] - env.robot_x, t["y"] - env.robot_y)
            if dist <= obs.nearest_trash_dist + 0.1:
                ring_r = int(14 + math.sin(frame * 0.15) * 3)
                pygame.draw.circle(cam_surf, (255, 255, 100), (cx, cy), ring_r, 2)
                for dx, dy in [(-18,-0),(-8,0),(8,0),(18,0),(0,-18),(0,-8),(0,8),(0,18)]:
                    pygame.draw.line(cam_surf, (255,255,100), (cx+dx, cy+dy), (cx+dx//2, cy+dy//2), 1)
            bob  = math.sin(frame * 0.07 + i * 1.3) * 2
            cy_b = int(cy + bob)
            pygame.draw.circle(cam_surf, c, (cx, cy_b), 6)
            pygame.draw.circle(cam_surf, (255,255,255), (cx, cy_b), 6, 1)

    # Robot center
    ctr = cam_size // 2
    pygame.draw.circle(cam_surf, (55, 180, 240), (ctr, ctr), 8)
    pygame.draw.circle(cam_surf, (80, 210, 255), (ctr, ctr), 8, 2)
    pygame.draw.line(cam_surf, (100,255,150), (ctr, ctr), (ctr, ctr-14), 2)
    pygame.draw.line(cam_surf, (100,255,150), (ctr, ctr-14), (ctr-4, ctr-10), 2)
    pygame.draw.line(cam_surf, (100,255,150), (ctr, ctr-14), (ctr+4, ctr-10), 2)

    # Detection ring
    det_r = int(cam_range * 0.6 / cam_range * cam_size / 2)
    pygame.draw.circle(cam_surf, (40,100,180), (ctr, ctr), det_r, 1)

    # Scanline
    sl_y = (frame * 3) % cam_size
    pygame.draw.line(cam_surf, (40, 80, 140), (0, sl_y), (cam_size, sl_y), 1)

    # Readouts
    font_cam = pygame.font.SysFont("monospace", 11)
    ang_deg  = math.degrees(obs.nearest_trash_angle)
    cam_surf.blit(font_cam.render(f"DIST: {obs.nearest_trash_dist:.1f}u", True, (100,200,255)), (5, cam_size-28))
    cam_surf.blit(font_cam.render(f"ANG:  {ang_deg:.0f}°",                True, (100,200,255)), (5, cam_size-14))
    tc = font_cam.render(f"TARGETS: {obs.trash_count}", True, (255,200,60))
    cam_surf.blit(tc, (cam_size - tc.get_width() - 5, cam_size - 14))
    if frame % 20 < 14:
        pygame.draw.circle(cam_surf, (255,50,50), (cam_size-12, 12), 5)
    cam_surf.blit(font_cam.render("REC", True, (255,80,80)), (cam_size-38, 5))

    surf.blit(cam_surf, (cam_x, cam_y))
    pygame.draw.rect(surf, CAM_BORDER, pygame.Rect(cam_x, cam_y, cam_size, cam_size), 2)
    font_lbl = pygame.font.SysFont("monospace", 12, bold=True)
    surf.blit(font_lbl.render("◉ ONBOARD CAMERA", True, ACCENT), (cam_x + 6, cam_y - 16))


def draw_panel(surf, env, obs, ai_mode, episode, frame, font, font_sm, font_lg):
    px    = MAP_SIZE
    panel = pygame.Surface((PANEL_W, WIN_H))
    panel.fill(PANEL_BG)
    pygame.draw.line(panel, PANEL_BORDER, (0,0), (0, WIN_H), 2)
    surf.blit(panel, (px, 0))

    x0 = px + 12
    y  = 14

    badge = font_lg.render("◉ AI AGENT" if ai_mode else "◎ MANUAL", True,
                            WARNING if ai_mode else (80, 200, 255))
    surf.blit(badge, (x0, y)); y += 34

    pygame.draw.line(surf, PANEL_BORDER, (x0, y), (px + PANEL_W - 12, y), 1); y += 10

    stats = [
        ("EPISODE", str(episode),                           TEXT_BRIGHT),
        ("STEP",    f"{env.state.step_count}/{env.MAX_STEPS}", TEXT_BRIGHT),
        ("LEVEL",   env.task_level.upper(),                  ACCENT),
        ("TRASH",   f"{obs.trash_count}/{env.total_trash}",
         SUCCESS if obs.trash_count == 0 else DANGER if obs.trash_count > env.total_trash * 0.5 else TEXT_BRIGHT),
        ("REWARD",  f"{env.cumulative_reward:.3f}",          SUCCESS),
    ]
    font_stat = pygame.font.SysFont("monospace", 13, bold=True)
    for lbl, val, col in stats:
        surf.blit(font_sm.render(lbl, True, TEXT_DIM),    (x0, y))
        surf.blit(font_stat.render(val, True, col),        (x0 + 72, y - 1))
        y += 22

    # Progress bar
    y += 6
    bar_w = PANEL_W - 28
    bar_h = 8
    prog  = (env.total_trash - obs.trash_count) / max(env.total_trash, 1)
    pygame.draw.rect(surf, PROGRESS_BG,  (x0, y, bar_w, bar_h), border_radius=3)
    fw = int(bar_w * prog)
    if fw > 0:
        pygame.draw.rect(surf, PROGRESS_FIL, (x0, y, fw, bar_h), border_radius=3)
    pct = font_sm.render(f"{int(prog*100)}% cleaned", True, TEXT_DIM)
    surf.blit(pct, (x0 + bar_w//2 - pct.get_width()//2, y + 12))
    y += 32

    pygame.draw.line(surf, PANEL_BORDER, (x0, y), (px + PANEL_W - 12, y), 1); y += 20

    # Camera
    cam_x = px + (PANEL_W - CAMERA_SIZE) // 2
    draw_camera(surf, env, obs, frame, cam_x, y, CAMERA_SIZE)
    y += CAMERA_SIZE + 14

    # Controls hint
    for line in ["↑↓←→  Move   A  AI toggle", "1/2/3  Level  R  Reset"]:
        surf.blit(font_sm.render(line, True, TEXT_DIM), (x0, y)); y += 14


def draw_done_overlay(surf, env, obs, font_lg):
    ov = pygame.Surface((MAP_SIZE, 60), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 180))
    surf.blit(ov, (0, MAP_SIZE // 2 - 30))
    if env.collected_count == env.total_trash:
        msg = f"ALL COLLECTED!  Reward: {env.cumulative_reward:.3f}"
        col = SUCCESS
    else:
        msg = f"TIME UP  {env.collected_count}/{env.total_trash} collected"
        col = DANGER
    txt = font_lg.render(msg, True, col)
    surf.blit(txt, (MAP_SIZE//2 - txt.get_width()//2, MAP_SIZE//2 - txt.get_height()//2))


# ──────────────── RENDER FRAME → numpy ────────────────

def render_frame(surf, env, obs, ai_mode, episode, frame,
                 font, font_sm, font_lg):
    """Draw everything onto surf and return an H×W×3 uint8 numpy array."""
    # Map area
    map_surf = surf.subsurface(pygame.Rect(0, 0, MAP_SIZE, MAP_SIZE))
    draw_ocean(map_surf, frame)
    draw_grid(map_surf)
    draw_trash(map_surf, env, frame)
    draw_robot(map_surf, env, frame)
    if obs.done:
        draw_done_overlay(surf, env, obs, font_lg)
    draw_panel(surf, env, obs, ai_mode, episode, frame, font, font_sm, font_lg)

    # pygame surface → numpy RGB array (W×H×3) → transpose to H×W×3
    arr = pygame.surfarray.array3d(surf)   # shape: (W, H, 3)
    return arr.transpose(1, 0, 2)          # shape: (H, W, 3)


# ──────────────── GLOBAL STATE ────────────────
# Gradio callbacks are stateless per-request; we keep app state in a dict
# passed as gr.State.

def make_state(task_level="medium"):
    env = WaterTrashEnvironment()
    obs = env.reset(task_level=task_level)
    return {
        "env":       env,
        "obs":       obs,
        "ai_mode":   False,
        "model":     None,
        "task":      task_level,
        "frame":     0,
        "episode":   1,
        "prev_trash": obs.trash_count,
        "particles": [],   # kept for compat; not drawn in this version
    }


def load_ppo_model():
    try:
        from stable_baselines3 import PPO
        d = os.path.join(os.path.dirname(__file__), "trained_models")
        for n in ["best_model.zip", "final_model.zip"]:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return PPO.load(p)
    except Exception as e:
        print(f"PPO load failed: {e}")
    return None


def obs_arr_for_model(env, obs):
    mc, md = 50.0, math.hypot(100, 100)
    return np.array([
        obs.robot_x / mc,
        obs.robot_y / mc,
        obs.robot_theta / math.pi,
        min(obs.nearest_trash_dist / md, 1.0),
        obs.nearest_trash_angle / math.pi,
        obs.trash_count / max(env.total_trash, 1),
    ], dtype=np.float32)


# ── Pygame surfaces (initialised once) ──
pygame.init()
_surf  = pygame.Surface((WIN_W, WIN_H))
_font    = pygame.font.SysFont("monospace", 14, bold=True)
_font_sm = pygame.font.SysFont("monospace", 11)
_font_lg = pygame.font.SysFont("monospace", 20, bold=True)


# ──────────────── GRADIO CALLBACKS ────────────────

def _get_frame_img(state: dict) -> np.ndarray:
    env  = state["env"]
    obs  = state["obs"]
    return render_frame(
        _surf, env, obs,
        state["ai_mode"], state["episode"], state["frame"],
        _font, _font_sm, _font_lg,
    )


def tick(action: str, state: dict):
    """
    Called every ~100 ms by the Gradio timer.
    action: one of 'up','down','left','right','none'
             (sent from the JS keyboard listener or buttons)
    Returns (image_numpy, state)
    """
    env = state["env"]
    obs = state["obs"]

    state["frame"] += 1

    lv, av = 0.0, 0.0

    if state["ai_mode"] and state["model"] and not obs.done:
        a, _ = state["model"].predict(obs_arr_for_model(env, obs), deterministic=True)
        lv, av = float(a[0]), float(a[1])
    elif not state["ai_mode"]:
        if   action == "up":    lv =  1.0
        elif action == "down":  lv = -1.0
        elif action == "left":  av =  1.0
        elif action == "right": av = -1.0

    if not obs.done:
        obs = env.step(WaterTrashAction(linear_velocity=lv, angular_velocity=av))
        state["obs"] = obs

    # Auto-reset when AI finishes
    if obs.done and state["ai_mode"]:
        obs = env.reset(task_level=state["task"])
        state["obs"] = obs
        state["episode"] += 1
        state["prev_trash"] = obs.trash_count

    img = _get_frame_img(state)
    return img, state


def reset_episode(state: dict):
    env = state["env"]
    obs = env.reset(task_level=state["task"])
    state["obs"]       = obs
    state["episode"]  += 1
    state["prev_trash"] = obs.trash_count
    state["frame"]     = 0
    img = _get_frame_img(state)
    return img, state


def set_task(task: str, state: dict):
    state["task"] = task
    env = state["env"]
    obs = env.reset(task_level=task)
    state["obs"]      = obs
    state["episode"] += 1
    state["frame"]    = 0
    img = _get_frame_img(state)
    return img, state


def toggle_ai(state: dict):
    state["ai_mode"] = not state["ai_mode"]
    if state["ai_mode"] and state["model"] is None:
        state["model"] = load_ppo_model()
        if state["model"] is None:
            state["ai_mode"] = False
    label = "◉ AI ON" if state["ai_mode"] else "◎ Manual"
    img   = _get_frame_img(state)
    return img, label, state


# ──────────────── GRADIO UI ────────────────

def build_ui():
    init_state = make_state("medium")
    init_img   = _get_frame_img(init_state)

    with gr.Blocks(
        title="🌊 Water Trash Collector",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        body { background: #060E1E; }
        #main-img { border: 2px solid #1E4682; border-radius: 8px; }
        .gr-button { font-family: monospace !important; }
        footer { display:none !important; }
        """
    ) as demo:

        gr.Markdown(
            "## 🌊 Water Surface Trash Collector\n"
            "Use the arrow buttons (or keyboard) to navigate the robot. "
            "Enable AI to watch the trained PPO agent play."
        )

        state = gr.State(init_state)

        with gr.Row():
            with gr.Column(scale=3):
                img_out = gr.Image(
                    value=init_img,
                    label="Simulation",
                    elem_id="main-img",
                    show_label=False,
                    interactive=False,
                    height=WIN_H,
                )

            with gr.Column(scale=1, min_width=180):
                gr.Markdown("### Controls")

                ai_btn    = gr.Button("◎ Manual → Toggle AI", variant="primary")
                reset_btn = gr.Button("↺ Reset Episode")

                gr.Markdown("**Difficulty**")
                with gr.Row():
                    easy_btn   = gr.Button("Easy",   size="sm")
                    medium_btn = gr.Button("Medium", size="sm", variant="primary")
                    hard_btn   = gr.Button("Hard",   size="sm")

                gr.Markdown("**Manual Drive**")
                with gr.Row():
                    gr.HTML("<div></div>")
                    fwd_btn = gr.Button("▲", size="sm")
                    gr.HTML("<div></div>")
                with gr.Row():
                    lft_btn = gr.Button("◀", size="sm")
                    bck_btn = gr.Button("▼", size="sm")
                    rgt_btn = gr.Button("▶", size="sm")

                gr.Markdown(
                    "_Tip: keyboard arrow keys also work — "
                    "click the simulation image first to focus._"
                )

        # ── Timer: auto-tick (no-action) every 100 ms ──
        timer = gr.Timer(value=0.1)

        # ── Wire up timer (idle tick — no directional action) ──
        timer.tick(
            fn=lambda s: tick("none", s),
            inputs=[state],
            outputs=[img_out, state],
        )

        # ── Direction buttons ──
        for btn, act in [
            (fwd_btn, "up"), (bck_btn, "down"),
            (lft_btn, "left"), (rgt_btn, "right"),
        ]:
            btn.click(
                fn=lambda s, a=act: tick(a, s),
                inputs=[state],
                outputs=[img_out, state],
            )

        # ── AI toggle ──
        ai_btn.click(
            fn=toggle_ai,
            inputs=[state],
            outputs=[img_out, ai_btn, state],
        )

        # ── Reset ──
        reset_btn.click(
            fn=reset_episode,
            inputs=[state],
            outputs=[img_out, state],
        )

        # ── Difficulty ──
        easy_btn.click(
            fn=lambda s: set_task("easy", s),
            inputs=[state], outputs=[img_out, state],
        )
        medium_btn.click(
            fn=lambda s: set_task("medium", s),
            inputs=[state], outputs=[img_out, state],
        )
        hard_btn.click(
            fn=lambda s: set_task("hard", s),
            inputs=[state], outputs=[img_out, state],
        )

        # ── Keyboard arrow keys via JS ──
        # Injects a listener that fires button clicks based on keydown
        demo.load(
            fn=None,
            js="""
            () => {
                const keyMap = {
                    ArrowUp:    () => document.querySelector('button[aria-label="fwd"]')?.click(),
                    ArrowDown:  () => document.querySelector('button[aria-label="bck"]')?.click(),
                    ArrowLeft:  () => document.querySelector('button[aria-label="lft"]')?.click(),
                    ArrowRight: () => document.querySelector('button[aria-label="rgt"]')?.click(),
                };
                window.addEventListener('keydown', (e) => {
                    if (keyMap[e.key]) { e.preventDefault(); keyMap[e.key](); }
                });
            }
            """
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )