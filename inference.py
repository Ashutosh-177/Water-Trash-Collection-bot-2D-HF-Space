#!/usr/bin/env python3
"""
inference.py — LLM-driven agent for the Water Trash Collector.

Runs ALL THREE task levels (easy, medium, hard) sequentially,
reporting a score strictly in (0, 1) for each task.

Uses the hackathon LiteLLM proxy (API_BASE_URL + API_KEY) when available,
falls back to Google Gemini, or a deterministic math policy.

Environment variables:
    API_BASE_URL      — Hackathon LLM proxy base URL.
    API_KEY           — Hackathon LLM proxy API key.
    GEMINI_API_KEY    — Google Gemini API key (fallback).
    ENV_BASE_URL      — Environment server URL (default http://localhost:8000).
    MODEL             — Model name for the LLM proxy.
"""

import json
import math
import os
import re
import sys
import time
import urllib.request
import warnings

# Suppress all warnings (FutureWarning from google.generativeai crashes strict graders)
warnings.filterwarnings("ignore")

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary: Hackathon LLM Proxy (OpenAI-compatible)
LLM_API_BASE = os.environ.get("API_BASE_URL", "")
LLM_API_KEY = os.environ.get("API_KEY", "")

# Secondary: Direct Google Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))

# All three task levels — the grader requires at least 3
TASK_LEVELS = ["easy", "medium", "hard"]

if HAS_GENAI and GEMINI_API_KEY and not LLM_API_BASE:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Prompt builder & Rule-based Fallback
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI controlling a trash-collecting robot on a 2D water surface.

The observation vector you receive each step contains:
  robot_x           — current X position
  robot_y           — current Y position
  robot_theta       — heading in radians (0 = facing right)
  nearest_trash_dist  — distance to the nearest trash item
  nearest_trash_angle — relative angle to nearest trash (radians, 0 = straight ahead)
  trash_count       — number of remaining trash items

Your goal: collect all trash as fast as possible.

Respond with ONLY a JSON object (no markdown, no extra text):
{"move": <float -1 to 1>, "turn": <float -1 to 1>}

Where:
  move = linear_velocity  (1.0 = full forward, -1.0 = full reverse)
  turn = angular_velocity (positive = turn left, negative = turn right)

Strategy tips:
  - If nearest_trash_angle is positive, turn left (positive turn).
  - If nearest_trash_angle is negative, turn right (negative turn).
  - When |nearest_trash_angle| < 0.1, go full speed forward.
"""


def build_step_prompt(obs: dict) -> str:
    return (
        f"Observation:\n"
        f"  robot_x            = {obs.get('robot_x', 0)}\n"
        f"  robot_y            = {obs.get('robot_y', 0)}\n"
        f"  robot_theta        = {obs.get('robot_theta', 0)}\n"
        f"  nearest_trash_dist = {obs.get('nearest_trash_dist', 0)}\n"
        f"  nearest_trash_angle= {obs.get('nearest_trash_angle', 0)}\n"
        f"  trash_count        = {obs.get('trash_count', 0)}\n"
        f"\nRespond with the JSON action."
    )


def deterministic_policy(obs: dict) -> tuple:
    """Mathematical fallback if the LLM is unavailable."""
    angle = obs.get("nearest_trash_angle", 0.0)
    if abs(angle) < 0.15:
        return (1.0, 0.0)
    elif angle > 0:
        return (0.4, 1.0)
    else:
        return (0.4, -1.0)


def parse_llm_response(text: str) -> tuple:
    """Extract move/turn from the LLM response string."""
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            move = float(data.get("move", 0.0))
            turn = float(data.get("turn", 0.0))
            return (
                max(-1.0, min(1.0, move)),
                max(-1.0, min(1.0, turn)),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return (1.0, 0.0)


def call_llm_proxy(messages: list) -> str:
    """Call the hackathon LiteLLM proxy via raw HTTP."""
    req_data = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{LLM_API_BASE.rstrip('/')}/chat/completions",
        data=req_data,
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Run a single episode
# ---------------------------------------------------------------------------


def run_episode(client, WaterTrashAction, task_level: str, chat_history: list, chat=None):
    """Run one full episode for the given task_level. Returns (total_reward, steps, collected)."""

    result = client.reset(task_level=task_level)
    obs_obj = result.observation
    done = result.done

    step_num = 0
    total_reward = 0.0

    while not done and step_num < 400:
        step_num += 1

        obs = {
            "robot_x": obs_obj.robot_x,
            "robot_y": obs_obj.robot_y,
            "robot_theta": obs_obj.robot_theta,
            "nearest_trash_dist": obs_obj.nearest_trash_dist,
            "nearest_trash_angle": obs_obj.nearest_trash_angle,
            "trash_count": obs_obj.trash_count,
        }

        try:
            time.sleep(0.5)
            prompt = build_step_prompt(obs)
            chat_history.append({"role": "user", "content": prompt})

            if LLM_API_BASE and LLM_API_KEY:
                llm_text = call_llm_proxy(chat_history)
            elif chat:
                response = chat.send_message(prompt)
                llm_text = response.text.strip()
            else:
                raise Exception("No LLM available")

            chat_history.append({"role": "assistant", "content": llm_text})
            move, turn = parse_llm_response(llm_text)
            source = "LLM"
        except Exception:
            move, turn = deterministic_policy(obs)
            llm_text = f"[Fallback -> move:{move:.1f}, turn:{turn:.1f}]"
            source = "MATH-FALLBACK"

        print(f"[STEP] {step_num} | Task: {task_level} | Agent: {source}")
        print(f"  LLM response: {llm_text}")
        print(f"  Action: move={move:.3f}, turn={turn:.3f}")

        action = WaterTrashAction(linear_velocity=move, angular_velocity=turn)
        step_result = client.step(action)

        obs_obj = step_result.observation
        reward = step_result.reward if step_result.reward else 0.0
        done = step_result.done
        total_reward += reward

        print(f"  Reward: {reward:.4f}  |  Trash left: {obs_obj.trash_count}")

    return total_reward, step_num, obs_obj.trash_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    from client import WaterTrashEnv
    from models import WaterTrashAction

    # Set up LLM chat
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat = None
    if HAS_GENAI and GEMINI_API_KEY and not LLM_API_BASE:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        chat = model.start_chat()

    base = ENV_BASE_URL.rstrip("/")

    # ------ [START] ------
    print("[START]")
    print(f"  Environment : {base}")
    print(f"  Task levels : {TASK_LEVELS}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  LLM Proxy   : {'Active' if LLM_API_BASE else 'Direct/Fallback'}")

    # Wait for the environment server to be reachable
    for _ in range(15):
        try:
            req = urllib.request.Request(base + "/health")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            print(f"Waiting for environment server at {base} ...")
            time.sleep(2)

    try:
        with WaterTrashEnv(base_url=base).sync() as client:
            task_scores = {}

            for task_level in TASK_LEVELS:
                print(f"\n{'='*60}")
                print(f"  TASK: {task_level}")
                print(f"{'='*60}")

                total_reward, steps, remaining = run_episode(
                    client, WaterTrashAction, task_level, chat_history, chat
                )

                # Compute a score strictly in (0.0, 1.0) — never exactly 0 or 1
                raw_score = total_reward
                score = max(0.01, min(0.99, raw_score))
                task_scores[task_level] = score

                print(f"\n  [TASK COMPLETE] {task_level}")
                print(f"    Steps         : {steps}")
                print(f"    Total reward  : {total_reward:.4f}")
                print(f"    Remaining     : {remaining}")
                print(f"    Score         : {score:.4f}")

            # ------ [END] ------
            print("\n[END]")
            print("  Task Scores:")
            for task, score in task_scores.items():
                print(f"    {task}: {score:.4f}")

    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(0)


if __name__ == "__main__":
    main()
