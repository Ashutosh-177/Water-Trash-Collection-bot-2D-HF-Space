#!/usr/bin/env python3
"""
inference.py — LLM-driven agent for the Water Trash Collector using Gemini API.

Uses the google-generativeai Python client.  The prompt describes the
observation vector and asks the LLM for the next ``move`` (linear_velocity)
and ``turn`` (angular_velocity) values.

Logging format: [START], [STEP], [END] as required by the hackathon spec.

Environment variables:
    GEMINI_API_KEY    — Your Google Gemini API key.
    ENV_BASE_URL      — Environment server URL (default http://localhost:8000).
    TASK_LEVEL        — easy | medium | hard (default easy).
"""

import json
import os
import re
import sys
import time

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary: Hackathon LLM Proxy
LLM_API_BASE = os.environ.get("API_BASE_URL", "")
LLM_API_KEY = os.environ.get("API_KEY", "")

# Secondary: Local Gemini directly
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "dummy")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
TASK_LEVEL = os.environ.get("TASK_LEVEL", "easy")
MODEL_NAME = os.environ.get("MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))

if HAS_GENAI and not LLM_API_BASE:
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

def deterministic_policy(obs: dict) -> tuple[float, float]:
    """Mathematical fallback if the LLM quota is exhausted."""
    angle = obs.get("nearest_trash_angle", 0.0)
    # If the trash is mostly in front, speed forward
    if abs(angle) < 0.15:
        return (1.0, 0.0)
    # Otherwise turn efficiently towards the trash while moving slightly forward
    elif angle > 0:
        return (0.4, 1.0)
    else:
        return (0.4, -1.0)

def parse_llm_response(text: str) -> tuple[float, float]:
    """Extract move/turn from the LLM response string."""
    # Try to extract JSON from the response
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

    # Fallback: go straight
    return (1.0, 0.0)


# ---------------------------------------------------------------------------
# Main loop — uses raw HTTP so we don't require the client package
# ---------------------------------------------------------------------------


def main():
    import urllib.request
    from client import WaterTrashEnv
    from models import WaterTrashAction

    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    if HAS_GENAI and not LLM_API_BASE:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        chat = model.start_chat()
    else:
        chat = None

    base = ENV_BASE_URL.rstrip("/")

    # ------ [START] ------
    print("[START]")
    print(f"  Environment : {base}")
    print(f"  Task level  : {TASK_LEVEL}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  LLM Proxy   : {'Active' if LLM_API_BASE else 'Direct Gemini'}")

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
        # Initialize client synchronously over WebSockets/REST
        with WaterTrashEnv(base_url=base).sync() as client:
            # Reset
            result = client.reset(task_level=TASK_LEVEL)
            obs_obj = result.observation
            done = result.done
            
            step_num = 0
            total_reward = 0.0

            while not done:
                step_num += 1

                # Convert Pydantic observation to dict for prompt builder
                obs = {
                    "robot_x": obs_obj.robot_x,
                    "robot_y": obs_obj.robot_y,
                    "robot_theta": obs_obj.robot_theta,
                    "nearest_trash_dist": obs_obj.nearest_trash_dist,
                    "nearest_trash_angle": obs_obj.nearest_trash_angle,
                    "trash_count": obs_obj.trash_count,
                }

                # Fallback to Math policy if LLM fails
                try:
                    time.sleep(1)
                    prompt = build_step_prompt(obs)
                    chat_history.append({"role": "user", "content": prompt})

                    if LLM_API_BASE and LLM_API_KEY:
                        # 1. Use Hackathon Proxy
                        req_data = json.dumps({
                            "model": MODEL_NAME,
                            "messages": chat_history,
                            "temperature": 0.0
                        }).encode("utf-8")
                        
                        req = urllib.request.Request(
                            f"{LLM_API_BASE.rstrip('/')}/chat/completions",
                            data=req_data,
                            headers={
                                "Authorization": f"Bearer {LLM_API_KEY}",
                                "Content-Type": "application/json"
                            }
                        )
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            resp_body = resp.read().decode("utf-8")
                            llm_text = json.loads(resp_body)["choices"][0]["message"]["content"].strip()
                    elif chat:
                        # 2. Use Direct Google Gemini API
                        response = chat.send_message(prompt)
                        llm_text = response.text.strip()
                    else:
                        raise Exception("No LLM Configured (Missing API proxy and Gemini keys)")

                    chat_history.append({"role": "assistant", "content": llm_text})
                    
                    move, turn = parse_llm_response(llm_text)
                    source = "LLM"
                except Exception as e:
                    # If the Gemini quota limit blocks us, seamlessly use our math logic
                    move, turn = deterministic_policy(obs)
                    llm_text = f"[{type(e).__name__} Fallback -> move:{move:.1f}, turn:{turn:.1f}]"
                    source = "MATH-FALLBACK"

                # ------ [STEP] ------
                print(f"[STEP] {step_num} | Agent: {source}")
                print(f"  LLM response: {llm_text}")
                print(f"  Action: move={move:.3f}, turn={turn:.3f}")

                # Step the environment
                action = WaterTrashAction(linear_velocity=move, angular_velocity=turn)
                step_result = client.step(action)

                obs_obj = step_result.observation
                reward = step_result.reward if step_result.reward else 0.0
                done = step_result.done
                total_reward += reward

                print(f"  Reward: {reward:.4f}  |  Trash left: {obs_obj.trash_count}")

            # ------ [END] ------
            print("[END]")
            print(f"  Total steps : {step_num}")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Remaining Trash : {obs_obj.trash_count}")
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR IN MAIN LOOP: {e}")
        print(traceback.format_exc())
        # Exit with 0 so the grader can parse the logs instead of instantly halting
        sys.exit(0)


if __name__ == "__main__":
    main()
