#!/usr/bin/env python3
"""
inference.py — LLM-driven agent for the Water Trash Collector.
Runs 3 tasks (easy, medium, hard) with [START]/[STEP]/[END] output format.
"""

import json
import os
import re
import sys
import time
import urllib.request
import warnings

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
LLM_API_BASE = os.environ.get("API_BASE_URL", "")
LLM_API_KEY = os.environ.get("API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
TASK_LEVELS = ["easy", "medium", "hard"]

if HAS_GENAI and GEMINI_API_KEY and not LLM_API_BASE:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Prompt & Policy
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an AI controlling a trash-collecting robot on a 2D water surface.
Observation: robot_x, robot_y, robot_theta, nearest_trash_dist, nearest_trash_angle, trash_count.
Goal: collect all trash fast.
Respond ONLY with JSON: {"move": <-1 to 1>, "turn": <-1 to 1>}
"""

def build_step_prompt(obs):
    return (f"robot_x={obs['robot_x']}, robot_y={obs['robot_y']}, "
            f"theta={obs['robot_theta']}, dist={obs['nearest_trash_dist']}, "
            f"angle={obs['nearest_trash_angle']}, trash={obs['trash_count']}. "
            f"Reply JSON only.")

def deterministic_policy(obs):
    angle = obs.get("nearest_trash_angle", 0.0)
    if abs(angle) < 0.15:
        return (1.0, 0.0)
    elif angle > 0:
        return (0.4, 1.0)
    else:
        return (0.4, -1.0)

def parse_llm_response(text):
    m = re.search(r'\{[^}]+\}', text)
    if m:
        try:
            d = json.loads(m.group())
            return (max(-1, min(1, float(d.get("move", 0)))),
                    max(-1, min(1, float(d.get("turn", 0)))))
        except Exception:
            pass
    return (1.0, 0.0)

def call_llm_proxy(messages):
    req_data = json.dumps({"model": MODEL_NAME, "messages": messages, "temperature": 0.0}).encode()
    req = urllib.request.Request(
        f"{LLM_API_BASE.rstrip('/')}/chat/completions",
        data=req_data,
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from client import WaterTrashEnv
    from models import WaterTrashAction

    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat = None
    if HAS_GENAI and GEMINI_API_KEY and not LLM_API_BASE:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        chat = model.start_chat()

    base = ENV_BASE_URL.rstrip("/")

    # Wait for env server
    for _ in range(15):
        try:
            req = urllib.request.Request(base + "/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(2)

    try:
        for task_level in TASK_LEVELS:
            with WaterTrashEnv(base_url=base).sync() as client:
                # === [START] ===
                print(f"[START] task={task_level}", flush=True)

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
                            raise Exception("No LLM")
                        chat_history.append({"role": "assistant", "content": llm_text})
                        move, turn = parse_llm_response(llm_text)
                    except Exception:
                        move, turn = deterministic_policy(obs)

                    action = WaterTrashAction(linear_velocity=move, angular_velocity=turn)
                    step_result = client.step(action)
                    obs_obj = step_result.observation
                    reward = step_result.reward if step_result.reward else 0.0
                    done = step_result.done
                    total_reward += reward

                    # === [STEP] ===
                    print(f"[STEP] step={step_num} reward={reward}", flush=True)

                # Compute score strictly in (0, 1)
                metadata = getattr(obs_obj, 'metadata', {})
                collected = metadata.get('collected', 0)
                total = metadata.get('total_trash', 1)
                raw_score = collected / max(total, 1)
                score = max(0.01, min(0.99, raw_score))

                # === [END] ===
                print(f"[END] task={task_level} score={score} steps={step_num}", flush=True)

    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    sys.exit(0)

if __name__ == "__main__":
    main()
