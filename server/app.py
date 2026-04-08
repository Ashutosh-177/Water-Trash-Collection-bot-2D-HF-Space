"""
FastAPI application for the Water Surface Trash Collector Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET  /state: Get current environment state
    - GET  /schema: Get action/observation schemas
    - WS   /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with:\n    pip install openenv-core[core]\n"
    ) from e

try:
    from ..models import WaterTrashAction, WaterTrashObservation
    from .water_trash_env_environment import WaterTrashEnvironment
except (ImportError, ModuleNotFoundError):
    from models import WaterTrashAction, WaterTrashObservation
    from server.water_trash_env_environment import WaterTrashEnvironment


# Create the app with web interface and README integration
app = create_app(
    WaterTrashEnvironment,
    WaterTrashAction,
    WaterTrashObservation,
    env_name="water_trash_env",
    max_concurrent_envs=1,
)

# Custom addition: Mount our animated Gradio UI over the root path
try:
    import gradio as gr
    import sys
    import os
    # Add the root directory to path so we can import web_gui
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from web_gui import demo
    
    app = gr.mount_gradio_app(app, demo, path="/")
except Exception as e:
    print(f"Warning: Could not mount custom Gradio GUI. Running standard OpenEnv server only. Error: {e}")



def main():
    """
    Entry point for direct execution.

        uv run --project . server
        python -m water_trash_env.server.app
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
