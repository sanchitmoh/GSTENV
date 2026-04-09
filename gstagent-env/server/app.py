"""
OpenEnv server entry point.

This module provides the standard server/app.py that openenv expects.
It re-exports the FastAPI app and provides a main() entry point
for running the server via `python -m server.app` or `openenv serve`.
"""

import uvicorn

from environment.server import app  # noqa: F401

__all__ = ["app"]


def main():
    """Run the GST Agent Environment server."""
    uvicorn.run(
        "environment.server:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
