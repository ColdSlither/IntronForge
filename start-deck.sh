#!/usr/bin/env bash
# Profile Deck launcher. Forge Neo must be running on 7860 first
# (start it with: bash /path/to/forge/webui.sh --api)
cd "$(dirname "$0")/profile_deck"
exec /path/to/forge/venv/bin/python -m uvicorn app:app \
    --host 127.0.0.1 --port 7877
