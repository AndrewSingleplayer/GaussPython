#!/bin/sh
# HA++ launcher (macOS / Linux). Usage: ./happ.sh build file.ha
DIR="$(cd "$(dirname "$0")" && pwd)"
exec env PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m happ "$@"
