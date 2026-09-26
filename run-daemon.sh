#!/bin/bash
cd "$HOME/watchflip" || exit 1
source .venv/bin/activate
exec caffeinate -i python run.py serve
