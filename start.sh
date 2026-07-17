#!/bin/bash
# Startup script for SSSH — mirrors the manual run sequence exactly:
#   1. cd sssh_main_copy
#   2. source .venv/bin/activate
#   3. cd integration
#   4. python3 main.py

cd /home/raspberry_pi4/sssh_main_copy
source .venv/bin/activate
cd integration
exec python3 main.py --config config.yaml
