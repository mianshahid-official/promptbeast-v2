@echo off
title PromptBeast Studio — 3D Stickman Style
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style 3d_stickman
pause
