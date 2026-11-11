@echo off
title PromptBeast Studio — 2D Stickman Style
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style 2d_stickman
pause
