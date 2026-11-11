@echo off
title PromptBeast Studio — 3D Red & Blue Stickman Animation
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style 3d_red_blue_stickman
pause
