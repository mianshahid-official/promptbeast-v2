@echo off
title PromptBeast — Scene Sync Tool
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --mode scene_sync
pause
