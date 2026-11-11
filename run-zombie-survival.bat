@echo off
title PromptBeast Studio — Zombie Survival
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style zombie_survival
pause
