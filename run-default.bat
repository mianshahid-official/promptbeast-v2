@echo off
title PromptBeast Studio — Default
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style default
pause
