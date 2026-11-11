@echo off
title PromptBeast Studio — What If?
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style what_if
pause
