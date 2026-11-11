@echo off
title PromptBeast Studio — Century-Old Photograph
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style vintage_photograph
pause
