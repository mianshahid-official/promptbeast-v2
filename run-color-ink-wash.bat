@echo off
title PromptBeast Studio — Color Ink Wash
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style ink
pause

