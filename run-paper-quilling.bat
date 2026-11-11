@echo off
title PromptBeast Studio — Paper Quilling Illustration
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style paper_quilling
pause
