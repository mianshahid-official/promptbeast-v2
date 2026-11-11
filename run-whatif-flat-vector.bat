@echo off
title PromptBeast Studio — What-If Flat Vector
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style whatif_flat_vector
pause
