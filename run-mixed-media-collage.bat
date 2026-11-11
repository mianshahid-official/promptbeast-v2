@echo off
title PromptBeast Studio — Mixed-Media Collage
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style mixed_media_collage
pause
