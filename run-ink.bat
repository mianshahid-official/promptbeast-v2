@echo off
title PromptBeast Studio — Ink Wash & Graphic Sketch
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style ink
pause
