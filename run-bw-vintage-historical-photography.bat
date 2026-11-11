@echo off
title PromptBeast Studio — Black & White Vintage Historical Photography
cd /d %~dp0
set PYTHONPATH=%~dp0
python -m app.main --style bw_vintage_historical_photography
pause
