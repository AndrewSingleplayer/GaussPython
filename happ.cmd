@echo off
rem HA++ launcher (Windows). Usage: happ build file.ha
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
python -m happ %*
