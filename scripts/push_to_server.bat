@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push_to_server.ps1" %*
endlocal
