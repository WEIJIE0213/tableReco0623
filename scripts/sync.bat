@echo off
setlocal
REM Local helper: commit all changes and push to GitHub.
cd /d %~dp0\..

echo [sync] adding changes...
git add -A

echo [sync] committing...
git commit -m "sync %date% %time%"
if errorlevel 1 echo [sync] nothing to commit, pushing anyway...

echo [sync] pushing to origin main...
git push origin main

echo [sync] done.
pause
endlocal
