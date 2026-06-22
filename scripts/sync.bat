@echo off
REM 本地一键同步：commit + push 到 GitHub。双击即可。
cd /d %~dp0\..
echo [sync] adding changes...
git add -A
git commit -m "sync %date% %time%"
if errorlevel 1 echo [sync] nothing to commit, pushing anyway...
echo [sync] pushing to origin main...
git push origin main
echo [sync] done.
pause
