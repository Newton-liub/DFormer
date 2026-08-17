@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish-canvas.ps1" %*
set "exitCode=%ERRORLEVEL%"
echo.
if "%exitCode%"=="0" (
    echo Canvas publish completed.
) else (
    echo Canvas publish failed with exit code %exitCode%.
)
pause
exit /b %exitCode%