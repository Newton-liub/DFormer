@echo off
setlocal
chcp 65001 >nul
set "SCRIPT=%~dp0paper_library.py"
set "PAUSE_AFTER=0"
if "%~1"=="" set "PAUSE_AFTER=1"

where py >nul 2>&1
if errorlevel 1 goto use_python
py -3 "%SCRIPT%" %*
set "RC=%ERRORLEVEL%"
goto finished

:use_python
python "%SCRIPT%" %*
set "RC=%ERRORLEVEL%"

:finished
if "%PAUSE_AFTER%"=="1" (
    if "%RC%"=="0" (
        echo.
        echo Paper library scan/update completed.
    ) else (
        echo.
        echo Paper library command failed with exit code %RC%.
    )
    pause
)
exit /b %RC%
