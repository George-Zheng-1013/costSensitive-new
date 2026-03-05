@echo off
setlocal

REM 确保无论从哪里调用，工作目录都切换到脚本所在目录
cd /d "%~dp0"

REM 用法:
REM 1) 默认: start_netguard.bat
REM 2) 指定 Python: start_netguard.bat "C:\\Path\\to\\python.exe"

set "PYTHON_EXE=python"
if not "%~1"=="" set "PYTHON_EXE=%~1"

echo [NetGuard] Using Python: %PYTHON_EXE%

echo [NetGuard] Starting backend service in a new window...
start "NetGuard Backend" cmd /k ""%PYTHON_EXE%" services\backend_engine.py"

echo [NetGuard] Starting Streamlit UI (foreground)...
"%PYTHON_EXE%" -m streamlit run ui\app.py

endlocal
