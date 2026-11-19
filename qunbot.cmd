@echo off
setlocal
if not exist "%~dp0.venv\Scripts\qunbot.exe" (
  echo Run uv sync --locked in the Qunbot repository first. 1>&2
  exit /b 1
)
"%~dp0.venv\Scripts\qunbot.exe" %*
exit /b %errorlevel%
