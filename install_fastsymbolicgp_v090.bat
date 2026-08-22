@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Installing FastSymbolicGP V0.9.0
echo ============================================================
python -m pip install --upgrade pip
python -m pip install "dist\fastsymbolicgp-0.9.0-py3-none-any.whl[acceleration]"
if errorlevel 1 goto :error
python verify_install.py
if errorlevel 1 goto :error
echo.
echo FastSymbolicGP V0.9.0 installation completed successfully.
exit /b 0
:error
echo.
echo Installation or verification failed.
exit /b 1
