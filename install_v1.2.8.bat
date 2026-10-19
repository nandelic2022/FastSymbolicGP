@echo off
setlocal
python -m pip install --upgrade "fastsymbolicgp[acceleration]==1.2.8"
if errorlevel 1 exit /b 1
fastsymbolicgp-self-test
endlocal
