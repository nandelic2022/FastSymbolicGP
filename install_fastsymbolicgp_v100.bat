@echo off
python -m pip install --upgrade pip
python -m pip install numba
python -m pip install --force-reinstall dist\fastsymbolicgp-1.0.0-py3-none-any.whl
python -c "import fastsymbolicgp; print('FastSymbolicGP', fastsymbolicgp.__version__)"
pause
