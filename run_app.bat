@echo off
REM Launch the Streamlit UI using the venv's Python.
REM
REM Do NOT just run `streamlit run app.py` -- on this machine the `streamlit` on
REM PATH belongs to the global Windows Store Python, which has torch and
REM sentence-transformers but NOT torchvision or peft. The app then dies with a
REM confusing "No module named 'torchvision'" even though the venv has it.
REM Going through .venv\Scripts\python.exe -m streamlit pins the interpreter.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: no .venv here. Create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-app.txt
    exit /b 1
)

.venv\Scripts\python.exe -m streamlit run app.py %*
