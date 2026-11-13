param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"

uv venv $Venv --python $Python
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$PythonVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($PythonVersion.Trim() -ne "3.12") {
    throw "ScamMark pilot requires Python 3.12; resolved $PythonVersion from $Python"
}
uv pip install --python $VenvPython torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $VenvPython -r (Join-Path $Root "requirements-pilot.txt")
uv pip freeze --python $VenvPython | Set-Content -Encoding utf8NoBOM (Join-Path $Root "requirements-pilot.lock.txt")
& $VenvPython -c "import torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})"
