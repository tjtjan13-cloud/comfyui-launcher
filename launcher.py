#!/usr/bin/env python3
"""ComfyUI Launcher (Windows-first, cross-platform).

One-command setup + launch for ComfyUI:
  1. Checks the Python version.
  2. Creates a virtual environment (./venv) on first run.
  3. Installs PyTorch (CUDA if an NVIDIA GPU is found, else CPU) and
     ComfyUI's requirements on first run. The CUDA wheel index is
     auto-detected from the GPU and Python version, newest first.
  4. Clones ComfyUI (./ComfyUI) on first run, or updates it with --update.
  5. Starts the ComfyUI server and opens it in the browser.

Usage:
  Double-click start.bat (Windows), or run:
      python launcher.py
      python launcher.py --port 8189 --update
      python launcher.py --cpu --no-browser

Persistent defaults live in launcher.json next to this file.
Only the Python standard library is used.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
COMFYUI_DIR = ROOT / "ComfyUI"
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
CONFIG_FILE = ROOT / "launcher.json"

MIN_PYTHON = (3, 10)
DEFAULT_PORT = 8188
DEFAULT_HOST = "127.0.0.1"
# Newest first. cu121 ships no Python 3.13 wheels, and Blackwell GPUs
# (RTX 50 series, compute capability 12.x) need CUDA 12.8+.
TORCH_CUDA_INDEXES = [
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu121",
]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
SERVER_WAIT_TIMEOUT = 180  # seconds


def log(msg: str) -> None:
    print(f"[launcher] {msg}", flush=True)


def fail(msg: str, code: int = 1) -> "SystemExit":
    print(f"[launcher] ERROR: {msg}", file=sys.stderr, flush=True)
    return SystemExit(code)


def run(cmd, cwd=None, check=True):
    """Run a command, streaming output, and return the CompletedProcess."""
    log(" ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        raise fail(f"command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")
    return result


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def find_system_python() -> str:
    """Return the python executable to use for creating the venv."""
    return sys.executable or "python"


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        raise fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}. "
            "Install a newer Python from https://www.python.org/downloads/ "
            "and tick 'Add python.exe to PATH' during setup."
        )
    log(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} OK")


def check_git() -> None:
    if shutil.which("git") is None:
        raise fail(
            "git was not found. Install Git from https://git-scm.com/downloads "
            "and make sure 'git' is on your PATH, then run the launcher again."
        )


def detect_gpu():
    """Return (name, compute_capability) of the first NVIDIA GPU, or (None, None)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None, None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None, None
    if result.returncode != 0:
        return None, None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, cap_text = line.rpartition(",")
        name = name.strip()
        cap = None
        try:
            cap = float(cap_text.strip())
        except ValueError:
            cap = None
        return name, cap
    return None, None


def cuda_index_candidates(torch_index=None, compute_cap=None):
    """Ordered PyTorch CUDA indexes to try, newest first.

    - An explicit --torch-index (or config value) always wins.
    - Blackwell GPUs (compute capability 12.x, RTX 50 series) only work
      with CUDA 12.8+, so only cu128 is tried for them.
    - cu121 ships no Python 3.13 wheels, so it is skipped on Python 3.13+.
    """
    if torch_index:
        return [torch_index]
    if compute_cap is not None and compute_cap >= 12.0:
        return [TORCH_CUDA_INDEXES[0]]
    candidates = list(TORCH_CUDA_INDEXES)
    if sys.version_info >= (3, 13):
        candidates = [c for c in candidates if "cu121" not in c]
    return candidates


def ensure_venv() -> Path:
    """Create ./venv on first run; return the venv's python executable."""
    python = venv_python()
    if python.exists():
        log(f"Virtual environment found: {VENV_DIR}")
        return python
    log("Creating virtual environment (./venv) ...")
    run([find_system_python(), "-m", "venv", str(VENV_DIR)])
    if not python.exists():
        raise fail(f"venv was created but {python} is missing.")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    return python


def ensure_comfyui(update: bool) -> None:
    """Clone ComfyUI on first run; pull latest with --update."""
    if not (COMFYUI_DIR / "main.py").exists():
        log("Cloning ComfyUI ...")
        run(["git", "clone", COMFYUI_REPO, str(COMFYUI_DIR)])
    elif update:
        log("Updating ComfyUI (git pull) ...")
        run(["git", "pull"], cwd=str(COMFYUI_DIR))
    else:
        log("ComfyUI found: ./ComfyUI")


def pip_install(python: Path, packages, index_url=None, check=True):
    cmd = [str(python), "-m", "pip", "install"]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += packages
    return run(cmd, check=check)


def install_cuda_torch(python: Path, torch_index=None, compute_cap=None) -> None:
    """Install CUDA PyTorch, trying indexes newest-first until one works."""
    candidates = cuda_index_candidates(torch_index, compute_cap)
    for i, index_url in enumerate(candidates):
        cuda_tag = index_url.rsplit("/", 1)[-1]  # e.g. "cu128"
        log(f"Installing PyTorch ({cuda_tag}) ...")
        result = pip_install(
            python, ["torch", "torchvision", "torchaudio"], index_url, check=False
        )
        if result.returncode == 0:
            log(f"PyTorch installed from {cuda_tag}.")
            return
        if i + 1 < len(candidates):
            log(f"Install from {cuda_tag} failed; trying the next index ...")
    raise fail(
        "Could not install PyTorch from any CUDA index. "
        "Check your network connection, then retry. If your GPU is older "
        "(GTX 10 series or earlier), try an older index explicitly, e.g.: "
        "python launcher.py --reinstall --torch-index "
        "https://download.pytorch.org/whl/cu121"
    )


def ensure_dependencies(python: Path, cpu_only: bool, reinstall: bool,
                        torch_index=None) -> None:
    """Install torch + ComfyUI requirements (first run, or --reinstall)."""
    marker = VENV_DIR / ".deps-installed"
    if marker.exists() and not reinstall:
        log("Dependencies already installed (use --reinstall to redo).")
        return
    if cpu_only:
        log("Installing PyTorch (CPU) ...")
        pip_install(python, ["torch", "torchvision", "torchaudio"], TORCH_CPU_INDEX)
    else:
        gpu_name, compute_cap = detect_gpu()
        if gpu_name is None:
            log("No NVIDIA GPU detected; installing PyTorch (CPU) ...")
            pip_install(python, ["torch", "torchvision", "torchaudio"], TORCH_CPU_INDEX)
        else:
            log(f"Detected GPU: {gpu_name}")
            install_cuda_torch(python, torch_index, compute_cap)
    requirements = COMFYUI_DIR / "requirements.txt"
    if requirements.exists():
        log("Installing ComfyUI requirements ...")
        pip_install(python, ["-r", str(requirements)])
    marker.write_text("ok", encoding="utf-8")
    log("Dependencies installed.")


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def wait_for_server(host: str, port: int, proc: subprocess.Popen, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if port_in_use(host, port):
            return True
        time.sleep(1)
    return port_in_use(host, port)


def load_config() -> dict:
    defaults = {
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "extra_args": [],
        "auto_open_browser": True,
        "torch_index": None,
    }
    if not CONFIG_FILE.exists():
        return defaults
    try:
        user_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"Warning: could not read launcher.json ({exc}); using defaults.")
        return defaults
    if not isinstance(user_config, dict):
        return defaults
    return {**defaults, **user_config}


def parse_args(config: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ComfyUI Launcher: set up and start ComfyUI."
    )
    parser.add_argument("--host", default=config["host"],
                        help=f"host to listen on (default: {config['host']})")
    parser.add_argument("--port", type=int, default=config["port"],
                        help=f"port to listen on (default: {config['port']})")
    parser.add_argument("--update", action="store_true",
                        help="update ComfyUI (git pull) before launching")
    parser.add_argument("--reinstall", action="store_true",
                        help="reinstall Python dependencies before launching")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU PyTorch even if an NVIDIA GPU is present")
    parser.add_argument("--torch-index", default=config.get("torch_index"),
                        help="PyTorch wheel index URL to use instead of "
                             "auto-detection, e.g. "
                             "https://download.pytorch.org/whl/cu121")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser automatically")
    parser.add_argument("--skip-install", action="store_true",
                        help="skip the dependency-install step")
    parser.add_argument("--comfyui-dir", default=str(COMFYUI_DIR),
                        help="use an existing ComfyUI folder instead of ./ComfyUI")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="extra arguments passed to ComfyUI (after --)")
    args = parser.parse_args()
    args.extra = [a for a in args.extra if a != "--"]
    return args


def main() -> int:
    config = load_config()
    args = parse_args(config)
    comfyui_dir = Path(args.comfyui_dir).resolve()

    log("=== ComfyUI Launcher ===")
    check_python()
    check_git()

    python = ensure_venv()
    # Allow --comfyui-dir to point at an existing install.
    global COMFYUI_DIR
    COMFYUI_DIR = comfyui_dir
    ensure_comfyui(update=args.update)
    if not args.skip_install:
        ensure_dependencies(python, cpu_only=args.cpu, reinstall=args.reinstall,
                            torch_index=args.torch_index)

    host, port = args.host, args.port
    if port_in_use(host, port):
        raise fail(
            f"Port {port} on {host} is already in use. "
            "Close the other ComfyUI (or app) using it, or pick another port: "
            "python launcher.py --port 8189"
        )

    url = f"http://{host}:{port}"
    cmd = [
        str(python), str(comfyui_dir / "main.py"),
        "--listen", host, "--port", str(port),
    ]
    cmd += list(config.get("extra_args") or []) + args.extra
    log(f"Starting ComfyUI on {url} ...")
    log("Press Ctrl+C to stop.")
    proc = subprocess.Popen(cmd, cwd=str(comfyui_dir))

    try:
        if wait_for_server(host, port, proc, SERVER_WAIT_TIMEOUT):
            log(f"ComfyUI is up: {url}")
            if config.get("auto_open_browser", True) and not args.no_browser:
                webbrowser.open(url)
        else:
            if proc.poll() is not None:
                log("ComfyUI exited during startup; see the output above.")
                return proc.returncode or 1
            log(f"ComfyUI did not answer on {url} within {SERVER_WAIT_TIMEOUT}s; "
                "it may still be loading. Check the window above.")
        proc.wait()
        return proc.returncode or 0
    except KeyboardInterrupt:
        log("Stopping ComfyUI ...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
