#!/usr/bin/env python3
"""ComfyUI Launcher (Windows-first, cross-platform).

One-command setup + launch for ComfyUI:
  1. Checks the Python version.
  2. Creates a virtual environment (./venv) on first run.
  3. Installs PyTorch (CUDA if an NVIDIA GPU is found, else CPU) and
     ComfyUI's requirements on first run. The CUDA wheel index is
     auto-detected from the GPU and Python version, newest first.
  4. Gets ComfyUI (./ComfyUI) on first run — git clone with retries,
     falling back to a direct ZIP download — or updates it with --update.
  5. Installs ComfyUI-Manager (the custom node manager) on first run.
  6. Installs optional acceleration libraries requested via --extras or
     the "extras" key in launcher.json (sageattention, flash-attn,
     llama-cpp, triton). Each is skipped when already installed, and a
     failed install only warns — ComfyUI still launches.
  7. Starts the ComfyUI server and opens it in the browser.

On Windows the launcher also installs a small startup hook in the venv
(a .pth file plus a helper module) that silences the harmless WinError
10054 asyncio traceback ComfyUI logs when a browser tab disconnects;
other errors are still reported.

Usage:
  Double-click start.bat (Windows), or run:
      python launcher.py
      python launcher.py --port 8189 --update
      python launcher.py --cpu --no-browser
      python launcher.py --extras sageattention,triton
      python launcher.py --check

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
import tempfile
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
COMFYUI_DIR = ROOT / "ComfyUI"
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_ZIP_URL = "https://codeload.github.com/comfyanonymous/ComfyUI/zip/refs/heads/master"
MANAGER_REPO = "https://github.com/ltdrdata/ComfyUI-Manager.git"
MANAGER_DIRNAME = "ComfyUI-Manager"
CLONE_ATTEMPTS = 3
CLONE_RETRY_DELAY = 5  # seconds
CONFIG_FILE = ROOT / "launcher.json"

MIN_PYTHON = (3, 10)
DEFAULT_PORT = 8188
DEFAULT_HOST = "127.0.0.1"
# Newest first. cu121 ships no Python 3.13 wheels. Blackwell GPUs
# (RTX 50 series, compute capability 12.x) need CUDA 12.8+ to run at
# all, and ComfyUI recommends CUDA 13.0+ for full performance on them.
TORCH_CUDA_INDEXES = [
    "https://download.pytorch.org/whl/cu130",
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu121",
]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
SERVER_WAIT_TIMEOUT = 180  # seconds
# Optional acceleration libraries, installed on request via --extras or the
# "extras" key in launcher.json. Each entry maps a short name to
# (pip package, import name, hint shown when the install fails).
# PyTorch itself is always installed by the launcher, so it is not an extra.
EXTRAS = {
    "sageattention": (
        "sageattention", "sageattention",
        "SageAttention needs a prebuilt wheel matching your torch/CUDA; "
        "see https://github.com/thu-ml/SageAttention for supported versions.",
    ),
    "flash-attn": (
        "flash-attn", "flash_attn",
        "flash-attn ships prebuilt wheels only for some Windows/torch "
        "combinations; without one it must be built from source, which "
        "needs a CUDA toolkit and takes a long time.",
    ),
    "llama-cpp": (
        "llama-cpp-python", "llama_cpp",
        "The default llama-cpp-python wheel is CPU-only; a CUDA build "
        "needs CMAKE_ARGS=-DGGML_CUDA=on (see its docs).",
    ),
    "triton": (
        "triton", "triton",
        "Triton usually comes with PyTorch already; installing it "
        "separately is rarely needed.",
    ),
}
# Startup hook that silences the harmless WinError 10054 asyncio noise
# on Windows (see ensure_quiet_asyncio). A .pth file in the venv's
# site-packages is used instead of sitecustomize.py, because a
# sitecustomize.py shipped with the system Python would shadow the
# venv's copy (the stdlib comes first on sys.path). The .pth file
# imports the helper module at interpreter startup, before ComfyUI
# runs, so the policy applies to every event loop ComfyUI creates.
# It only swallows ConnectionResetError callbacks (a browser tab
# disconnecting mid-request, logged by the proactor loop as WinError
# 10054); every other error is reported exactly as before.
QUIET_MODULE_NAME = "_comfyui_launcher_quiet"
QUIET_PTH_NAME = "_comfyui_launcher_quiet.pth"
# Marker identifying files installed by this launcher, so
# ensure_quiet_asyncio() never overwrites someone else's files.
QUIET_MARKER = "# ComfyUI-Launcher: asyncio noise filter"
QUIET_MODULE_CONTENT = '''# ComfyUI-Launcher: asyncio noise filter
# On Windows, ComfyUI's server runs on asyncio's proactor event loop,
# which logs a scary-looking ConnectionResetError traceback (WinError
# 10054) whenever a browser tab disconnects. The server keeps running;
# the message is just noise. This module hides only that message.
# Installed by the ComfyUI launcher; safe to delete (together with the
# _comfyui_launcher_quiet.pth file that imports it).
try:
    import asyncio

    class _QuietProactorPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            loop = super().new_event_loop()
            prev_handler = loop.get_exception_handler()

            def handler(loop, context):
                if isinstance(context.get("exception"), ConnectionResetError):
                    return
                if prev_handler is not None:
                    prev_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            loop.set_exception_handler(handler)
            return loop

    asyncio.set_event_loop_policy(_QuietProactorPolicy())
except Exception:
    pass
'''
QUIET_PTH_CONTENT = QUIET_MARKER + "\nimport " + QUIET_MODULE_NAME + "\n"


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
      with CUDA 12.8+, and run best on CUDA 13.0+, so only cu130/cu128
      are tried for them.
    - cu121 ships no Python 3.13 wheels, so it is skipped on Python 3.13+.
    """
    if torch_index:
        return [torch_index]
    if compute_cap is not None and compute_cap >= 12.0:
        return TORCH_CUDA_INDEXES[:2]
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


def _install_quiet_file(target: Path, content: str) -> str:
    """Install one noise-filter file. Returns 'current', 'wrote' or 'blocked'.

    'blocked' means a foreign file is in the way (never overwritten) or
    the write failed.
    """
    try:
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if existing == content:
                return "current"
            if QUIET_MARKER not in existing:
                return "blocked"
        target.write_text(content, encoding="utf-8")
        return "wrote"
    except OSError:
        return "blocked"


def ensure_quiet_asyncio(python: Path) -> None:
    """Silence the harmless WinError 10054 asyncio noise (Windows only).

    ComfyUI's server runs on asyncio's proactor loop, which logs a
    ConnectionResetError traceback whenever a browser tab disconnects.
    The server keeps running; the message is just noise. A .pth file
    plus a helper module in the venv hide only that message.
    Never overwrites files the launcher did not install.
    """
    if os.name != "nt":
        return
    site_packages = VENV_DIR / "Lib" / "site-packages"
    if not site_packages.is_dir():
        # Non-standard venv layout; ask the venv python where it lives.
        try:
            result = subprocess.run(
                [str(python), "-c",
                 "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                capture_output=True, text=True, timeout=30,
            )
            site_packages = Path(result.stdout.strip())
        except (subprocess.SubprocessError, OSError):
            return
        if not site_packages.is_dir():
            return
    results = [
        _install_quiet_file(site_packages / (QUIET_MODULE_NAME + ".py"),
                            QUIET_MODULE_CONTENT),
        _install_quiet_file(site_packages / QUIET_PTH_NAME, QUIET_PTH_CONTENT),
    ]
    if "wrote" in results:
        log("Installed asyncio noise filter.")
    if "blocked" in results:
        log("WARNING: could not install the asyncio noise filter; "
            "the WinError 10054 message may still appear.")


def clone_comfyui() -> bool:
    """Clone ComfyUI with git, retrying transient failures. Returns success."""
    for attempt in range(1, CLONE_ATTEMPTS + 1):
        log(f"Cloning ComfyUI (attempt {attempt}/{CLONE_ATTEMPTS}) ...")
        result = run(["git", "clone", COMFYUI_REPO, str(COMFYUI_DIR)], check=False)
        if result.returncode == 0 and (COMFYUI_DIR / "main.py").exists():
            return True
        # A failed clone can leave a partial directory behind, and git
        # refuses to clone into a non-empty directory, so remove it first.
        if COMFYUI_DIR.exists():
            shutil.rmtree(COMFYUI_DIR, ignore_errors=True)
        if attempt < CLONE_ATTEMPTS:
            log(f"Clone failed; retrying in {CLONE_RETRY_DELAY}s ...")
            time.sleep(CLONE_RETRY_DELAY)
    return False


def download_comfyui_zip() -> bool:
    """Download ComfyUI as a ZIP (fallback when git clone fails)."""
    log("Git clone failed; trying a direct ZIP download instead ...")
    try:
        request = urllib.request.Request(
            COMFYUI_ZIP_URL, headers={"User-Agent": "ComfyUI-Launcher"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "ComfyUI.zip"
            with urllib.request.urlopen(request, timeout=180) as response, \
                    open(zip_path, "wb") as f:
                shutil.copyfileobj(response, f)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            extracted = next(Path(tmp).glob("ComfyUI-*"))
            shutil.move(str(extracted), str(COMFYUI_DIR))
    except Exception as exc:  # network error, bad zip, ...
        log(f"ZIP download failed: {exc}")
        return False
    return (COMFYUI_DIR / "main.py").exists()


def ensure_comfyui(update: bool) -> None:
    """Get ComfyUI on first run; pull latest with --update (git installs)."""
    if not (COMFYUI_DIR / "main.py").exists():
        # A previous failed run may have left a partial directory behind.
        if COMFYUI_DIR.exists():
            log("Removing incomplete ComfyUI folder from a previous run ...")
            shutil.rmtree(COMFYUI_DIR, ignore_errors=True)
        if clone_comfyui():
            return
        if download_comfyui_zip():
            log("ComfyUI downloaded (ZIP install; --update needs a git checkout).")
            return
        raise fail(
            "Could not download ComfyUI. Check your network connection and "
            "try again (a VPN, proxy, firewall or antivirus can block git). "
            "You can also download the ZIP yourself from "
            "https://github.com/comfyanonymous/ComfyUI and extract it so that "
            "main.py sits at ./ComfyUI/main.py, then run the launcher again."
        )
    if update:
        if not (COMFYUI_DIR / ".git").exists():
            log("ComfyUI was installed from a ZIP, so --update cannot pull "
                "updates; re-download it to update. Continuing with the "
                "installed version.")
        else:
            log("Updating ComfyUI (git pull) ...")
            run(["git", "pull"], cwd=str(COMFYUI_DIR))
    else:
        log("ComfyUI found: ./ComfyUI")


def ensure_manager(python: Path, update: bool) -> None:
    """Install ComfyUI-Manager (custom node manager) on first run.

    Non-fatal: if the download fails, ComfyUI still launches.
    """
    manager_dir = COMFYUI_DIR / "custom_nodes" / MANAGER_DIRNAME
    if not (manager_dir / "__init__.py").exists():
        if manager_dir.exists():
            log("Removing incomplete ComfyUI-Manager folder ...")
            shutil.rmtree(manager_dir, ignore_errors=True)
        log("Installing ComfyUI-Manager ...")
        # Shallow clone: the Manager repo history is large and not needed.
        result = run(["git", "clone", "--depth", "1", MANAGER_REPO,
                      str(manager_dir)], check=False)
        if result.returncode != 0 or not (manager_dir / "__init__.py").exists():
            log("WARNING: Could not download ComfyUI-Manager; continuing "
                "without it. You can install it later from "
                "https://github.com/ltdrdata/ComfyUI-Manager")
            if manager_dir.exists():
                shutil.rmtree(manager_dir, ignore_errors=True)
            return
        requirements = manager_dir / "requirements.txt"
        if requirements.exists():
            log("Installing ComfyUI-Manager requirements ...")
            result = pip_install(python, ["-r", str(requirements)], check=False)
            if result.returncode != 0:
                log("WARNING: ComfyUI-Manager requirements failed to install; "
                    "the Manager may not work fully.")
        log("ComfyUI-Manager installed.")
    elif update and (manager_dir / ".git").exists():
        log("Updating ComfyUI-Manager (git pull) ...")
        run(["git", "pull"], cwd=str(manager_dir), check=False)
    else:
        log("ComfyUI-Manager found.")


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


def module_installed(python: Path, module: str) -> bool:
    """Check whether a module is importable in the venv."""
    code = (
        "import importlib.util, sys; "
        f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    )
    try:
        result = subprocess.run([str(python), "-c", code],
                                capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


def install_extra(python: Path, name: str) -> bool:
    """Install one optional extra (best effort). Returns True on success.

    Already-installed extras are skipped. Installs use prebuilt wheels
    only (--only-binary), so a missing wheel fails fast instead of
    triggering a long source build (flash-attn can take 20+ minutes
    to build from source on Windows).
    """
    package, module, hint = EXTRAS[name]
    if module_installed(python, module):
        log(f"Extra '{name}' already installed; skipping.")
        return True
    log(f"Installing extra '{name}' ({package}) ...")
    result = pip_install(python, ["--only-binary", ":all:", package], check=False)
    if result.returncode == 0:
        log(f"Extra '{name}' installed.")
        return True
    log(f"WARNING: could not install extra '{name}' (no prebuilt wheel "
        f"found for this Python/torch/CUDA combination). {hint}")
    return False


def ensure_extras(python: Path, extras) -> None:
    """Install the requested optional extras. Failures only warn."""
    wanted = []
    for raw in extras or []:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in EXTRAS:
            log(f"WARNING: unknown extra '{name}' (known extras: "
                f"{', '.join(sorted(EXTRAS))}); skipping.")
            continue
        if name not in wanted:
            wanted.append(name)
    for name in wanted:
        install_extra(python, name)


# Script run inside the venv to report installed packages as JSON.
# Keys are pip distribution names; each value carries whether the module
# is importable, the installed version (if any), and torch's CUDA info.
VENV_STATUS_SCRIPT = '''
import importlib.metadata, importlib.util, json
pkgs = {"torch": "torch", "sageattention": "sageattention",
        "flash-attn": "flash_attn", "llama-cpp-python": "llama_cpp",
        "triton": "triton"}
out = {}
for pkg, mod in pkgs.items():
    info = {"importable": importlib.util.find_spec(mod) is not None}
    try:
        info["version"] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        info["version"] = None
    out[pkg] = info
try:
    import torch
    out["torch"]["cuda"] = torch.version.cuda
    out["torch"]["cuda_available"] = torch.cuda.is_available()
except Exception:
    pass
print(json.dumps(out))
'''

# Order and short names used by the status check.
STATUS_PACKAGES = ("torch", "sageattention", "flash-attn",
                   "llama-cpp-python", "triton")
# pip distribution name -> --extras short name.
EXTRA_SHORT_NAMES = {
    "sageattention": "sageattention",
    "flash-attn": "flash-attn",
    "llama-cpp-python": "llama-cpp",
    "triton": "triton",
}


def query_venv_packages(python: Path) -> dict:
    """Ask the venv python which status packages are installed.

    Returns {pip name: {"importable": bool, "version": str|None, ...}}.
    Returns {} when the query itself fails.
    """
    try:
        result = subprocess.run(
            [str(python), "-c", VENV_STATUS_SCRIPT],
            capture_output=True, text=True, timeout=180,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    if result.returncode != 0:
        return {}
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        data = json.loads(lines[-1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def check_status(python: Path, comfyui_dir: Path) -> int:
    """Print the install status of torch, extras, ComfyUI and Manager.

    Read-only: never creates the venv or downloads anything.
    Always returns 0; missing pieces are reported, not errors.
    """
    log("=== ComfyUI Launcher status check ===")
    log(f"Python: {sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro} ({sys.executable})")
    gpu_name, compute_cap = detect_gpu()
    if gpu_name:
        cap = f" (compute capability {compute_cap})" if compute_cap else ""
        log(f"GPU: {gpu_name}{cap}")
    else:
        log("GPU: no NVIDIA GPU detected")
    if not python.exists():
        log("Virtual environment: not created yet (run the launcher once "
            "to set it up).")
    else:
        log(f"Virtual environment: {VENV_DIR}")
        status = query_venv_packages(python)
        if not status:
            log("Could not query the virtual environment; is its Python "
                "working? Try running the launcher once normally.")
        else:
            print()
            print(f"  {'package':<18}{'status':<14}version")
            for pkg in STATUS_PACKAGES:
                info = status.get(pkg, {})
                if info.get("importable") and info.get("version"):
                    state = "installed"
                    version = info["version"]
                    if pkg == "torch":
                        if info.get("cuda"):
                            gpu = ("GPU available" if info.get("cuda_available")
                                   else "GPU not available")
                            version += f" (CUDA {info['cuda']}, {gpu})"
                        else:
                            version += " (CPU build)"
                elif info.get("importable"):
                    state = "installed"
                    version = "(version unknown)"
                elif info.get("version"):
                    state = "BROKEN"
                    version = (f"{info['version']} "
                               "(installed but cannot be imported)")
                else:
                    state = "not installed"
                    version = "-"
                print(f"  {pkg:<18}{state:<14}{version}")
            print()
            missing = [p for p in STATUS_PACKAGES[1:]
                       if not status.get(p, {}).get("importable")]
            if missing:
                log("Missing extras can be installed with: "
                    "start.bat --extras "
                    + ",".join(EXTRA_SHORT_NAMES[p] for p in missing))
    comfyui_state = ("installed" if (comfyui_dir / "main.py").exists()
                     else "not downloaded yet")
    log(f"ComfyUI: {comfyui_state} ({comfyui_dir})")
    manager_dir = comfyui_dir / "custom_nodes" / MANAGER_DIRNAME
    manager_state = ("installed" if (manager_dir / "__init__.py").exists()
                     else "not installed")
    log(f"ComfyUI-Manager: {manager_state}")
    return 0


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
        "install_manager": True,
        "extras": [],
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
    parser.add_argument("--extras", default=None, metavar="LIST",
                        help="comma-separated optional acceleration "
                             "libraries to install, e.g. "
                             "--extras sageattention,flash-attn,llama-cpp,"
                             "triton (choices: "
                             + ", ".join(sorted(EXTRAS)) + ")")
    parser.add_argument("--no-manager", action="store_true",
                        default=not config.get("install_manager", True),
                        help="do not install ComfyUI-Manager")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser automatically")
    parser.add_argument("--skip-install", action="store_true",
                        help="skip the dependency-install step")
    parser.add_argument("--check", action="store_true",
                        help="print the install status of PyTorch, the "
                             "optional extras, ComfyUI and ComfyUI-Manager, "
                             "then exit without launching")
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
    if args.check:
        return check_status(venv_python(), comfyui_dir)
    check_git()

    python = ensure_venv()
    ensure_quiet_asyncio(python)
    # Allow --comfyui-dir to point at an existing install.
    global COMFYUI_DIR
    COMFYUI_DIR = comfyui_dir
    ensure_comfyui(update=args.update)
    if not args.skip_install:
        ensure_dependencies(python, cpu_only=args.cpu, reinstall=args.reinstall,
                            torch_index=args.torch_index)
        extras = config.get("extras") or []
        if isinstance(extras, str):
            extras = extras.split(",")
        extras = list(extras)
        if args.extras:
            extras += [e.strip() for e in args.extras.split(",")]
        ensure_extras(python, extras)
    if not args.no_manager:
        ensure_manager(python, update=args.update)

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
