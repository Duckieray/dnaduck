#!/usr/bin/env python3
"""Install the DNADuck WebbDuck web-app plugin into a plugins directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _source_plugin_dir() -> Path:
    return _repo_root() / "integrations" / "webbduck_plugin" / "webapps" / "dnaduck"

def _resolve_plugins_root(args: argparse.Namespace) -> Path:
    if args.plugins_dir:
        return Path(args.plugins_dir).expanduser().resolve()

    if args.webbduck_dir:
        return (Path(args.webbduck_dir).expanduser().resolve() / "plugins").resolve()

    env_dir = os.environ.get("WEBBDUCK_PLUGINS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    return (Path.home() / ".webbduck" / "plugins").resolve()

def _find_webbduck_port() -> int | None:
    """Scan common ports to find a live WebbDuck server."""
    env_port = os.environ.get("WEBBDUCK_PORT")
    candidates = [8010, 8020, 8030]
    if env_port:
        try:
            candidates.insert(0, int(env_port))
        except (ValueError, TypeError):
            pass
    for port in candidates:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode())
                if body.get("ok") or body.get("status") == "ok":
                    return port
        except Exception:
            continue
    return None


def _webbduck_port() -> int:
    port = _find_webbduck_port()
    if port is not None:
        return port
    env_port = os.environ.get("WEBBDUCK_PORT")
    if env_port:
        return int(env_port)
    return 8020


def _try_hot_reload(plugin_id: str) -> bool:
    port = _webbduck_port()
    url = f"http://localhost:{port}/plugins/web/{plugin_id}/reload"
    try:
        req = urllib.request.Request(url, method="POST", data=b"{}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode())
            if body.get("reloaded"):
                print(f"Hot-reloaded plugin '{plugin_id}' via {url}")
                return True
            print(f"Reload endpoint returned unexpected response: {body}")
            return False
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(
                "WebbDuck server is running but does not support hot-reload "
                "(endpoint not found). Restart WebbDuck to pick up the updated plugin."
            )
        else:
            detail = exc.read().decode()[:200]
            print(f"Reload request failed (HTTP {exc.code}): {detail}")
        return False
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        print("WebbDuck server is not running -- no reload needed.")
        return False


def _kill_existing_dnaduck() -> list[str] | None:
    """Kill any running dnaduck API server processes.
    Returns the command-line args of the killed process so it can be restarted."""
    import signal
    import subprocess

    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_api.py"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        pids = [int(p) for p in result.stdout.strip().split()]

        # Capture cmdline before killing (proc entries vanish on kill)
        cmdline = None
        for pid in pids:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    raw = f.read().split(b"\0")
                    parts = [p.decode("utf-8", errors="replace") for p in raw if p]
                    if any("run_api.py" in p for p in parts):
                        cmdline = parts
                        break
            except (FileNotFoundError, ProcessLookupError, OSError):
                continue

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Killed existing dnaduck API server (PID {pid})")
            except ProcessLookupError:
                pass

        import time
        time.sleep(1)

        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        return cmdline
    except Exception as exc:
        print(f"Warning: could not kill existing dnaduck process: {exc}")
        return None


def _restart_dnaduck(cmdline: list[str] | None) -> None:
    """Re-launch the dnaduck API server. Uses the previous cmdline if available,
    otherwise falls back to a sensible default."""
    import subprocess

    if cmdline:
        print(f"Restarting with: {' '.join(cmdline)}")
        subprocess.Popen(cmdline, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    # Fallback: try the standard location
    python = _resolve_python()
    dnaduck_root = _repo_root()
    fallback = [
        python,
        str(dnaduck_root / "run_api.py"),
        "--host", "127.0.0.1",
        "--port", "8020",
        "--config", str(dnaduck_root / "config.della.yaml"),
    ]
    print(f"Starting default dnaduck API server: {' '.join(fallback)}")
    subprocess.Popen(fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _resolve_python() -> str:
    """Find a Python interpreter that has dnaduck modules available.
    Checks current env first, then common conda envs."""
    import subprocess

    def _has_dnaduck(py: str) -> bool:
        try:
            r = subprocess.run([py, "-c", "import core.autogen, core.database"], capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    # 1. Current Python
    current = sys.executable
    if _has_dnaduck(current):
        return current

    # 2. Conda run (picks whatever env is active)
    conda_python = os.environ.get("CONDA_PREFIX", "")
    if conda_python:
        candidate = str(Path(conda_python) / "bin" / "python")
        if candidate != current and _has_dnaduck(candidate):
            return candidate

    # 3. Scan common env names
    conda_root = str(Path(conda_python).parent.parent) if conda_python else os.environ.get("CONDA_ROOT", str(Path.home() / "miniconda3"))
    for env_name in ["dnaduck", "webbduck", "web_img"]:
        candidate = str(Path(conda_root) / "envs" / env_name / "bin" / "python")
        if _has_dnaduck(candidate):
            return candidate

    # 4. Try conda run with each env
    for env_name in ["dnaduck", "webbduck", "web_img"]:
        try:
            r = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c", "import core.autogen, core.database"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                r2 = subprocess.run(
                    ["conda", "run", "-n", env_name, "which", "python"],
                    capture_output=True, text=True, timeout=5,
                )
                candidate = r2.stdout.strip()
                if candidate:
                    return candidate
        except Exception:
            continue

    return current  # last resort, will likely fail but the error will be informative


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install DNADuck as a WebbDuck web-app plugin.",
    )
    parser.add_argument(
        "--plugins-dir",
        default=None,
        help="WebbDuck plugins root (contains webapps/ and captioners/).",
    )
    parser.add_argument(
        "--webbduck-dir",
        default=None,
        help="Path to WebbDuck repo root (installs into <webbduck-dir>/plugins).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plugin files if already installed.",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Skip hot-reload attempt after installation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    killed_cmdline = _kill_existing_dnaduck()

    source_dir = _source_plugin_dir()
    if not source_dir.exists():
        print(f"ERROR: source plugin directory not found: {source_dir}", file=sys.stderr)
        return 1

    plugins_root = _resolve_plugins_root(args)
    target_dir = plugins_root / "webapps" / "dnaduck"

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and not args.overwrite:
        print(
            "ERROR: target already exists. Re-run with --overwrite to replace.\n"
            f"target={target_dir}",
            file=sys.stderr,
        )
        return 2

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    print("DNADuck WebbDuck plugin installed.")
    print(f"source: {source_dir}")
    print(f"target: {target_dir}")

    if not args.no_reload:
        _try_hot_reload("dnaduck")

    _restart_dnaduck(killed_cmdline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
