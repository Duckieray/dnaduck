#!/usr/bin/env python3
"""Install the DNADuck WebbDuck web-app plugin into a plugins directory."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
    print("")
    print("Next:")
    print("1) Start/restart WebbDuck.")
    print("2) Open WebbDuck and select the DNADuck tab.")
    print("3) Use WebbDuck Settings -> Connect Remote Plugin for remote host:port routing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
