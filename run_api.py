#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DNADuck REST API service.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API bind host")
    parser.add_argument("--port", type=int, default=8025, help="API bind port")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    os.environ["DNADUCK_CONFIG"] = str(args.config)
    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=max(1, min(65535, int(args.port))),
        reload=False,
    )


if __name__ == "__main__":
    main()
