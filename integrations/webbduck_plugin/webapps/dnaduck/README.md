# DNADuck WebbDuck Plugin

This plugin embeds DNADuck inside WebbDuck as an optional web-app plugin package distributed from the DNADuck repo.

The same UI also runs standalone — the API server serves it at `/ui/` for direct browser access without WebbDuck.

## What it does

- Adds a `DNADuck` tab in WebbDuck UI with three tabs: **Studio**, **Characters**, **Config**.
- Hosts plugin UI at `/plugins/web/dnaduck/ui/index.html`.
- Exposes plugin API at `/plugins/web/dnaduck/api/*`.
- Supports connection modes:
  - `auto` (default): prefer managed API, fallback to local CLI.
  - `managed_api`: auto-start or reuse DNADuck API.
  - `local_cli`: call DNADuck CLI directly.
  - `remote_api`: forward to a running DNADuck API server.
- Connection control is handled by WebbDuck plugin settings (remote plugin connect), not by DNADuck page controls.
- LoRA export currently uses identity-token captioning only.
- Rich descriptive captioning is planned (Coming Soon).

## Install From DNADuck Repo

Use DNADuck installer:

```bash
cd /path/to/dnaduck
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

Or install to shared plugins dir:

```bash
python3 tools/install_webbduck_plugin.py --plugins-dir ~/.webbduck/plugins --overwrite
```

## Local/Auto Modes

Backend resolves DNADuck root in this order:

1. `DNADUCK_ROOT` env var
2. sibling path `../dnaduck` next to `webbduck`
3. `./dnaduck`
4. `../dnaduck`

Override python binary with:

- `DNADUCK_PYTHON`

Override config path with:

- `DNADUCK_CONFIG`

## Mode 2: Remote API (Port-based)

Set:

- `DNADUCK_API_BASE=http://127.0.0.1:8020` (or any DNADuck API port)

When this is set, the plugin calls DNADuck REST endpoints instead of local CLI.

Example:

```bash
# terminal 1: run DNADuck API
cd /path/to/dnaduck
python3 run_api.py --port 8020

# terminal 2: run WebbDuck with remote bridge
cd /path/to/webbduck
DNADUCK_API_BASE=http://127.0.0.1:8020 python3 run.py
```

## Typical local setup

1. Ensure DNADuck exists next to WebbDuck:
   - `/path/to/webbduck`
   - `/path/to/dnaduck`
2. Put kohya `sd-scripts` in:
   - `/path/to/dnaduck/trainer/sd-scripts`
3. Configure DNADuck `config.yaml` (for trainer use `kohya_sd_scripts_dir: ./trainer/sd-scripts`).
4. Start WebbDuck and open the `DNADuck` tab.

Connection API endpoints:

- `GET /plugins/web/dnaduck/api/connection`
- `POST /plugins/web/dnaduck/api/connection`

### Config management (proxy)

When the plugin backend is in API mode (managed or remote), config endpoints proxy to the DNADuck API server supporting config switching. In local_cli mode, config read/write works directly on the YAML file but switching requires the API server.

- `GET /plugins/web/dnaduck/api/configs`
- `GET /plugins/web/dnaduck/api/config`
- `POST /plugins/web/dnaduck/api/config/switch`
- `PUT /plugins/web/dnaduck/api/config`

## Standalone (without WebbDuck)

Run the API server and open `http://localhost:8025/ui/` in a browser:

```bash
python3 run_api.py --config ./config.yaml --port 8025
```

The full UI (Studio, Characters, Config tabs) works standalone without WebbDuck connected.

## Your Two Cases

1. **DNADuck in plugin/local path**: plugin backend uses `auto` behavior and will try managed API, then local CLI fallback.
2. **DNADuck running separately**: connect by host:port in WebbDuck `Settings -> Connect Remote Plugin` (for example `127.0.0.1:8020`).
