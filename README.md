# DNADuck

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

DNADuck is a local identity extraction and clustering system for image datasets.
It recursively scans directories, embeds faces, clusters/assigns identities, and exports reusable metadata for downstream tooling (including LoRA dataset prep) without requiring image duplication.

## Implemented Phases

- Phase A: Batch face embedding and DBSCAN clustering.
- Phase B: Persistent SQLite identity database with incremental re-scan behavior.
- Phase C: REST API for scan, identity management, and search.
- Phase D: LoRA-oriented dataset export with hardlink/symlink/copy modes plus identity management operations.

## Project Layout

```text
dnaduck/
├── trainer/
│   ├── README.md
│   └── sd-scripts/           # place kohya-ss sd-scripts here
├── core/
│   ├── cluster.py
│   ├── database.py
│   ├── embedder.py
│   ├── exporter.py
│   ├── pipeline.py
│   ├── service.py
│   └── utils.py
├── server/
│   └── app.py
├── config.yaml
├── main.py
├── run_api.py
└── requirements.txt
```

## Key Outputs (No Image Duplication Required)

After scan:

- `output_folder/manifest.json`:
  - one entry per tracked image with absolute path, identity assignment, status, hash, timestamps.
- `output_folder/identities.json`:
  - identity groups with counts and labels.
- optional `output_folder/identities/`:
  - identity folders created using `identity_view_link_mode` (`none`, `hardlink`, `symlink`, `copy`).

For LoRA export:

- `output_folder/lora_export/identity_<id>/images/` (hardlinks by default).
- `output_folder/lora_export/identity_<id>/metadata.jsonl`.
- `output_folder/lora_export/identity_<id>/images/<name>.txt` caption sidecars for trainer compatibility.

## Captioning Mode (Current)

- Current mode is **identity-token captioning only**.
- Each exported image gets a `.txt` sidecar containing only the identity label/token.
- No visual caption model is used during LoRA export.
- **Rich descriptive captioning: Coming Soon**.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you run CPU-only, replace `onnxruntime-gpu` with `onnxruntime`.

## Optional WebbDuck Plugin Install

DNADuck is not bundled with WebbDuck. Install it as an optional plugin from this repo:

- `https://github.com/Duckieray/dnaduck`

```bash
cd /path/to/dnaduck
python3 tools/install_webbduck_plugin.py
```

Default target is `~/.webbduck/plugins/webapps/dnaduck`.

Install into a specific WebbDuck checkout:

```bash
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

The plugin supports:

- `auto` / `managed_api`: attempts to start and connect to DNADuck API.
- `local_cli`: executes DNADuck CLI commands directly.
- `remote_api`: connects to a separately running DNADuck API (`host:port`).

## Configuration

Edit `config.yaml`.

Important fields:

- `input_folder`: root folder to scan recursively.
- `output_folder`: metadata/export destination.
- `database_path`: persistent SQLite path.
- `mode`: `realism` | `anime` | `hybrid`.
- `eps_*`: DBSCAN thresholds for unknown faces (stricter defaults are set to reduce over-grouping).
- `assign_eps_*`: assignment thresholds against existing identities (stricter defaults are set to reduce catch-all identities).
- `exclude_name_contains`: filename substrings to skip during scan (default: `_upscaled`, `.thumb`).
- `identity_view_link_mode`: `none` | `symlink` | `hardlink` | `copy`.
- `lora_link_mode`: `hardlink` by default.
- `lora_trainer`: default `kohya_ss`.
- `kohya_sd_scripts_dir`: set to `./trainer/sd-scripts` (recommended).
- `kohya_base_model`: required for built-in trainer launch.
- `lora_train_command`: optional command template with `{dataset_dir}` placeholder (overrides built-in trainer).

### Trainer Folder Setup (Recommended)

Put kohya `sd-scripts` directly under DNADuck:

- `dnaduck/trainer/sd-scripts/`

Recommended config:

```yaml
kohya_sd_scripts_dir: ./trainer/sd-scripts
```

## CLI

Default command is `scan`.

```bash
python3 main.py scan
python3 main.py scan-recluster
python3 main.py scan --input-folder /path/to/images --output-folder /path/to/output
python3 main.py identities --min-members 1
python3 main.py search /path/to/query.jpg --top-k 5
python3 main.py label 12 --text "character_alice"
python3 main.py merge 12 15 18
python3 main.py export-lora --min-images 8
python3 main.py train-lora
python3 main.py images
```

## API

Start service:

```bash
python3 run_api.py --config ./config.yaml --port 8025
```

Example endpoints:

- `GET /health`
- `POST /scan`
- `POST /scan/recluster`
- `GET /identities?min_members=1`
- `GET /identity/{identity_id}`
- `POST /image/action` (`remove` | `blacklist` | `restore`)
- `GET /image?path=...`
- `POST /identity/{identity_id}/label`
- `POST /identity/merge`
- `POST /search`
- `POST /export/lora`
- `POST /train/lora`

## Notes

- Scan is recursive (`rglob`) and deterministic (sorted path traversal).
- Non-image files (including `.json`) are ignored during discovery.
- Filenames containing configured `exclude_name_contains` tokens are skipped.
- Existing images are skipped on re-scan if `size` + `mtime` match DB records.
- Updated/new files are re-embedded and reassigned incrementally.
- Metadata/identity counts are DB-backed and can include prior tracked images until moderated or reset.
- No external APIs are used.
- Default trainer hook targets `kohya_ss/sd-scripts` via `tools/train_kohya_lora.py`.

See `TESTING.md` for exact validation steps.
