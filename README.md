# DNADuck

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

DNADuck is a local identity extraction and clustering system for image datasets.
It recursively scans directories, embeds faces, clusters/assigns identities, and exports reusable metadata for downstream tooling (including LoRA dataset prep) without requiring image duplication.

DNADuck includes a full-featured **web UI** accessible both as a WebbDuck plugin and as a standalone browser app.

## Implemented Phases

- Phase A: Batch face embedding and DBSCAN clustering.
- Phase B: Persistent SQLite identity database with incremental re-scan behavior.
- Phase C: REST API for scan, identity management, and search.
- Phase D: LoRA-oriented dataset export with hardlink/symlink/copy modes plus identity management operations.
- Phase E: Needs Review workflow — unassigned (noise + no_face) image browser with reassign, re-cluster, and re-analyze tools.
- Phase F: Config management — switch/edit config files at runtime via the web UI without server restart.
- Phase G: Auto-Generate — generate synthetic training images via WebbDuck API, matched against identity embeddings, to grow your character datasets.

## Project Layout

```text
dnaduck/
├── trainer/
│   ├── README.md
│   └── sd-scripts/              # place kohya-ss sd-scripts here
├── core/
│   ├── autogen.py               # auto-generation loop (generate → embed → match → keep/discard)
│   ├── cluster.py
│   ├── database.py
│   ├── embedder.py
│   ├── exporter.py
│   ├── pipeline.py
│   ├── service.py
│   └── utils.py
├── server/
│   └── app.py
├── integrations/
│   └── webbduck_plugin/
│       └── webapps/dnaduck/     # plugin + standalone webui
│           ├── backend.py
│           ├── plugin.json
│           └── ui/
│               ├── index.html
│               ├── app.js
│               └── styles.css
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

DNADuck dependencies are designed to install on top of WebbDuck's environment.

### Option 1: Standalone env

```bash
conda create -n dnaduck python=3.10 -y
conda activate dnaduck
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
conda create -n dnaduck python=3.10 -y
conda activate dnaduck
pip install -r requirements.txt
```

If you run CPU-only, replace `onnxruntime-gpu` with `onnxruntime`.

### Option 2: Install into WebbDuck's env (single env)

```bash
conda activate webbduck
pip install -r requirements.txt
```

This avoids needing `DNADUCK_PYTHON` — the WebbDuck plugin can use `sys.executable` directly since all deps are in one place.

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

Edit `config.yaml` directly or use the web UI Config tab.

### Environment Variables

Set per-process environment variables (e.g. `PYTORCH_ALLOC_CONF`) under the `env` key:

```yaml
env:
  PYTORCH_ALLOC_CONF: expandable_segments:True
```

These are injected into the LoRA training subprocess environment.

### Core config fields

- `input_folder`: root folder to scan recursively.
- `output_folder`: metadata/export destination.
- `database_path`: persistent SQLite path.
- `mode`: `realism` | `anime` | `hybrid`.
- `eps_*`: DBSCAN thresholds for unknown faces (stricter defaults are set to reduce over-grouping).
- `assign_eps_*`: assignment thresholds against existing identities (stricter defaults are set to reduce catch-all identities).
- `exclude_name_contains`: filename substrings to skip during scan (default: `_upscaled`, `.thumb`).
- `identity_view_link_mode`: `none` | `symlink` | `hardlink` | `copy`.

### Training config

- `lora_link_mode`: `hardlink` by default.
- `lora_trainer`: default `kohya_ss`.
- `kohya_sd_scripts_dir`: set to `./trainer/sd-scripts` (recommended).
- `kohya_base_model`: required for built-in trainer launch.
- `kohya_optimizer_type`: `auto` by default (prefers `AdamW8bit` if `bitsandbytes` exists, else `AdamW`).
- `kohya_attention_backend`: `auto` by default (prefers `xformers` if installed, else `sdpa`).
- `kohya_max_data_loader_n_workers` / `kohya_persistent_data_loader_workers`: dataloader speed tuning.
- `kohya_save_state_every_n_steps`: defaults to `250` to reduce save overhead while keeping resume support.
- `lora_train_command`: optional command template with `{dataset_dir}` placeholder (overrides built-in trainer).

### Auto-Generate config

Under the `auto_generate` key:

```yaml
auto_generate:
  enabled: false
  webbduck_url: http://localhost:8020
  base_model: ""                          # SDXL checkpoint path (required)
  scheduler: UniPC
  steps: 30
  cfg: 7.5
  width: 1024
  height: 1024
  second_pass_model: None
  negative_prompt: "low quality, blurry, bad anatomy, disfigured, extra limbs, bad hands"
  target_count: 50                        # images to collect before stopping
  max_attempts: 500                       # generation tries before giving up
  assign_eps_realism: 0.27                # match threshold for realism mode
  assign_eps_anime: 0.39                  # match threshold for anime mode
  prompt_templates:
    shot:                                 # weighted random — values are relative weights
      ultra closeup: 20
      portrait: 30
      "3/4 shot": 10
      full body: 10
      side profile: 30
    pose:
      facing camera: 40
      facing to the side: 25
      standing: 15
      sitting: 20
    hair:
      hair up: 50
      hair down: 50
    setting:
      in a coffee shop: 15
      in a forest: 15
      hiking on a trail: 10
      in a park: 15
      on a city street: 15
      in a studio: 10
      in a garden: 10
      at the beach: 10
    clothing:
      casual clothes: 30
      formal wear: 15
      summer dress: 20
      leather jacket: 20
      uniform: 15
    style:
      "": 30                              # empty string = no style modifier
      cinematic lighting: 25
      soft natural lighting: 25
      professional photography: 20
```

Each prompt template category supports three formats:
- `list[str]` — uniform random (backward compatible)
- `dict[str, number]` — weighted random (recommended; values normalised to probabilities)
- `list[dict]` with `text` and `weight` keys

The generated prompt is built as:
`"{label}, {shot}, {clothing}, {hair}, {pose}, {setting}, {style}, detailed face, high quality"`

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
python3 main.py list-unassigned
python3 main.py count-unassigned
python3 main.py reassign 42 /path/to/image.png
python3 main.py recluster-noise
python3 main.py reanalyze-no-face
python3 main.py autogen 42 --target-count 50 --max-attempts 500
python3 main.py autogen-cancel
python3 main.py autogen-status
```

## API

Start service:

```bash
python3 run_api.py --config ./config.yaml --port 8025
```

The API serves the web UI at `http://localhost:8025/ui/`.

### Scan & Identity endpoints

- `GET /health`
- `POST /scan`
- `POST /scan/recluster`
- `GET /identities?min_members=1`
- `GET /identity/{identity_id}`
- `POST /identity/{identity_id}/label`
- `POST /identity/merge`
- `POST /search`
- `POST /image/action` (`remove` | `blacklist` | `restore`)

### Review (unassigned) endpoints

- `GET /images/unassigned` — list noise + no_face images
- `GET /images/unassigned/count` — count by status
- `POST /image/reassign` — assign unassigned image to an identity
- `POST /images/unassigned/recluster` — re-run DBSCAN on noise embeddings
- `POST /images/unassigned/reanalyze` — re-run face detection on no_face images

### Training endpoints

- `POST /export/lora`
- `POST /train/lora`
- `GET /jobs/train/active`
- `GET /jobs/{job_id}`
- `POST /jobs/train/pause`

### Config management endpoints

- `GET /configs` — list available config files
- `GET /config` — get current config as JSON
- `POST /config/switch` — switch active config file
- `PUT /config` — update config values (including `env`)

### Auto-Generate endpoints

- `POST /autogen/start` — start auto-generation for an identity (body: `identity_id`, `target_count`, `max_attempts`)
- `POST /autogen/cancel` — cancel running auto-generation
- `GET /autogen/status` — get current progress (matched, attempts, target_count, message)

## Web UI

The web UI provides five tabs:

- **Studio** — scan, recluster, activity monitor, stats, training controls
- **Characters** — identity management with photo preview, rename, remove/blacklist, select for export
- **Review** — browse unassigned images (noise + no_face) with inline assign, re-cluster noise into tentative groups, re-analyze images that had no face
- **Build Dataset** — auto-generate synthetic training images for a character via WebbDuck
- **Config** — config file selection, field editing (paths, clustering, LoRA export, training params, env vars)

### Build Dataset Tab

The **Build Dataset** tab lets you generate synthetic training images for any character that has been labeled by the scan.

How it works:
1. Select a character from the dropdown (only characters with labels appear)
2. Set target photos and max generation attempts
3. Click **Generate** — the loop runs in the background:
   - Builds random prompts from weighted template categories (shot, pose, hair, setting, clothing, style)
   - Sends them to WebbDuck's `/test` API for generation
   - Detects the face, extracts the embedding, and compares it to the identity's centroid
   - Keeps matching images (adds to database as `assigned`) and deletes non-matching ones
4. A progress bar shows matched/target count in real time (polls every 3s)
5. Click **Cancel** to stop early

Prerequisites:
- The character must have a **label** (set in the Characters tab)
- The character must have a **centroid** (run a scan first)
- `auto_generate.base_model` must point to a valid SDXL checkpoint in `config.yaml`
- A running WebbDuck instance at `auto_generate.webbduck_url` (default `http://localhost:8020`)

Access it at `http://localhost:8025/ui/` when the API server is running. When installed as a WebbDuck plugin, the same UI appears in the DNADuck tab.

## Notes

- Scan is recursive (`rglob`) and deterministic (sorted path traversal).
- Non-image files (including `.json`) are ignored during discovery.
- Filenames containing configured `exclude_name_contains` tokens are skipped.
- Existing images are skipped on re-scan if `size` + `mtime` match DB records.
- Updated/new files are re-embedded and reassigned incrementally.
- Metadata/identity counts are DB-backed and can include prior tracked images until moderated or reset.
- No external APIs are used (except auto-generation, which calls WebbDuck's local `/test` endpoint).
- Default trainer hook targets `kohya_ss/sd-scripts` via `tools/train_kohya_lora.py`.
- Default trainer optimizer is `auto` (`AdamW8bit` with `bitsandbytes`, otherwise `AdamW`).
- Default attention backend is `auto` (`xformers` when installed, otherwise `sdpa`).
- If you force `kohya_optimizer_type: AdamW8bit`, `bitsandbytes` is required.
- If you force `kohya_attention_backend: xformers`, `xformers` is required.
- `POST /train/lora` now starts a background job by default and returns a `job_id`.
- Use `POST /train/lora` with `{"wait_for_result": true}` for synchronous/blocking behavior.
- Training progress now reports live step counters and ETA in `/activity` when available.
- Pause/resume: call `POST /jobs/train/pause`, then start training again to auto-resume from latest saved state.
- Config `env` variables are injected into the training subprocess via `subprocess.Popen(env=...)`.
- Auto-generation requires a running WebbDuck server at `auto_generate.webbduck_url` (default `http://localhost:8020`).
- The identity must have a label and centroid (run scan + set name first) before auto-generation will work.
- Auto-generation prompt templates use **weighted random selection** — adjust weights in `config.yaml` to control distribution (e.g. `side profile: 30` means ~30% of shots will be side profiles).
- Generated images that match the identity's embedding are added to the database as `assigned`; non-matching images are deleted.
- The `Review` tab shows unassigned images. Use **Re-cluster Noise** to find tentative groups, then assign to identities.
- Use **Re-analyze No-Face** to re-run face detection on images that previously had no face detected (different `det_size` or improved model may find faces now).

See `TESTING.md` for exact validation steps.
