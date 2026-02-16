# DNADuck Testing Guide

This guide validates recursive scanning, identity persistence, metadata exports, and API flows.

## 1) Environment

```bash
cd /path/to/dnaduck
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only alternative:

```bash
pip uninstall -y onnxruntime-gpu
pip install onnxruntime
```

## 2) Configure Paths

Edit `config.yaml` and set:

- `input_folder`: directory containing nested subfolders of images.
- `output_folder`: destination for manifests/exports.
- `database_path`: persistent sqlite file location.

Optional:

- `identity_view_link_mode: hardlink` to build identity folders without duplication.
- `exclude_name_contains: ["_upscaled", ".thumb"]` to skip these variants.

## 3) Run First Recursive Scan

```bash
python3 main.py scan
```

Expected:

- command prints `DNADuck scan complete`.
- `output_folder/manifest.json` exists.
- `output_folder/identities.json` exists.
- `database_path` exists.

## 4) Verify Recursive Behavior

Confirm nested files were indexed:

```bash
python3 - <<'PY'
import json
from pathlib import Path
manifest = Path("dnaduck_output/manifest.json")
data = json.loads(manifest.read_text(encoding="utf-8"))
print("manifest_entries", len(data))
print("sample_relative_paths", [row["relative_path"] for row in data[:5]])
PY
```

You should see subdirectory-style `relative_path` values (for example `set_a/face_001.png`).

## 5) Incremental Scan Test

Run scan again immediately:

```bash
python3 main.py scan
```

Expected:

- `processed_count` should usually be low (often `0`) if files did not change.
- identity assignments persist in SQLite across runs.

## 5b) Recluster From Scratch Test

```bash
python3 main.py scan-recluster
```

Expected:

- existing `database_path` identities are rebuilt from current scan scope.
- large prior catch-all identities should be reduced when thresholds are stricter.

## 6) Identity Ops Test

List identities:

```bash
python3 main.py identities --min-members 1
```

Label one:

```bash
python3 main.py label 1 --text "character_example"
```

Search by query image:

```bash
python3 main.py search /absolute/path/to/query.jpg --top-k 5
```

Inspect one identity page:

```bash
python3 main.py identity 1 --limit 20 --offset 0
```

Moderate one image:

```bash
python3 main.py image-action remove /absolute/path/to/image.png
python3 main.py image-action blacklist /absolute/path/to/image.png
python3 main.py image-action restore /absolute/path/to/image.png
```

## 7) LoRA Export Test (No Duplication via Hardlink)

```bash
python3 main.py export-lora --min-images 5
```

Expected:

- `output_folder/lora_export/identity_<id>/images/` populated.
- `metadata.jsonl` exists per identity.
- `.txt` sidecars exist per image and contain identity token text only.
- rich descriptive captioning is not enabled yet (Coming Soon).

## 8) API Test

Start API:

```bash
python3 run_api.py --config ./config.yaml --port 8025
```

In another shell:

```bash
curl http://127.0.0.1:8025/health
curl -X POST http://127.0.0.1:8025/scan -H "Content-Type: application/json" -d '{}'
curl "http://127.0.0.1:8025/identities?min_members=1"
curl -X POST http://127.0.0.1:8025/search -H "Content-Type: application/json" -d '{"image_path":"/absolute/path/to/query.jpg","top_k":5}'
curl -X POST http://127.0.0.1:8025/export/lora -H "Content-Type: application/json" -d '{"min_images":5}'
```

## 9) Optional Training Trigger Hook

### Default Trainer: kohya_ss (Chosen)

Set these in `config.yaml`:

```yaml
lora_trainer: kohya_ss
kohya_sd_scripts_dir: ./trainer/sd-scripts
kohya_base_model: /absolute/path/to/sdxl_base_model.safetensors
```

Place kohya `sd-scripts` at:

- `dnaduck/trainer/sd-scripts/`

Then run:

```bash
python3 main.py train-lora
```

DNADuck will invoke `tools/train_kohya_lora.py`, auto-build a dataset config TOML from `lora_export`, and launch `accelerate`.

### Custom Trainer Command (override)

Set `lora_train_command` in `config.yaml`:

```yaml
lora_train_command: "python train_lora.py --data {dataset_dir}"
```

Then run:

```bash
python3 main.py train-lora
```

or:

```bash
curl -X POST http://127.0.0.1:8025/train/lora -H "Content-Type: application/json" -d '{}'
```

## 10) WebbDuck Plugin Integration Test

1. Install DNADuck plugin from this repo:

```bash
cd /path/to/dnaduck
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

2. Start WebbDuck:

```bash
cd /path/to/webbduck
python3 run.py
```

3. Open WebbDuck UI and verify a `DNADuck` tab appears in the top nav.
4. Open the `DNADuck` tab and verify:
- Health block returns DNADuck paths.
- `Run Scan` works.
- Identities table populates.
- `Export LoRA Dataset` works.
- `Train LoRA` triggers training hook.

Remote bridge option:

```bash
# terminal 1
cd /path/to/dnaduck
python3 run_api.py --port 8020

# terminal 2
cd /path/to/webbduck
python3 run.py
```

Then in WebbDuck -> DNADuck tab -> Connection:

- Mode: `remote_api`
- Remote API Base: `127.0.0.1:8020`
- Click `Save Connection`
