#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDXL LoRA with kohya-ss/sd-scripts.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="DNADuck lora_export directory")
    parser.add_argument("--sd-scripts-dir", type=Path, required=True, help="Path to kohya sd-scripts repo")
    parser.add_argument("--base-model", type=str, required=True, help="Base model path/name for SDXL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for LoRA weights")
    parser.add_argument("--output-name", type=str, required=True, help="Output LoRA name prefix")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--network-dim", type=int, default=32, help="LoRA network dim")
    parser.add_argument("--network-alpha", type=int, default=16, help="LoRA network alpha")
    parser.add_argument("--batch-size", type=int, default=1, help="Train batch size")
    parser.add_argument("--resolution", type=str, default="1024,1024", help="Resolution, e.g. 1024,1024")
    bucket_group = parser.add_mutually_exclusive_group()
    bucket_group.add_argument(
        "--enable-bucket",
        dest="enable_bucket",
        action="store_true",
        help="Enable aspect-ratio bucketing.",
    )
    bucket_group.add_argument(
        "--disable-bucket",
        dest="enable_bucket",
        action="store_false",
        help="Disable bucketing and force fixed resolution crops.",
    )
    parser.set_defaults(enable_bucket=True)
    parser.add_argument("--bucket-no-upscale", action="store_true", help="Do not upscale images in buckets.")
    parser.add_argument("--min-bucket-reso", type=int, default=256, help="Minimum bucket resolution.")
    parser.add_argument("--max-bucket-reso", type=int, default=1536, help="Maximum bucket resolution.")
    parser.add_argument("--num-repeats", type=int, default=10, help="Repeats per identity subset")
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="auto",
        help="Optimizer type: auto | AdamW | AdamW8bit",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="auto",
        help="Attention backend: auto | xformers | sdpa | none",
    )
    parser.add_argument(
        "--max-data-loader-workers",
        type=int,
        default=2,
        help="Maximum dataloader workers (0 disables workers).",
    )
    persistent_workers_group = parser.add_mutually_exclusive_group()
    persistent_workers_group.add_argument(
        "--persistent-data-loader-workers",
        dest="persistent_data_loader_workers",
        action="store_true",
        help="Keep dataloader workers alive between epochs.",
    )
    persistent_workers_group.add_argument(
        "--no-persistent-data-loader-workers",
        dest="persistent_data_loader_workers",
        action="store_false",
        help="Disable persistent dataloader workers.",
    )
    parser.set_defaults(persistent_data_loader_workers=True)
    save_state_group = parser.add_mutually_exclusive_group()
    save_state_group.add_argument(
        "--save-state",
        dest="save_state",
        action="store_true",
        help="Save training state for resume support.",
    )
    save_state_group.add_argument(
        "--no-save-state",
        dest="save_state",
        action="store_false",
        help="Disable periodic training state saves.",
    )
    parser.set_defaults(save_state=True)
    parser.add_argument(
        "--save-state-every-steps",
        type=int,
        default=100,
        help="How often to save state snapshots (steps).",
    )
    auto_resume_group = parser.add_mutually_exclusive_group()
    auto_resume_group.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Auto-resume from the newest state folder in output-dir.",
    )
    auto_resume_group.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Do not auto-resume from state folders.",
    )
    parser.set_defaults(auto_resume=True)
    parser.add_argument(
        "--resume-state",
        type=Path,
        default=None,
        help="Explicit state directory path to resume from.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    sd_scripts_dir = args.sd_scripts_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not (sd_scripts_dir / "sdxl_train_network.py").exists():
        raise FileNotFoundError(f"sdxl_train_network.py not found in: {sd_scripts_dir}")

    subsets = discover_subsets(dataset_dir)
    if not subsets:
        raise RuntimeError(
            f"No identity subsets found in dataset dir: {dataset_dir}. "
            "Run `python main.py export-lora` first."
        )

    dataset_config = output_dir / "dataset_config.toml"
    dataset_resolution = parse_resolution(args.resolution)
    dataset_config.write_text(
        build_dataset_toml(
            subsets=subsets,
            repeats=max(1, int(args.num_repeats)),
            resolution=dataset_resolution,
            batch_size=max(1, int(args.batch_size)),
            enable_bucket=bool(args.enable_bucket),
            bucket_no_upscale=bool(args.bucket_no_upscale),
            min_bucket_reso=max(64, int(args.min_bucket_reso)),
            max_bucket_reso=max(64, int(args.max_bucket_reso)),
        ),
        encoding="utf-8",
    )

    resume_state: Path | None = None
    if args.resume_state is not None:
        resume_state = args.resume_state.expanduser().resolve()
        if not resume_state.exists() or not resume_state.is_dir():
            raise FileNotFoundError(f"Resume state directory does not exist: {resume_state}")
    elif bool(args.auto_resume):
        resume_state = find_latest_state_dir(output_dir=output_dir, output_name=args.output_name)

    requested_optimizer_raw = str(args.optimizer_type)
    effective_optimizer, optimizer_note = resolve_optimizer_type(requested_optimizer_raw)
    effective_attention, attention_note = resolve_attention_backend(str(args.attention))
    effective_optimizer, resume_state, resume_note = adapt_resume_optimizer_compatibility(
        requested_optimizer=requested_optimizer_raw,
        effective_optimizer=effective_optimizer,
        resume_state=resume_state,
    )

    command = build_command(
        python=sys.executable,
        sd_scripts_dir=sd_scripts_dir,
        base_model=args.base_model,
        dataset_config=dataset_config,
        output_dir=output_dir,
        output_name=args.output_name,
        steps=max(1, int(args.steps)),
        learning_rate=float(args.learning_rate),
        network_dim=max(1, int(args.network_dim)),
        network_alpha=max(1, int(args.network_alpha)),
        batch_size=max(1, int(args.batch_size)),
        resolution=args.resolution,
        optimizer_type=effective_optimizer,
        attention_backend=effective_attention,
        max_data_loader_workers=max(0, int(args.max_data_loader_workers)),
        persistent_data_loader_workers=bool(args.persistent_data_loader_workers),
        save_state=bool(args.save_state),
        save_state_every_steps=max(1, int(args.save_state_every_steps)),
        resume_state=resume_state,
    )

    print("Kohya trainer command:")
    print(" ".join(command))
    print(f"Dataset config: {dataset_config}")
    print(f"Subsets: {len(subsets)}")
    print(f"Optimizer: {effective_optimizer}")
    print(f"Attention: {effective_attention}")
    if resume_state is not None:
        print(f"Resume state: {resume_state}")
    if optimizer_note:
        print(f"Optimizer note: {optimizer_note}")
    if attention_note:
        print(f"Attention note: {attention_note}")
    if resume_note:
        print(f"Resume note: {resume_note}")

    if args.dry_run:
        return

    env = os.environ.copy()
    # Force UTF-8 for sd-scripts output on Windows; avoids cp1252 encode crashes.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Disable Python stdio buffering so progress lines stream to DNADuck in real time.
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        command,
        cwd=str(sd_scripts_dir),
        env=env,
        capture_output=False,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kohya training failed with return code {result.returncode}")


def discover_subsets(dataset_dir: Path) -> list[dict]:
    subsets: list[dict] = []
    for identity_dir in sorted(dataset_dir.glob("identity_*")):
        if not identity_dir.is_dir():
            continue
        images_dir = identity_dir / "images"
        identity_meta = identity_dir / "identity.json"
        if not images_dir.exists() or not identity_meta.exists():
            continue

        try:
            metadata = json.loads(identity_meta.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        class_tokens = str(metadata.get("label") or identity_dir.name)
        subsets.append({"image_dir": images_dir.resolve(), "class_tokens": class_tokens})
    return subsets


def parse_resolution(value: str) -> tuple[int, int]:
    text = str(value or "").strip().lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format: {value!r}. Expected 'width,height' like '1024,1024'.")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except Exception as exc:
        raise ValueError(f"Invalid resolution numbers: {value!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive: {value!r}")
    return width, height


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def resolve_optimizer_type(requested: str) -> tuple[str, str | None]:
    raw = str(requested or "").strip()
    normalized = raw.lower()
    if normalized in {"", "auto"}:
        if _module_available("bitsandbytes"):
            return "AdamW8bit", None
        return "AdamW", "bitsandbytes not found, using AdamW."
    if normalized == "adamw":
        return "AdamW", None
    if normalized == "adamw8bit":
        if not _module_available("bitsandbytes"):
            raise RuntimeError(
                "Optimizer AdamW8bit requires bitsandbytes, but it is not installed. "
                "Install it in the current environment or set kohya_optimizer_type to AdamW/auto."
            )
        return "AdamW8bit", None
    raise ValueError(f"Unsupported optimizer type: {requested!r}. Use auto | AdamW | AdamW8bit.")


def resolve_attention_backend(requested: str) -> tuple[str, str | None]:
    raw = str(requested or "").strip()
    normalized = raw.lower()
    if normalized in {"", "auto"}:
        if _module_available("xformers"):
            return "xformers", None
        return "sdpa", "xformers not found, using SDPA."
    if normalized in {"none", "off", "disabled"}:
        return "none", None
    if normalized == "sdpa":
        return "sdpa", None
    if normalized == "xformers":
        if not _module_available("xformers"):
            raise RuntimeError(
                "Attention backend xformers was requested, but xformers is not installed. "
                "Install xformers or set kohya_attention_backend to auto/sdpa."
            )
        return "xformers", None
    raise ValueError(f"Unsupported attention backend: {requested!r}. Use auto | xformers | sdpa | none.")


def detect_resume_optimizer_family(resume_state: Path) -> str | None:
    optimizer_file = resume_state / "optimizer.bin"
    if not optimizer_file.exists():
        return None
    try:
        import torch
    except Exception:
        return None
    try:
        payload = torch.load(str(optimizer_file), map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    if not isinstance(state, dict) or not state:
        return None
    sample = next(iter(state.values()), None)
    if not isinstance(sample, dict):
        return None
    if "state1" in sample or "state2" in sample:
        return "AdamW8bit"
    if "exp_avg" in sample or "exp_avg_sq" in sample:
        return "AdamW"
    return None


def adapt_resume_optimizer_compatibility(
    *,
    requested_optimizer: str,
    effective_optimizer: str,
    resume_state: Path | None,
) -> tuple[str, Path | None, str | None]:
    if resume_state is None:
        return effective_optimizer, resume_state, None

    detected = detect_resume_optimizer_family(resume_state)
    if not detected:
        return effective_optimizer, resume_state, None

    requested = str(requested_optimizer or "").strip().lower()
    effective = str(effective_optimizer or "").strip()
    if detected == effective:
        return effective, resume_state, None

    # Respect explicit user choice, but fail with a concrete message.
    if requested not in {"", "auto"}:
        raise RuntimeError(
            f"Resume state optimizer is {detected}, but optimizer '{effective}' was requested. "
            "Use optimizer 'auto' or clear resume state."
        )

    if detected == "AdamW8bit" and not _module_available("bitsandbytes"):
        return (
            effective,
            None,
            "Resume state requires AdamW8bit, but bitsandbytes is unavailable. Starting fresh without resume.",
        )

    return (
        detected,
        resume_state,
        f"Resume state was created with {detected}; using {detected} for compatibility.",
    )


def build_dataset_toml(
    subsets: list[dict],
    repeats: int,
    *,
    resolution: tuple[int, int],
    batch_size: int,
    enable_bucket: bool,
    bucket_no_upscale: bool,
    min_bucket_reso: int,
    max_bucket_reso: int,
) -> str:
    width, height = resolution
    min_bucket = max(64, int(min_bucket_reso))
    max_bucket = max(min_bucket, int(max_bucket_reso))
    lines = [
        "[[datasets]]",
        f"resolution = [{int(width)}, {int(height)}]",
        f"batch_size = {max(1, int(batch_size))}",
        f"enable_bucket = {str(bool(enable_bucket)).lower()}",
        f"bucket_no_upscale = {str(bool(bucket_no_upscale)).lower()}",
        f"min_bucket_reso = {min_bucket}",
        f"max_bucket_reso = {max_bucket}",
        "",
    ]
    for subset in subsets:
        image_dir = str(subset["image_dir"]).replace("\\", "\\\\")
        class_tokens = str(subset["class_tokens"]).replace('"', '\\"')
        lines.extend(
            [
                "[[datasets.subsets]]",
                f'image_dir = "{image_dir}"',
                f'class_tokens = "{class_tokens}"',
                f"num_repeats = {repeats}",
                "",
            ]
        )
    return "\n".join(lines)


def build_command(
    *,
    python: str,
    sd_scripts_dir: Path,
    base_model: str,
    dataset_config: Path,
    output_dir: Path,
    output_name: str,
    steps: int,
    learning_rate: float,
    network_dim: int,
    network_alpha: int,
    batch_size: int,
    resolution: str,
    optimizer_type: str,
    attention_backend: str,
    max_data_loader_workers: int,
    persistent_data_loader_workers: bool,
    save_state: bool,
    save_state_every_steps: int,
    resume_state: Path | None,
) -> list[str]:
    train_script = sd_scripts_dir / "sdxl_train_network.py"
    command = [
        python,
        "-u",
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        "2",
        str(train_script),
        "--pretrained_model_name_or_path",
        base_model,
        "--dataset_config",
        str(dataset_config),
        "--output_dir",
        str(output_dir),
        "--output_name",
        output_name,
        "--save_model_as",
        "safetensors",
        "--network_module",
        "networks.lora",
        "--network_dim",
        str(network_dim),
        "--network_alpha",
        str(network_alpha),
        "--resolution",
        resolution,
        "--train_batch_size",
        str(batch_size),
        "--max_train_steps",
        str(steps),
        "--learning_rate",
        str(learning_rate),
        "--optimizer_type",
        str(optimizer_type or "AdamW"),
        "--lr_scheduler",
        "cosine",
        "--mixed_precision",
        "bf16",
        "--cache_latents",
        "--max_data_loader_n_workers",
        str(max(0, int(max_data_loader_workers))),
        "--caption_extension",
        ".txt",
    ]
    attention_choice = str(attention_backend or "").strip().lower()
    if attention_choice == "xformers":
        command.append("--xformers")
    elif attention_choice == "sdpa":
        command.append("--sdpa")
    if bool(persistent_data_loader_workers):
        command.append("--persistent_data_loader_workers")
    if save_state:
        command.extend(["--save_state", "--save_every_n_steps", str(max(1, int(save_state_every_steps)))])
    if resume_state is not None:
        command.extend(["--resume", str(resume_state)])
    return command


def find_latest_state_dir(output_dir: Path, output_name: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{output_name}-step*-state"):
        if not path.is_dir():
            continue
        match = re.search(r"-step(\d+)-state$", path.name)
        if not match:
            continue
        try:
            step_no = int(match.group(1))
        except Exception:
            continue
        candidates.append((step_no, path))
    if candidates:
        candidates.sort(key=lambda row: row[0], reverse=True)
        return candidates[0][1].resolve()

    last_state = output_dir / f"{output_name}-state"
    if last_state.exists() and last_state.is_dir():
        return last_state.resolve()
    return None


if __name__ == "__main__":
    main()
