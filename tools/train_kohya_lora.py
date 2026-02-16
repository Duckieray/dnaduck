#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--num-repeats", type=int, default=10, help="Repeats per identity subset")
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
    dataset_config.write_text(
        build_dataset_toml(subsets=subsets, repeats=max(1, int(args.num_repeats))),
        encoding="utf-8",
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
    )

    print("Kohya trainer command:")
    print(" ".join(command))
    print(f"Dataset config: {dataset_config}")
    print(f"Subsets: {len(subsets)}")

    if args.dry_run:
        return

    result = subprocess.run(
        command,
        cwd=str(sd_scripts_dir),
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


def build_dataset_toml(subsets: list[dict], repeats: int) -> str:
    lines = [
        "[[datasets]]",
        "resolution = [1024, 1024]",
        "batch_size = 1",
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
) -> list[str]:
    train_script = sd_scripts_dir / "sdxl_train_network.py"
    return [
        "accelerate",
        "launch",
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
        "AdamW8bit",
        "--lr_scheduler",
        "cosine",
        "--mixed_precision",
        "bf16",
        "--cache_latents",
        "--caption_extension",
        ".txt",
    ]


if __name__ == "__main__":
    main()

