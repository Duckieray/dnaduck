#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNADuck - identity clustering and export toolkit.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML")

    subparsers = parser.add_subparsers(dest="command")

    scan = subparsers.add_parser("scan", help="Scan directory recursively and update identity database")
    scan.add_argument("--input-folder", type=Path, default=None, help="Override input folder")
    scan.add_argument("--output-folder", type=Path, default=None, help="Override output folder")

    scan_recluster = subparsers.add_parser(
        "scan-recluster",
        help="Reset DB and rescan directory recursively from scratch",
    )
    scan_recluster.add_argument("--input-folder", type=Path, default=None, help="Override input folder")
    scan_recluster.add_argument("--output-folder", type=Path, default=None, help="Override output folder")

    export_lora = subparsers.add_parser("export-lora", help="Export LoRA-ready dataset views from identity DB")
    export_lora.add_argument("--output-folder", type=Path, default=None, help="Override output folder")
    export_lora.add_argument("--min-images", type=int, default=None, help="Minimum images per identity")
    export_lora.add_argument(
        "--identity-id",
        dest="identity_ids",
        action="append",
        type=int,
        default=None,
        help="Export only this identity ID (repeat flag for multiple IDs)",
    )

    train_lora = subparsers.add_parser("train-lora", help="Run optional configured LoRA training command")
    train_lora.add_argument("--output-folder", type=Path, default=None, help="Override output folder")
    train_lora.add_argument("--min-images", type=int, default=None, help="Minimum images per identity")
    train_lora.add_argument(
        "--identity-id",
        dest="identity_ids",
        action="append",
        type=int,
        default=None,
        help="Train from dataset prepared with only this identity ID (repeat for multiple IDs)",
    )
    train_lora.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Run export-lora before launching training.",
    )

    identities = subparsers.add_parser("identities", help="List known identities")
    identities.add_argument("--min-members", type=int, default=1, help="Minimum members to include")

    search = subparsers.add_parser("search", help="Search closest identities for one image path")
    search.add_argument("image_path", type=Path, help="Image path to evaluate")
    search.add_argument("--top-k", type=int, default=5, help="Number of candidates")

    relabel = subparsers.add_parser("label", help="Set or clear an identity label")
    relabel.add_argument("identity_id", type=int, help="Identity ID")
    relabel.add_argument("--text", type=str, default=None, help="Label text (omit to clear)")

    merge = subparsers.add_parser("merge", help="Merge identity groups")
    merge.add_argument("target_id", type=int, help="Identity to keep")
    merge.add_argument("source_ids", nargs="+", type=int, help="Identity IDs to merge into target")

    identity_detail = subparsers.add_parser("identity", help="Show one identity with image members")
    identity_detail.add_argument("identity_id", type=int, help="Identity ID")
    identity_detail.add_argument("--limit", type=int, default=120, help="Page size")
    identity_detail.add_argument("--offset", type=int, default=0, help="Page offset")

    image_action = subparsers.add_parser("image-action", help="Apply action to one tracked image")
    image_action.add_argument("action", type=str, help="remove | blacklist | restore")
    image_action.add_argument("image_path", type=Path, help="Image path")

    subparsers.add_parser("images", help="List tracked images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from core.service import (
        apply_image_action,
        export_lora,
        get_identities,
        get_identity_detail,
        list_images,
        merge_identity_groups,
        relabel_identity,
        scan_images,
        scan_recluster_from_scratch,
        search_by_image,
        trigger_lora_training,
    )
    from core.utils import configure_logging, load_config

    config_path = args.config.resolve()
    base_config = load_config(config_path)
    configure_logging(str(base_config.get("log_level", "INFO")))

    command = args.command or "scan"
    if command == "scan":
        overrides = {
            "input_folder": getattr(args, "input_folder", None),
            "output_folder": getattr(args, "output_folder", None),
        }
        result = scan_images(config_path=config_path, overrides=overrides)
        print("DNADuck scan complete")
        for key in (
            "scan_id",
            "discovered_count",
            "processed_count",
            "assigned_count",
            "noise_count",
            "no_face_count",
            "identity_count",
            "database_path",
            "output_folder",
        ):
            print(f"{key}: {result[key]}")
        return

    if command == "scan-recluster":
        overrides = {
            "input_folder": getattr(args, "input_folder", None),
            "output_folder": getattr(args, "output_folder", None),
        }
        result = scan_recluster_from_scratch(config_path=config_path, overrides=overrides)
        print("DNADuck scan-recluster complete")
        for key in (
            "scan_id",
            "discovered_count",
            "processed_count",
            "assigned_count",
            "noise_count",
            "no_face_count",
            "identity_count",
            "database_path",
            "output_folder",
            "recluster_from_scratch",
        ):
            print(f"{key}: {result.get(key)}")
        return

    if command == "export-lora":
        overrides = {"output_folder": args.output_folder}
        if args.min_images is not None:
            overrides["lora_min_images"] = int(args.min_images)
        if args.identity_ids:
            overrides["lora_identity_ids"] = sorted({int(v) for v in args.identity_ids if int(v) > 0})
        result = export_lora(config_path=config_path, overrides=overrides)
        print("DNADuck LoRA export complete")
        print(f"identities_exported: {result['identities_exported']}")
        print(f"images_exported: {result['images_exported']}")
        print(f"caption_mode: {result['caption_mode']}")
        print(f"rich_captioning: {result['rich_captioning_status']}")
        print(f"requested_identity_ids: {result['requested_identity_ids']}")
        print(f"exported_identity_ids: {result['exported_identity_ids']}")
        print(f"output_folder: {result['output_folder']}")
        return

    if command == "identities":
        rows = get_identities(config_path=config_path, min_members=int(args.min_members))
        for row in rows:
            print(
                f"id={row['identity_id']} members={row['member_count']} label={row['label']} "
                f"updated={row['updated_at']}"
            )
        print(f"total={len(rows)}")
        return

    if command == "train-lora":
        export_result = None
        prepare_dataset = bool(args.prepare_dataset or args.min_images is not None or args.identity_ids)
        if prepare_dataset:
            export_overrides = {"output_folder": args.output_folder}
            if args.min_images is not None:
                export_overrides["lora_min_images"] = int(args.min_images)
            if args.identity_ids:
                export_overrides["lora_identity_ids"] = sorted({int(v) for v in args.identity_ids if int(v) > 0})
            export_result = export_lora(config_path=config_path, overrides=export_overrides)
            print("DNADuck LoRA export complete")
            print(f"identities_exported: {export_result['identities_exported']}")
            print(f"images_exported: {export_result['images_exported']}")
            print(f"requested_identity_ids: {export_result['requested_identity_ids']}")
            print(f"exported_identity_ids: {export_result['exported_identity_ids']}")
        try:
            result = trigger_lora_training(
                config_path=config_path,
                overrides={"output_folder": args.output_folder},
            )
        except ValueError as exc:
            print(f"error: {exc}")
            return
        if export_result is not None:
            print("prepared_dataset: true")
        print(f"returncode: {result['returncode']}")
        print(f"dataset_dir: {result['dataset_dir']}")
        print(f"command: {' '.join(result['command'])}")
        if result["stdout"]:
            print("stdout:")
            print(result["stdout"])
        if result["stderr"]:
            print("stderr:")
            print(result["stderr"])
        return

    if command == "search":
        rows = search_by_image(config_path=config_path, image_path=args.image_path, top_k=int(args.top_k))
        for row in rows:
            print(
                f"identity_id={row['identity_id']} similarity={row['similarity']:.4f} "
                f"distance={row['distance']:.4f} members={row['member_count']} label={row['label']}"
            )
        print(f"total={len(rows)}")
        return

    if command == "label":
        updated = relabel_identity(config_path=config_path, identity_id=int(args.identity_id), label=args.text)
        print("updated=true" if updated else "updated=false")
        return

    if command == "merge":
        merge_identity_groups(config_path=config_path, target_id=int(args.target_id), source_ids=args.source_ids)
        print("merge_complete=true")
        return

    if command == "identity":
        detail = get_identity_detail(
            config_path=config_path,
            identity_id=int(args.identity_id),
            limit=max(1, int(args.limit)),
            offset=max(0, int(args.offset)),
        )
        print(json.dumps(detail, ensure_ascii=True))
        return

    if command == "image-action":
        try:
            result = apply_image_action(
                config_path=config_path,
                image_path=args.image_path,
                action=str(args.action),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return
        print(json.dumps(result, ensure_ascii=True))
        return

    if command == "images":
        rows = list_images(config_path=config_path)
        for row in rows:
            print(f"path={row['path']} status={row['status']} identity_id={row['identity_id']}")
        print(f"total={len(rows)}")
        return


if __name__ == "__main__":
    main()
