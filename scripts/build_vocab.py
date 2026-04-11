#!/usr/bin/env python3
"""
scripts/build_vocab.py — Build Gloss Vocabulary from Dataset Metadata

DESCRIPTION:
    Scans dataset directories and JSON metadata files to build a unified
    vocabulary mapping: gloss_label ↔ integer_id. Saves as JSON.

USAGE:
    python scripts/build_vocab.py --dataset asl_alphabet
    python scripts/build_vocab.py --dataset wlasl
    python scripts/build_vocab.py --all

INPUTS:
    --dataset   Name of dataset to build vocab from
    --all       Build vocabularies for all available datasets
    --config    Path to config.yaml

OUTPUTS:
    - data/processed/{dataset}/vocab.json — { "label_to_id": {...}, "id_to_label": {...} }
    - data/processed/unified_vocab.json — merged vocabulary across all datasets
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_vocab")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Load the master config.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_asl_alphabet_vocab(config: dict) -> dict:
    """
    Build vocabulary from ASL Alphabet dataset.
    Uses class directory names as labels.
    """
    ds_config = config["datasets"]["asl_alphabet"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]
    processed_path = PROJECT_ROOT / config["paths"]["data"]["processed"] / "asl_alphabet"

    # Try to get classes from raw data directories
    train_dir = raw_path / ds_config["train_dir"]
    if train_dir.exists():
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    else:
        # Fall back to config
        classes = ds_config.get("classes", [])

    if not classes:
        logger.error("No classes found for ASL Alphabet")
        return {}

    label_to_id = {label: idx for idx, label in enumerate(classes)}
    id_to_label = {idx: label for idx, label in enumerate(classes)}

    vocab = {
        "dataset": "asl_alphabet",
        "num_classes": len(classes),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "classes": classes,
    }

    # Save to processed directory
    processed_path.mkdir(parents=True, exist_ok=True)
    vocab_path = processed_path / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    logger.info(f"✅ ASL Alphabet vocab: {len(classes)} classes → {vocab_path}")
    for i, label in enumerate(classes):
        logger.debug(f"   {i:3d}: {label}")

    return vocab


def build_wlasl_vocab(config: dict) -> dict:
    """
    Build vocabulary from WLASL dataset.
    Reads from WLASL JSON metadata if available.
    """
    ds_config = config["datasets"]["wlasl"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]
    processed_path = PROJECT_ROOT / config["paths"]["data"]["processed"] / "wlasl"

    # Look for WLASL JSON file
    json_files = list(raw_path.glob("*.json")) + list(raw_path.glob("**/*.json"))

    if not json_files:
        # Try to build from directory names
        video_dirs = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])
        if video_dirs:
            classes = video_dirs
        else:
            logger.warning("No WLASL metadata or video directories found")
            return {}
    else:
        # Parse WLASL JSON format
        json_path = json_files[0]
        logger.info(f"Reading WLASL metadata from: {json_path}")
        with open(json_path, "r") as f:
            wlasl_data = json.load(f)

        if isinstance(wlasl_data, list):
            classes = sorted([entry.get("gloss", "") for entry in wlasl_data if "gloss" in entry])
        elif isinstance(wlasl_data, dict):
            classes = sorted(wlasl_data.keys())
        else:
            logger.error(f"Unexpected WLASL JSON format")
            return {}

    label_to_id = {label: idx for idx, label in enumerate(classes)}
    id_to_label = {idx: label for idx, label in enumerate(classes)}

    vocab = {
        "dataset": "wlasl",
        "num_classes": len(classes),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "classes": classes,
    }

    processed_path.mkdir(parents=True, exist_ok=True)
    vocab_path = processed_path / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    logger.info(f"✅ WLASL vocab: {len(classes)} classes → {vocab_path}")
    return vocab


def build_unified_vocab(vocabs: list[dict], config: dict) -> None:
    """Merge all dataset vocabularies into one unified mapping."""
    processed_path = PROJECT_ROOT / config["paths"]["data"]["processed"]
    all_labels = set()

    for vocab in vocabs:
        all_labels.update(vocab.get("classes", []))

    all_labels = sorted(all_labels)
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    id_to_label = {idx: label for idx, label in enumerate(all_labels)}

    unified = {
        "dataset": "unified",
        "num_classes": len(all_labels),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "classes": all_labels,
        "source_datasets": [v.get("dataset", "unknown") for v in vocabs],
    }

    vocab_path = processed_path / "unified_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(unified, f, indent=2)

    logger.info(f"✅ Unified vocab: {len(all_labels)} total classes → {vocab_path}")


def main():
    parser = argparse.ArgumentParser(description="ASL Bridge — Vocabulary Builder")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["asl_alphabet", "wlasl"],
        help="Build vocab for a specific dataset",
    )
    parser.add_argument("--all", action="store_true", help="Build all vocabularies")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to config.yaml",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))

    vocabs = []

    if args.dataset == "asl_alphabet" or args.all:
        vocab = build_asl_alphabet_vocab(config)
        if vocab:
            vocabs.append(vocab)

    if args.dataset == "wlasl" or args.all:
        vocab = build_wlasl_vocab(config)
        if vocab:
            vocabs.append(vocab)

    if args.all and len(vocabs) > 1:
        build_unified_vocab(vocabs, config)

    if not vocabs:
        logger.warning("No vocabularies built. Check that datasets are available.")


if __name__ == "__main__":
    main()
