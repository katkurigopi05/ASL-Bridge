#!/usr/bin/env python3
"""
scripts/extract_keypoints.py — Batch Keypoint Extraction for ASL Bridge

DESCRIPTION:
    Batch-processes a dataset of video files or image directories using
    MediaPipe Holistic to extract hand/pose/face landmarks. Stores results
    as .npy files for training.

USAGE:
    python scripts/extract_keypoints.py --dataset asl_alphabet
    python scripts/extract_keypoints.py --dataset wlasl --split train
    python scripts/extract_keypoints.py --input path/to/video.mp4 --output path/to/output.npy

INPUTS:
    --dataset       Dataset name (must match config.yaml)
    --split         Dataset split: train, val, test (default: all)
    --input         Single video or image file path
    --output        Output .npy file path (for single file mode)
    --config        Path to config.yaml

OUTPUTS:
    - .npy files containing extracted keypoint arrays
    - Shape: (num_frames, num_features) for videos
    - Shape: (1, num_features) for single images
    - Saved to: data/processed/{dataset}/{split}/{class}/{sample_id}.npy
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.extractor import MediaPipeExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_keypoints")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Load the master config.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_asl_alphabet(config: dict, split: str = "all") -> None:
    """
    Extract keypoints from ASL Alphabet image dataset.
    Each image produces a single-frame keypoint vector.
    """
    ds_config = config["datasets"]["asl_alphabet"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]
    processed_path = PROJECT_ROOT / config["paths"]["data"]["processed"] / "asl_alphabet"

    # Determine which splits to process
    splits_to_process = []
    if split in ("all", "train"):
        train_dir = raw_path / ds_config["train_dir"]
        if train_dir.exists():
            splits_to_process.append(("train", train_dir))
        else:
            logger.warning(f"Train directory not found: {train_dir}")

    if split in ("all", "test"):
        test_dir = raw_path / ds_config["test_dir"]
        if test_dir.exists():
            splits_to_process.append(("test", test_dir))
        else:
            logger.warning(f"Test directory not found: {test_dir}")

    if not splits_to_process:
        logger.error("No data directories found. Run download_data.py --copy-local first.")
        return

    extractor = MediaPipeExtractor(config)

    for split_name, split_dir in splits_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split_name} split from: {split_dir}")
        logger.info(f"{'='*60}")

        # For train: iterate over class directories
        if split_name == "train":
            class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            logger.info(f"Found {len(class_dirs)} classes")

            for class_dir in class_dirs:
                class_name = class_dir.name
                output_dir = processed_path / split_name / class_name
                output_dir.mkdir(parents=True, exist_ok=True)

                image_files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                logger.info(f"  {class_name}: {len(image_files)} images")

                success_count = 0
                fail_count = 0

                for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
                    sample_id = img_path.stem
                    output_file = output_dir / f"{sample_id}.npy"

                    # Skip if already extracted
                    if output_file.exists():
                        success_count += 1
                        continue

                    try:
                        keypoints = extractor.extract_from_image(str(img_path))
                        if keypoints is not None:
                            np.save(str(output_file), keypoints)
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        logger.debug(f"    Failed: {img_path.name} — {e}")
                        fail_count += 1

                logger.info(f"  {class_name}: ✅ {success_count} | ❌ {fail_count}")

            # Cleanup persistent MediaPipe instance
            extractor.close_static()

        # For test: single directory with labeled files
        elif split_name == "test":
            output_dir = processed_path / split_name
            output_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))
            logger.info(f"Found {len(image_files)} test images")

            for img_path in tqdm(image_files, desc="  test"):
                sample_id = img_path.stem
                output_file = output_dir / f"{sample_id}.npy"

                if output_file.exists():
                    continue

                try:
                    keypoints = extractor.extract_from_image(str(img_path))
                    if keypoints is not None:
                        np.save(str(output_file), keypoints)
                except Exception as e:
                    logger.debug(f"  Failed: {img_path.name} — {e}")


def extract_single(input_path: Path, output_path: Path, config: dict) -> None:
    """Extract keypoints from a single video or image file."""
    extractor = MediaPipeExtractor(config)

    if input_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        logger.info(f"Extracting from video: {input_path}")
        frames = []
        for frame_idx, landmarks in extractor.extract_from_video(str(input_path)):
            if landmarks is not None:
                frames.append(landmarks)
        if frames:
            keypoints = np.stack(frames)
            np.save(str(output_path), keypoints)
            logger.info(f"✅ Saved {keypoints.shape} to {output_path}")
        else:
            logger.error("❌ No landmarks detected in video")
    else:
        logger.info(f"Extracting from image: {input_path}")
        keypoints = extractor.extract_from_image(str(input_path))
        if keypoints is not None:
            np.save(str(output_path), keypoints)
            logger.info(f"✅ Saved {keypoints.shape} to {output_path}")
        else:
            logger.error("❌ No landmarks detected in image")


def main():
    parser = argparse.ArgumentParser(
        description="ASL Bridge — Batch Keypoint Extraction",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["asl_alphabet", "wlasl", "how2sign", "ms_asl"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which split to process",
    )
    parser.add_argument("--input", type=str, help="Single file to process")
    parser.add_argument("--output", type=str, help="Output .npy path (single file mode)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to config.yaml",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))

    if args.input:
        if not args.output:
            logger.error("--output is required when using --input")
            sys.exit(1)
        extract_single(Path(args.input), Path(args.output), config)
    elif args.dataset == "asl_alphabet":
        extract_asl_alphabet(config, args.split)
    elif args.dataset:
        logger.error(f"Extraction for {args.dataset} is not yet implemented")
        logger.info("Currently supported: asl_alphabet")
    else:
        logger.error("Specify --dataset or --input")
        sys.exit(1)


if __name__ == "__main__":
    main()
