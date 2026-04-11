#!/usr/bin/env python3
"""
scripts/download_data.py — Dataset Downloader & Verifier for ASL Bridge

DESCRIPTION:
    Checks which datasets are already present locally and either copies/symlinks
    them into the project data/raw/ directory, downloads them automatically
    (when possible), or prints manual download instructions.

SUPPORTED DATASETS:
    1. ASL Alphabet (Kaggle)        — local copy from Desktop or Kaggle API
    2. ASL MNIST (HuggingFace)      — auto-download via `datasets` library
    3. WLASL-2000 (Kaggle)          — Kaggle API download
    4. How2Sign                     — manual download (prints URL)
    5. MS-ASL (Microsoft)           — manual download (requires application)

USAGE:
    python scripts/download_data.py                    # check all datasets
    python scripts/download_data.py --dataset asl_alphabet  # check one dataset
    python scripts/download_data.py --copy-local       # copy Desktop data into project
    python scripts/download_data.py --dataset asl_mnist --download  # auto-download

INPUTS:
    --dataset       Name of a specific dataset to check/download
    --copy-local    Copy the ASL Alphabet data from Desktop into data/raw/
    --download      Attempt automatic download (where supported)
    --config        Path to config.yaml (default: config.yaml)

OUTPUTS:
    - Status report for each dataset (FOUND / MISSING / PARTIAL)
    - Copies or symlinks data into data/raw/{dataset}/ when --copy-local is used
    - Downloads data when --download is used (ASL MNIST, WLASL)
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

# ── Setup logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_data")

# ── Resolve project root ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Load the master config.yaml."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_asl_alphabet(config: dict, copy_local: bool = False) -> bool:
    """
    Check if ASL Alphabet dataset exists.
    Optionally copy from the user's Desktop location into the project.
    """
    ds_config = config["datasets"]["asl_alphabet"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]
    local_source = Path(ds_config.get("local_source", ""))

    # Check if already in project
    train_path = raw_path / ds_config["train_dir"]
    if train_path.exists():
        num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
        logger.info(f"✅ ASL Alphabet FOUND at {raw_path}")
        logger.info(f"   Classes: {num_classes}, Expected: {ds_config['num_classes']}")
        if num_classes >= ds_config["num_classes"]:
            return True
        else:
            logger.warning(f"   ⚠️  Only {num_classes}/{ds_config['num_classes']} classes found")
            return False

    # Check Desktop source
    if local_source.exists():
        logger.info(f"📂 ASL Alphabet found on Desktop: {local_source}")
        source_train = local_source / ds_config["train_dir"]
        source_test = local_source / ds_config["test_dir"]

        if source_train.exists():
            num_classes = len([d for d in source_train.iterdir() if d.is_dir()])
            num_samples = sum(1 for _ in source_train.rglob("*.jpg"))
            logger.info(f"   Train: {num_classes} classes, {num_samples} images")
        if source_test.exists():
            num_test = sum(1 for _ in source_test.rglob("*.jpg"))
            logger.info(f"   Test: {num_test} images")

        if copy_local:
            logger.info(f"📋 Copying ASL Alphabet into project: {raw_path}")
            raw_path.mkdir(parents=True, exist_ok=True)

            # Copy train directory
            dest_train = raw_path / Path(ds_config["train_dir"]).parent
            if not dest_train.exists():
                logger.info(f"   Copying train data...")
                shutil.copytree(local_source / Path(ds_config["train_dir"]).parent, dest_train)
                logger.info(f"   ✅ Train data copied")

            # Copy test directory
            dest_test = raw_path / Path(ds_config["test_dir"]).parent
            if not dest_test.exists():
                logger.info(f"   Copying test data...")
                shutil.copytree(local_source / Path(ds_config["test_dir"]).parent, dest_test)
                logger.info(f"   ✅ Test data copied")

            return True
        else:
            logger.info("   💡 Run with --copy-local to copy into the project")
            return False
    else:
        logger.warning(f"❌ ASL Alphabet NOT FOUND")
        logger.info("   📥 Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        logger.info(f"   📂 Extract to: {raw_path}/")
        logger.info("   OR run: kaggle datasets download -d grassknoted/asl-alphabet")
        return False


def check_asl_mnist(config: dict, download: bool = False) -> bool:
    """
    Check if ASL MNIST dataset exists.
    Can auto-download from HuggingFace.
    """
    ds_config = config["datasets"]["asl_mnist"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]

    # Check if already downloaded
    if raw_path.exists() and any(raw_path.iterdir()):
        num_files = sum(1 for _ in raw_path.rglob("*") if _.is_file())
        logger.info(f"✅ ASL MNIST FOUND at {raw_path} ({num_files} files)")
        return True

    if download:
        logger.info("📥 Downloading ASL MNIST from HuggingFace...")
        try:
            from datasets import load_dataset

            ds = load_dataset(ds_config["hf_repo"])
            raw_path.mkdir(parents=True, exist_ok=True)

            # Save splits as individual image directories
            for split_name in ds.keys():
                split_dir = raw_path / split_name
                split_dir.mkdir(exist_ok=True)

                split_data = ds[split_name]
                logger.info(f"   Processing {split_name}: {len(split_data)} samples")

                # Save metadata
                metadata = {
                    "split": split_name,
                    "num_samples": len(split_data),
                    "features": str(split_data.features),
                }
                with open(split_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # Save the dataset to disk in arrow format for fast loading
                split_data.save_to_disk(str(split_dir / "arrow_data"))
                logger.info(f"   ✅ Saved {split_name} to {split_dir}")

            logger.info(f"✅ ASL MNIST downloaded to {raw_path}")
            return True

        except ImportError:
            logger.error("   ❌ 'datasets' library not installed. Run: pip install datasets")
            return False
        except Exception as e:
            logger.error(f"   ❌ Download failed: {e}")
            return False
    else:
        logger.warning(f"❌ ASL MNIST NOT FOUND at {raw_path}")
        logger.info(f"   📥 Auto-download: python scripts/download_data.py --dataset asl_mnist --download")
        logger.info(f"   📦 HuggingFace repo: {ds_config['hf_repo']}")
        logger.info(f"   💻 Manual: from datasets import load_dataset; ds = load_dataset('{ds_config['hf_repo']}')")
        return False


def check_wlasl(config: dict, download: bool = False) -> bool:
    """
    Check if WLASL-2000 dataset exists.
    Can attempt Kaggle API download.
    """
    ds_config = config["datasets"]["wlasl"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]

    if raw_path.exists() and any(raw_path.iterdir()):
        num_files = sum(1 for _ in raw_path.rglob("*") if _.is_file())
        logger.info(f"✅ WLASL FOUND at {raw_path} ({num_files} files)")
        return True

    if download:
        logger.info("📥 Attempting WLASL download via Kaggle API...")
        try:
            import subprocess

            raw_path.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", ds_config["kaggle_dataset"],
                    "-p", str(raw_path),
                    "--unzip",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"✅ WLASL downloaded to {raw_path}")
                return True
            else:
                logger.error(f"   ❌ Kaggle download failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.error("   ❌ Kaggle CLI not found. Run: pip install kaggle")
            logger.info("   📋 Then configure: export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx")
            return False
    else:
        logger.warning(f"❌ WLASL NOT FOUND at {raw_path}")
        logger.info(f"   📥 Kaggle: kaggle datasets download -d {ds_config['kaggle_dataset']}")
        logger.info(f"   📂 Extract to: {raw_path}/")
        return False


def check_how2sign(config: dict) -> bool:
    """Check if How2Sign dataset exists. Always manual download."""
    ds_config = config["datasets"]["how2sign"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]

    if raw_path.exists() and any(raw_path.iterdir()):
        num_files = sum(1 for _ in raw_path.rglob("*") if _.is_file())
        logger.info(f"✅ How2Sign FOUND at {raw_path} ({num_files} files)")
        return True

    logger.warning(f"❌ How2Sign NOT FOUND at {raw_path}")
    logger.info(f"   📥 Download from: {ds_config['url']}")
    logger.info("   📋 This is an 80-hour dataset — download size is very large")
    logger.info("   💡 Priority: LOW — use after WLASL training is working")
    return False


def check_ms_asl(config: dict) -> bool:
    """Check if MS-ASL dataset exists. Requires Microsoft application."""
    ds_config = config["datasets"]["ms_asl"]
    raw_path = PROJECT_ROOT / ds_config["raw_path"]

    if raw_path.exists() and any(raw_path.iterdir()):
        num_files = sum(1 for _ in raw_path.rglob("*") if _.is_file())
        logger.info(f"✅ MS-ASL FOUND at {raw_path} ({num_files} files)")
        return True

    logger.warning(f"❌ MS-ASL NOT FOUND at {raw_path}")
    logger.info("   📥 Apply for access: https://www.microsoft.com/en-us/research/project/ms-asl/")
    logger.info("   📋 Requires institutional application to Microsoft Research")
    logger.info("   💡 Priority: LOW — use as secondary vocabulary after WLASL")
    return False


def print_summary(results: dict) -> None:
    """Print a summary table of dataset availability."""
    logger.info("")
    logger.info("=" * 65)
    logger.info("  ASL Bridge — Dataset Status Summary")
    logger.info("=" * 65)

    for name, status in results.items():
        icon = "✅" if status else "❌"
        logger.info(f"  {icon}  {name}")

    found = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("-" * 65)
    logger.info(f"  {found}/{total} datasets available")

    if not results.get("ASL Alphabet (Kaggle)", False):
        logger.info("")
        logger.info("  ⚠️  ASL Alphabet is the PRIMARY dataset for initial training.")
        logger.info("     You have it on your Desktop — run with --copy-local to import.")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="ASL Bridge — Dataset Downloader & Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["asl_alphabet", "asl_mnist", "wlasl", "how2sign", "ms_asl"],
        help="Check/download a specific dataset only",
    )
    parser.add_argument(
        "--copy-local",
        action="store_true",
        help="Copy ASL Alphabet data from Desktop into project",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt automatic download (ASL MNIST, WLASL)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help=f"Path to config.yaml (default: {CONFIG_PATH})",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))

    logger.info("🔍 ASL Bridge — Checking dataset availability...\n")

    results = {}

    # Define check functions
    checks = {
        "asl_alphabet": ("ASL Alphabet (Kaggle)", lambda: check_asl_alphabet(config, args.copy_local)),
        "asl_mnist": ("ASL MNIST (HuggingFace)", lambda: check_asl_mnist(config, args.download)),
        "wlasl": ("WLASL-2000 (Kaggle)", lambda: check_wlasl(config, args.download)),
        "how2sign": ("How2Sign", lambda: check_how2sign(config)),
        "ms_asl": ("MS-ASL (Microsoft)", lambda: check_ms_asl(config)),
    }

    if args.dataset:
        # Check only the specified dataset
        name, check_fn = checks[args.dataset]
        results[name] = check_fn()
    else:
        # Check all datasets
        for key, (name, check_fn) in checks.items():
            results[name] = check_fn()
            logger.info("")

    print_summary(results)


if __name__ == "__main__":
    main()
