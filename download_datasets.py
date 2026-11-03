
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download benchmark datasets from Hugging Face using hf-mirror.

Datasets:
1. ShareGPT
2. GSM8K
3. LongBench v1 (21 tasks)
4. HumanEval

Output:
~/datasets/pd_benchmark/
"""

import os

# ============================================================
# 1. Hugging Face mirror configuration
# Must be set before importing huggingface libraries.
# ============================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import zipfile
import shutil
import json
import argparse
import traceback
from pathlib import Path, PurePosixPath
from datetime import datetime

from huggingface_hub import hf_hub_download
from datasets import load_dataset


# ============================================================
# 2. Configuration
# ============================================================

OUTPUT_DIR = Path.home() / "datasets" / "pd_benchmark"

DATASETS = {
    "sharegpt",
    "gsm8k",
    "longbench",
    "humaneval",
}

# Original LongBench v1: 21 tasks
LONGBENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]


# ============================================================
# 3. Utility functions
# ============================================================

def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def save_manifest(manifest):
    path = OUTPUT_DIR / "download_manifest.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )


def write_jsonl(dataset, output_path, force=False):
    """
    Save Hugging Face Dataset to JSONL.
    Uses a temporary file to avoid leaving a partial output.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        log(f"Already exists, skip: {output_path}")
        return sum(1 for _ in open(output_path, encoding="utf-8"))

    tmp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    count = 0

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:

            for row in dataset:
                f.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                count += 1

        os.replace(tmp_path, output_path)

        log(f"Saved: {output_path}")
        log(f"Records: {count}")

        return count

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ============================================================
# 4. ShareGPT
# ============================================================

def download_sharegpt(force=False):

    log("=" * 60)
    log("Downloading ShareGPT")

    repo_id = "anon8231489123/ShareGPT_Vicuna_unfiltered"

    filename = "ShareGPT_V3_unfiltered_cleaned_split.json"

    output_dir = OUTPUT_DIR / "sharegpt"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    if output_path.exists() and not force:
        log(f"Already exists: {output_path}")
        return str(output_path)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(output_dir),
        force_download=force,
    )

    log(f"ShareGPT downloaded: {downloaded_path}")

    return str(downloaded_path)


# ============================================================
# 5. GSM8K
# ============================================================

def download_gsm8k(force=False):

    log("=" * 60)
    log("Downloading GSM8K")

    output_dir = OUTPUT_DIR / "gsm8k"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"]

    if not force and all(
        (output_dir / f"{split}.jsonl").exists()
        for split in splits
    ):
        log("GSM8K already downloaded")
        return

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        download_mode=(
            "force_redownload"
            if force else "reuse_dataset_if_exists"
        ),
    )

    for split in splits:

        if split not in dataset:
            log(f"Warning: split {split} not found")
            continue

        output_path = output_dir / f"{split}.jsonl"

        write_jsonl(
            dataset[split],
            output_path,
            force=force
        )

    log("GSM8K completed")


# ============================================================
# 6. LongBench v1
# ============================================================


def download_longbench(force=False):

    log("=" * 60)
    log("Downloading LongBench v1 from raw JSONL archive")

    repo_id = "THUDM/LongBench"

    output_dir = OUTPUT_DIR / "longbench"
    output_dir.mkdir(parents=True, exist_ok=True)

    download_dir = OUTPUT_DIR / "_downloads" / "longbench"
    download_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Download official data.zip through hf-mirror
    # --------------------------------------------------------

    log("Downloading LongBench data.zip ...")

    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename="data.zip",
        repo_type="dataset",
        local_dir=str(download_dir),
        force_download=force,
    )

    log(f"Archive downloaded: {zip_path}")

    # --------------------------------------------------------
    # 2. Extract 21 JSONL task files
    # --------------------------------------------------------

    successful = []
    failed = []

    with zipfile.ZipFile(zip_path, "r") as zf:

        archive_names = zf.namelist()

        log(f"Archive contains {len(archive_names)} entries")

        for index, task in enumerate(LONGBENCH_TASKS, 1):

            filename = f"{task}.jsonl"

            log(
                f"[{index}/{len(LONGBENCH_TASKS)}] "
                f"Extracting {filename}"
            )

            output_path = output_dir / filename

            if output_path.exists() and not force:
                log(f"Already exists, skip: {filename}")
                successful.append(task)
                continue

            # Match by basename because archive directory
            # structure may contain an additional parent folder.

            matches = [
                name for name in archive_names
                if PurePosixPath(name).name == filename
            ]

            if not matches:

                log(f"FAILED: {filename} not found in archive")

                failed.append({
                    "task": task,
                    "error": "File not found in data.zip"
                })

                continue

            try:

                archive_file = matches[0]

                tmp_path = output_path.with_suffix(".jsonl.tmp")

                count = 0

                with zf.open(archive_file, "r") as src:
                    with open(tmp_path, "wb") as dst:

                        shutil.copyfileobj(src, dst)

                # Validate JSONL before committing the file.
                with open(tmp_path, "r", encoding="utf-8") as f:

                    for line in f:

                        if not line.strip():
                            continue

                        json.loads(line)
                        count += 1

                if count == 0:
                    raise RuntimeError("Empty dataset file")

                os.replace(tmp_path, output_path)

                log(f"Saved: {output_path}")
                log(f"Records: {count}")

                successful.append(task)

            except Exception as e:

                if tmp_path.exists():
                    tmp_path.unlink()

                log(f"FAILED: {task}")
                log(repr(e))

                failed.append({
                    "task": task,
                    "error": repr(e)
                })

    # --------------------------------------------------------
    # 3. Summary
    # --------------------------------------------------------

    log("=" * 60)
    log(f"LongBench successful: {len(successful)}")
    log(f"LongBench failed: {len(failed)}")

    return {
        "successful": successful,
        "failed": failed,
    }


# ============================================================
# 7. HumanEval
# ============================================================

def download_humaneval(force=False):

    log("=" * 60)
    log("Downloading HumanEval")

    output_dir = OUTPUT_DIR / "humaneval"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "test.jsonl"

    if output_path.exists() and not force:
        log("HumanEval already downloaded")
        return

    dataset = load_dataset(
        "openai/openai_humaneval",
        split="test",
        download_mode=(
            "force_redownload"
            if force else "reuse_dataset_if_exists"
        ),
    )

    write_jsonl(
        dataset,
        output_path,
        force=force
    )

    log("HumanEval completed")


# ============================================================
# 8. Main
# ============================================================

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets via HF mirror"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Dataset output directory"
    )

    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Select datasets to download"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload and overwrite output files"
    )

    args = parser.parse_args()


    OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Hugging Face Dataset Downloader")
    log("=" * 60)

    log(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Selected datasets: {args.only}")

    manifest = {
        "mirror": os.environ["HF_ENDPOINT"],
        "output_dir": str(OUTPUT_DIR),
        "start_time": datetime.now().isoformat(),
        "results": {},
    }

    download_functions = {
        "sharegpt": download_sharegpt,
        "gsm8k": download_gsm8k,
        "longbench": download_longbench,
        "humaneval": download_humaneval,
    }

    for name in args.only:

        try:

            result = download_functions[name](
                force=args.force
            )

            manifest["results"][name] = {
                "status": "success",
                "result": result,
            }

        except Exception as e:

            log(f"Dataset {name} failed: {repr(e)}")
            traceback.print_exc()

            manifest["results"][name] = {
                "status": "failed",
                "error": repr(e),
            }

        finally:
            save_manifest(manifest)

    manifest["end_time"] = datetime.now().isoformat()

    save_manifest(manifest)

    log("=" * 60)
    log("Download finished")
    log(f"Manifest: {OUTPUT_DIR / 'download_manifest.json'}")
    log("=" * 60)


if __name__ == "__main__":
    main()

