"""Kaggle Dataset checkpoint transport. Use one private dataset per training run.

This module is intentionally independent from Colab's filesystem-backed backup.
Never treat CLI upload acceptance as durable completion: download and hash-check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from zipfile import ZipFile


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle(*args, timeout=900):
    result = subprocess.run(
        ["kaggle", "datasets", *args],
        text=True, capture_output=True, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"Kaggle CLI failed (exit={result.returncode}): {result.stderr[-2000:]} "
            f"{result.stdout[-2000:]}"
        )
    return result.stdout


def verify_dataset_file(dataset, filename, expected_sha, *, max_attempts=12):
    """Independently download from Kaggle, never verify against the upload folder."""
    with tempfile.TemporaryDirectory(prefix="visdrone-kaggle-verify-") as directory:
        directory = Path(directory)
        for attempt in range(max_attempts):
            try:
                listing = kaggle("files", dataset, timeout=60)
                if filename in listing:
                    kaggle("download", "-d", dataset, "-f", filename,
                           "-p", str(directory), "--force", timeout=900)
                    file = directory / filename
                    if not file.exists():
                        archives = list(directory.glob("*.zip"))
                        if len(archives) != 1:
                            raise RuntimeError("Unexpected Kaggle download layout")
                        with ZipFile(archives[0]) as archive:
                            if filename not in archive.namelist():
                                raise RuntimeError("Checkpoint not in downloaded ZIP")
                            with archive.open(filename) as src, open(file, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                    actual = sha256(file)
                    if actual == expected_sha:
                        return
                    # An old dataset version may be temporarily served during publish.
                    raise RuntimeError("Downloaded SHA-256 differs from submitted checkpoint")
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt == max_attempts - 1:
                    raise
                time.sleep(5)
        raise RuntimeError("Kaggle did not publish the expected checkpoint")


def publish_checkpoint(run, dataset, *, version_note=None):
    """Publish one run's last.pt; validate round-trip before reporting success.

    Dataset must exist and be dedicated to this run. No --delete-old-versions.
    Other run data cannot silently enter the publication.
    """
    run = Path(run).resolve()
    checkpoint = run / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    required = (
        "config.json", "provenance.json", "data.json",
        "preprocessing.json", "storage.json",
        "metrics.json", "summary.json", "commit.json",
        "best_weights.pt", "backup.json",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing backup metadata: {missing}")
    expected = sha256(checkpoint)
    with tempfile.TemporaryDirectory(prefix="visdrone-kaggle-stage-") as directory:
        stage = Path(directory)
        # Stage small metadata, but do not make a second copy of multi-GB last.pt
        # until required by the Kaggle CLI's directory-only upload interface.
        for name in required:
            shutil.copyfile(run / name, stage / name)
        shutil.copyfile(checkpoint, stage / "last.pt")
        manifest = {
            "dataset": dataset,
            "checkpoint_sha256": expected,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "run_name": run.name,
            "files_sha256": {
                name: sha256(stage / name)
                for name in (*required, "last.pt")
            },
        }
        (stage / "kaggle-backup.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (stage / "dataset-metadata.json").write_text(
            json.dumps({"title": f"VisDrone checkpoint {run.name}",
                        "id": dataset, "licenses": [{"name": "CC0-1.0"}]}) + "\n",
            encoding="utf-8")
        kaggle("version", "-p", str(stage),
               "-m", version_note or f"Full checkpoint {run.name}",
               timeout=1800)
    verify_dataset_files(dataset, manifest["files_sha256"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", required=True,
                        help="EXISTING private Kaggle Dataset dedicated to this run")
    args = parser.parse_args()
    result = publish_checkpoint(args.run, args.dataset)
    print(json.dumps(result, indent=2))
    print("KAGGLE_BACKUP_VERIFIED")


if __name__ == "__main__":
    main()
