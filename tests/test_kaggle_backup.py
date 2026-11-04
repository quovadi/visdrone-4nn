"""Offline tests for Kaggle checkpoint transport (never contact Kaggle)."""
import hashlib
import json
from pathlib import Path

import pytest

from training import kaggle_backup


def test_sha256_streaming(tmp_path):
    payload = b"visdrone checkpoint\x00" * 123
    path = tmp_path / "last.pt"
    path.write_bytes(payload)
    assert kaggle_backup.sha256(path) == hashlib.sha256(payload).hexdigest()


def test_missing_metadata_fails_before_network(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "last.pt").write_bytes(b"test")

    def unexpected(*args, **kwargs):
        raise AssertionError("Kaggle must not be called")

    monkeypatch.setattr(kaggle_backup, "kaggle", unexpected)
    with pytest.raises(FileNotFoundError, match="Missing backup metadata"):
        kaggle_backup.publish_checkpoint(run, "apricot88/test")


def test_publish_stages_one_checkpoint_and_verifies_download(tmp_path, monkeypatch):
    run = tmp_path / "full-test-run"
    run.mkdir()
    contents = b"checkpoint bytes"
    (run / "last.pt").write_bytes(contents)
    for name in (
        "config.json", "provenance.json", "data.json",
        "preprocessing.json", "storage.json",
        "metrics.json", "summary.json", "commit.json",
        "best_weights.pt", "backup.json",
    ):
        (run / name).write_text("{}")
    calls = []
    def fake_kaggle(*args, **kwargs):
        assert args[0] == "version"
        stage = Path(args[args.index("-p") + 1])
        assert (stage / "last.pt").read_bytes() == contents
        metadata = json.loads((stage / "dataset-metadata.json").read_text())
        assert metadata["id"] == "apricot88/test"
        calls.append("upload")
        return "Dataset version is being created"

    def fake_verify(dataset, hashes):
        assert dataset == "apricot88/test"
        assert set(hashes) == {
            "last.pt", "config.json", "provenance.json",
            "data.json", "preprocessing.json", "storage.json",
            "metrics.json", "summary.json", "commit.json",
            "best_weights.pt", "backup.json",
        }
        for filename, digest in hashes.items():
            assert digest == hashlib.sha256(
                (run / filename).read_bytes()
            ).hexdigest()
        calls.append("verified_all")

    monkeypatch.setattr(kaggle_backup, "kaggle", fake_kaggle)
    monkeypatch.setattr(kaggle_backup, "verify_dataset_files", fake_verify)
    manifest = kaggle_backup.publish_checkpoint(run, "apricot88/test")
    assert calls == ["upload", "verified_all"]
    assert manifest["checkpoint_sha256"] == hashlib.sha256(contents).hexdigest()


def test_download_mismatch_must_fail(tmp_path, monkeypatch):
    """Never mark an accepted upload durable when returned bytes differ."""
    def fake_kaggle(*args, **kwargs):
        if args[0] == "files":
            return "last.pt"
        if args[0] == "download":
            destination = Path(args[args.index("-p") + 1])
            (destination / "last.pt").write_bytes(b"wrong")
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(kaggle_backup, "kaggle", fake_kaggle)
    monkeypatch.setattr(kaggle_backup.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="SHA-256"):
        kaggle_backup.verify_dataset_file(
            "apricot88/test", "last.pt", hashlib.sha256(b"right").hexdigest(),
            max_attempts=2,
        )


def test_restore_kaggle_backup_files(tmp_path, monkeypatch):
    import json
    import shutil

    dataset = "apricot88/test"
    backup = tmp_path / "published"
    backup.mkdir()

    names = (
        "last.pt", "config.json", "provenance.json",
        "data.json", "preprocessing.json", "storage.json",
        "metrics.json", "summary.json", "commit.json",
        "best_weights.pt", "backup.json",
    )

    for name in names:
        (backup / name).write_bytes(("test:" + name).encode())

    manifest = {
        "dataset": dataset,
        "files_sha256": {
            name: kaggle_backup.sha256(backup / name)
            for name in names
        },
    }
    (backup / "kaggle-backup.json").write_text(
        json.dumps(manifest)
    )

    def fake_kaggle(*args, **kwargs):
        assert args[0] == "download"
        destination = Path(args[args.index("-p") + 1])
        for file in backup.iterdir():
            shutil.copyfile(file, destination / file.name)
        return ""

    # This test isolates transport and integrity checking.
    # Full PyTorch checkpoint validation is tested separately.
    monkeypatch.setattr(kaggle_backup, "kaggle", fake_kaggle)

    import training.backup
    monkeypatch.setattr(
        training.backup, "verify_backup", lambda directory: True
    )

    destination = tmp_path / "restored"
    result = kaggle_backup.restore_kaggle_backup(
        dataset, destination
    )

    assert result == destination
    for name in names:
        assert kaggle_backup.sha256(destination / name) == (
            manifest["files_sha256"][name]
        )

    with pytest.raises(FileExistsError):
        kaggle_backup.restore_kaggle_backup(
            dataset, destination
        )

    restored_manifest = json.loads(
        (destination / "kaggle-backup.json").read_text()
    )
    assert restored_manifest == manifest


def test_canonical_recovery_paths(tmp_path):
    """Accept canonical paths; reject paths from another layout."""
    import json

    output = tmp_path / "durable"
    work = tmp_path / "work"
    indices = tmp_path / "indices"

    backup = tmp_path / "downloaded"
    backup.mkdir()

    config = {
        "model": "convnext_base",
        "output": str(output),
        "work_dir": str(work),
        "indices": str(indices),
        "economical": True,
    }
    run_name = "full-test-run"

    (backup / "config.json").write_text(json.dumps(config))
    (backup / "kaggle-backup.json").write_text(
        json.dumps({"run_name": run_name})
    )

    storage_file = backup / "storage.json"

    # Wrong directory layout must be rejected.
    storage_file.write_text(json.dumps({
        "durable_directory": str(output / run_name)
    }))

    with pytest.raises(
        ValueError, match="Noncanonical checkpoint storage path"
    ):
        kaggle_backup.canonical_recovery_paths(backup)

    # Correct directory layout must be accepted.
    expected_durable = output / "convnext_base" / run_name
    expected_working = work / "convnext_base" / run_name

    storage_file.write_text(json.dumps({
        "durable_directory": str(expected_durable)
    }))

    durable, working = kaggle_backup.canonical_recovery_paths(backup)

    assert durable == expected_durable.resolve()
    assert working == expected_working.resolve()


def test_canonical_restore_refuses_overwrite(tmp_path, monkeypatch):
    """Existing run files must never be overwritten by recovery."""
    dataset = "apricot88/test"

    durable = tmp_path / "durable" / "convnext_base" / "run-1"
    working = tmp_path / "work" / "convnext_base" / "run-1"

    durable.mkdir(parents=True)
    original = durable / "last.pt"
    original.write_bytes(b"existing checkpoint")

    def fake_download(dataset_name, destination):
        assert dataset_name == dataset
        destination.mkdir()
        return destination

    def fake_paths(_directory):
        return durable, working

    monkeypatch.setattr(
        kaggle_backup, "restore_kaggle_backup", fake_download
    )
    monkeypatch.setattr(
        kaggle_backup, "canonical_recovery_paths", fake_paths
    )

    with pytest.raises(FileExistsError, match="already exists"):
        kaggle_backup.restore_canonical_run(dataset)

    assert original.read_bytes() == b"existing checkpoint"
    assert not working.exists()
