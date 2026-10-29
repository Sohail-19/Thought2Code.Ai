import os
import time
from uuid import uuid4

import pytest

import api.job_cleanup as cleanup_module
from api.job_cleanup import active_job, cleanup_orphan_jobs


def make_job(root, *, modified_at):
    job_id = uuid4().hex
    directory = root / job_id
    directory.mkdir(parents=True)
    (directory / "partial.txt").write_text("diagnostic output")
    os.utime(directory, (modified_at, modified_at))
    return job_id, directory


def cleanup(root, **overrides):
    arguments = {
        "now": time.time(),
        "retention_seconds": 60,
        "max_jobs": 10,
        "max_entries": 100,
    }
    arguments.update(overrides)
    return cleanup_orphan_jobs(root, **arguments)


def test_cleanup_removes_stale_partial_job_and_is_idempotent(tmp_path):
    root = tmp_path / "jobs"
    job_id, directory = make_job(root, modified_at=time.time() - 600)

    first = cleanup(root)
    second = cleanup(root)

    assert first.removed_jobs == 1
    assert not directory.exists()
    assert second.removed_jobs == 0
    assert not (root / ".active" / f"{job_id}.lock").exists()


def test_cleanup_skips_active_job_even_when_its_directory_is_old(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    job_id = uuid4().hex
    with active_job(root, job_id):
        directory = root / job_id
        directory.mkdir()
        os.utime(directory, (time.time() - 600, time.time() - 600))
        result = cleanup(root)
        assert result.removed_jobs == 0
        assert result.skipped_active == 1
        assert directory.exists()

    # Ending a failed job refreshes its retention clock.
    assert cleanup(root).removed_jobs == 0
    os.utime(directory, (time.time() - 600, time.time() - 600))
    assert cleanup(root).removed_jobs == 1


def test_cleanup_preserves_completed_and_recent_jobs(tmp_path):
    root = tmp_path / "jobs"
    old_completed_id, old_completed = make_job(
        root, modified_at=time.time() - 600
    )
    recent_id, recent = make_job(root, modified_at=time.time())
    marker_root = root / ".completed"
    marker_root.mkdir()
    (marker_root / f"{old_completed_id}.done").touch()

    result = cleanup(root)

    assert result.removed_jobs == 0
    assert old_completed.exists()
    assert recent.exists()
    assert not (root / ".active" / f"{recent_id}.lock").exists()


def test_cleanup_does_not_follow_job_symlinks_outside_root(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    important = outside / "keep.txt"
    important.write_text("preserve")
    link = root / uuid4().hex
    link.symlink_to(outside, target_is_directory=True)
    os.utime(link, (time.time() - 600, time.time() - 600), follow_symlinks=False)

    result = cleanup(root)

    assert result.removed_jobs == 0
    assert link.is_symlink()
    assert important.read_text() == "preserve"


def test_cleanup_unlinks_symlink_entries_without_following_them(tmp_path):
    root = tmp_path / "jobs"
    job_id, directory = make_job(root, modified_at=time.time() - 600)
    outside = tmp_path / "outside"
    outside.mkdir()
    important = outside / "keep.txt"
    important.write_text("preserve")
    (directory / "outside-link").symlink_to(outside, target_is_directory=True)
    stale_time = time.time() - 600
    os.utime(directory, (stale_time, stale_time))

    result = cleanup(root)

    assert result.removed_jobs == 1
    assert not directory.exists()
    assert important.read_text() == "preserve"
    assert not (root / ".active" / f"{job_id}.lock").exists()


def test_cleanup_skips_job_when_lock_path_is_a_symlink(tmp_path):
    root = tmp_path / "jobs"
    job_id, directory = make_job(root, modified_at=time.time() - 600)
    outside = tmp_path / "outside-lock"
    outside.write_text("preserve")
    active_root = root / ".active"
    active_root.mkdir()
    (active_root / f"{job_id}.lock").symlink_to(outside)

    result = cleanup(root)

    assert result.removed_jobs == 0
    assert result.skipped_active == 1
    assert directory.exists()
    assert outside.read_text() == "preserve"


def test_cleanup_is_bounded_and_can_continue_on_later_invocations(tmp_path):
    root = tmp_path / "jobs"
    stale = [
        make_job(root, modified_at=time.time() - 600)[1]
        for _ in range(3)
    ]

    first = cleanup(root, max_jobs=1, max_entries=100)
    second = cleanup(root, max_jobs=1, max_entries=100)
    third = cleanup(root, max_jobs=1, max_entries=100)

    assert first.removed_jobs == second.removed_jobs == third.removed_jobs == 1
    assert sum(path.exists() for path in stale) == 0


def test_cleanup_bounds_directory_entries_scanned(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    for index in range(20):
        (root / f"unrelated-{index}").mkdir()

    result = cleanup(root, max_entries=3)

    assert result.scanned_entries == 3
    assert result.removed_jobs <= 10


def test_repeated_bounded_scans_reach_later_job_directories(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    for index in range(230):
        (root / f"unrelated-{index}").mkdir()
    job_id, stale = make_job(root, modified_at=time.time() - 600)

    results = [cleanup(root, max_entries=100) for _ in range(4)]

    assert sum(result.scanned_entries for result in results) <= 400
    assert any(result.removed_jobs for result in results)
    assert not stale.exists()
    assert not (root / ".active" / f"{job_id}.lock").exists()


def test_cleanup_handles_filesystem_errors_without_deleting_job(monkeypatch, tmp_path):
    root = tmp_path / "jobs"
    _job_id, directory = make_job(root, modified_at=time.time() - 600)

    def fail_remove(*_args):
        raise PermissionError()

    monkeypatch.setattr(cleanup_module, "_remove_tree_bounded", fail_remove)

    result = cleanup(root)

    assert result.removed_jobs == 0
    assert result.errors == 1
    assert directory.exists()


def test_cleanup_enforces_budget_during_deletion_after_concurrent_growth(
    monkeypatch, tmp_path
):
    root = tmp_path / "jobs"
    _job_id, directory = make_job(root, modified_at=time.time() - 600)
    original_mtime = directory.stat().st_mtime
    original_scan = cleanup_module._tree_within_entry_limit

    def grow_after_preflight(job_root, max_entries):
        assert original_scan(job_root, max_entries) is True
        for index in range(6):
            (directory / f"concurrent-{index}.txt").write_text("added after scan")
        os.utime(directory, (original_mtime, original_mtime))
        return True

    monkeypatch.setattr(cleanup_module, "_tree_within_entry_limit", grow_after_preflight)

    result = cleanup(root, max_tree_entries=2)

    assert result.removed_jobs == 0
    assert result.skipped_oversized == 1
    assert directory.is_dir()
    assert len(list(directory.iterdir())) >= 5


def test_cleanup_retains_job_when_tree_changes_after_manifest(monkeypatch, tmp_path):
    root = tmp_path / "jobs"
    _job_id, directory = make_job(root, modified_at=time.time() - 600)
    original_remove = cleanup_module._remove_tree_bounded

    def add_entry_before_delete(root_descriptor, job_id, expected_stat, manifest, limit):
        (directory / "late-entry.txt").write_text("concurrent addition")
        return original_remove(
            root_descriptor, job_id, expected_stat, manifest, limit
        )

    monkeypatch.setattr(cleanup_module, "_remove_tree_bounded", add_entry_before_delete)

    result = cleanup(root)

    assert result.removed_jobs == 0
    assert result.skipped_oversized == 1
    assert directory.is_dir()
    assert (directory / "late-entry.txt").read_text() == "concurrent addition"


def test_cleanup_skips_job_with_tree_larger_than_deletion_budget(tmp_path):
    root = tmp_path / "jobs"
    _job_id, directory = make_job(root, modified_at=time.time() - 600)
    (directory / "another.txt").write_text("bounded cleanup")
    os.utime(directory, (time.time() - 600, time.time() - 600))

    result = cleanup(root, max_tree_entries=1)

    assert result.removed_jobs == 0
    assert result.skipped_oversized == 1
    assert directory.exists()


def test_cleanup_rejects_symlinked_configured_root(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "jobs"
    linked_root.symlink_to(real_root, target_is_directory=True)

    result = cleanup(linked_root)

    assert result.errors == 1
    assert list(real_root.iterdir()) == []
