"""Bounded cleanup for stale, incomplete local generation jobs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections import OrderedDict
import os
import pathlib
import re
import stat
import threading
import time

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms use msvcrt.
    fcntl = None


_JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SCAN_LOCK = threading.Lock()
_SCAN_CURSORS: OrderedDict[
    pathlib.Path, tuple[tuple[int, int], os.ScandirIterator]
] = OrderedDict()
_MAX_SCAN_CURSORS = 8


@contextmanager
def _bounded_scandir(root: pathlib.Path, max_entries: int):
    """Keep a bounded scan cursor so repeated calls eventually visit all jobs."""
    _SCAN_LOCK.acquire()
    try:
        root_stat = root.stat()
        identity = (root_stat.st_dev, root_stat.st_ino)
        existing = _SCAN_CURSORS.get(root)
        if existing is not None and existing[0] != identity:
            existing[1].close()
            del _SCAN_CURSORS[root]
            existing = None
        if existing is None:
            iterator = os.scandir(root)
            _SCAN_CURSORS[root] = (identity, iterator)
            while len(_SCAN_CURSORS) > _MAX_SCAN_CURSORS:
                _old_root, (_old_identity, old_iterator) = _SCAN_CURSORS.popitem(
                    last=False
                )
                old_iterator.close()
        else:
            _SCAN_CURSORS.move_to_end(root)
            iterator = existing[1]

        def next_entries():
            for _ in range(max_entries):
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator.close()
                    _SCAN_CURSORS.pop(root, None)
                    return

        yield next_entries()
    finally:
        _SCAN_LOCK.release()


def _acquire_lock(descriptor: int, *, blocking: bool) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(
            descriptor, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1
        )
        return
    if fcntl is None:
        raise OSError("No supported advisory lock implementation")
    fcntl.flock(
        descriptor,
        fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB,
    )


def _release_lock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    if fcntl is None:
        raise OSError("No supported advisory lock implementation")
    fcntl.flock(descriptor, fcntl.LOCK_UN)


@dataclass(frozen=True)
class CleanupResult:
    scanned_entries: int = 0
    removed_jobs: int = 0
    skipped_active: int = 0
    skipped_oversized: int = 0
    errors: int = 0


class _DeletionBudgetExceeded(Exception):
    pass


class _ConcurrentTreeMutation(Exception):
    pass


def _tree_entry_manifest(
    job_root: pathlib.Path, max_entries: int
) -> dict[str, tuple[int, int, int]] | None:
    """Capture bounded entry identities; return None if the tree exceeds budget."""
    resolved_root = job_root.resolve(strict=True)
    pending: list[tuple[pathlib.Path, pathlib.Path]] = [(resolved_root, pathlib.Path())]
    manifest: dict[str, tuple[int, int, int]] = {}
    while pending:
        directory, relative_directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                if len(manifest) >= max_entries:
                    return None
                child_stat = child.stat(follow_symlinks=False)
                relative = relative_directory / child.name
                manifest[relative.as_posix()] = (
                    child_stat.st_dev,
                    child_stat.st_ino,
                    stat.S_IFMT(child_stat.st_mode),
                )
                if stat.S_ISDIR(child_stat.st_mode):
                    resolved_child = pathlib.Path(child.path).resolve(strict=True)
                    if resolved_child.parent != directory:
                        raise OSError("Job entry changed during cleanup scan")
                    pending.append((resolved_child, relative))
    return manifest


def _remove_tree_bounded(
    root_descriptor: int,
    job_id: str,
    expected_stat: os.stat_result,
    expected_entries: dict[str, tuple[int, int, int]],
    max_entries: int,
) -> int:
    """Delete by directory descriptor, stopping as soon as the work cap is hit.

    Symlinks are unlinked as entries and never opened. Every descendant entry is
    counted before inspection. One additional directory entry may be observed
    to establish that the cap was exceeded; no work is performed on that entry.
    """
    required_calls = (os.open, os.stat, os.unlink, os.rmdir)
    if not all(call in os.supports_dir_fd for call in required_calls):
        raise OSError("Descriptor-relative deletion is unavailable")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    job_descriptor = os.open(job_id, flags, dir_fd=root_descriptor)
    inspected = 0
    observed: set[str] = set()
    try:
        opened_stat = os.fstat(job_descriptor)
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino)
            != (expected_stat.st_dev, expected_stat.st_ino)
        ):
            raise OSError("Job directory changed before cleanup")

        def remove_contents(directory_descriptor: int, relative_directory: str = "") -> None:
            nonlocal inspected
            with os.scandir(directory_descriptor) as children:
                for child in children:
                    if inspected >= max_entries:
                        raise _DeletionBudgetExceeded
                    inspected += 1
                    child_stat = os.stat(
                        child.name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    relative_name = (
                        f"{relative_directory}/{child.name}"
                        if relative_directory
                        else child.name
                    )
                    identity = (
                        child_stat.st_dev,
                        child_stat.st_ino,
                        stat.S_IFMT(child_stat.st_mode),
                    )
                    if expected_entries.get(relative_name) != identity:
                        raise _ConcurrentTreeMutation
                    observed.add(relative_name)
                    if stat.S_ISDIR(child_stat.st_mode):
                        child_descriptor = os.open(
                            child.name,
                            flags,
                            dir_fd=directory_descriptor,
                        )
                        try:
                            opened_child_stat = os.fstat(child_descriptor)
                            if (opened_child_stat.st_dev, opened_child_stat.st_ino) != (
                                child_stat.st_dev,
                                child_stat.st_ino,
                            ):
                                raise OSError("Job entry changed during cleanup")
                            remove_contents(child_descriptor, relative_name)
                        finally:
                            os.close(child_descriptor)

                        current_child_stat = os.stat(
                            child.name,
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                        if (current_child_stat.st_dev, current_child_stat.st_ino) != (
                            opened_child_stat.st_dev,
                            opened_child_stat.st_ino,
                        ):
                            raise OSError("Job entry changed during cleanup")
                        os.rmdir(child.name, dir_fd=directory_descriptor)
                    else:
                        os.unlink(child.name, dir_fd=directory_descriptor)

        remove_contents(job_descriptor)
        if observed != expected_entries.keys():
            raise _ConcurrentTreeMutation
    finally:
        os.close(job_descriptor)

    current_job_stat = os.stat(job_id, dir_fd=root_descriptor, follow_symlinks=False)
    if (current_job_stat.st_dev, current_job_stat.st_ino) != (
        opened_stat.st_dev,
        opened_stat.st_ino,
    ):
        raise _ConcurrentTreeMutation
    os.rmdir(job_id, dir_fd=root_descriptor)
    return inspected


@contextmanager
def active_job(projects_root: pathlib.Path, job_id: str):
    """Hold a cross-process advisory lock for the full lifetime of a job."""
    if not _JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job identifier")
    configured_root = pathlib.Path(projects_root)
    if configured_root.is_symlink():
        raise OSError("Unsafe generated-project root")
    root = configured_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    active_root = root / ".active"
    if active_root.is_symlink():
        raise OSError("Unsafe active-job directory")
    active_root.mkdir(mode=0o700, exist_ok=True)
    if active_root.is_symlink() or active_root.resolve().parent != root:
        raise OSError("Unsafe active-job directory")
    lock_path = active_root / f"{job_id}.lock"
    if lock_path.is_symlink():
        raise OSError("Unsafe active-job lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Invalid active-job lock")
        _acquire_lock(descriptor, blocking=True)
        try:
            yield
        finally:
            job_directory = root / job_id
            try:
                job_stat = job_directory.lstat()
                if stat.S_ISDIR(job_stat.st_mode):
                    os.utime(job_directory, None, follow_symlinks=False)
            except OSError:
                pass
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            _release_lock(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _try_job_lock(projects_root: pathlib.Path, job_id: str):
    active_root = projects_root / ".active"
    try:
        if active_root.is_symlink():
            yield False, None
            return
        active_root.mkdir(mode=0o700, exist_ok=True)
        if active_root.resolve().parent != projects_root:
            yield False, None
            return
        lock_path = active_root / f"{job_id}.lock"
        if lock_path.is_symlink():
            yield False, None
            return
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError:
        yield False, None
        return

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            yield False, None
            return
        _acquire_lock(descriptor, blocking=False)
    except (BlockingIOError, OSError):
        os.close(descriptor)
        yield False, None
        return

    try:
        yield True, lock_path
    finally:
        _release_lock(descriptor)
        os.close(descriptor)


def _has_completion_marker(root: pathlib.Path, job_id: str) -> bool | None:
    marker_root = root / ".completed"
    try:
        marker_root_stat = marker_root.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    if not stat.S_ISDIR(marker_root_stat.st_mode):
        return None
    marker = marker_root / f"{job_id}.done"
    try:
        marker_stat = marker.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    # Even a malformed marker is treated conservatively as completed so cleanup
    # never removes a job whose completion state cannot be safely determined.
    return True


def _tree_within_entry_limit(job_root: pathlib.Path, max_entries: int) -> bool | None:
    """Inspect at most max_entries + 1 descendants without following symlinks."""
    try:
        resolved_root = job_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    pending = [resolved_root]
    inspected = 0
    while pending:
        directory = pending.pop()
        try:
            resolved_directory = directory.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            return None
        if resolved_directory != resolved_root and resolved_root not in resolved_directory.parents:
            return False
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    inspected += 1
                    if inspected > max_entries:
                        return False
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            child = pathlib.Path(entry.path)
                            resolved_child = child.resolve(strict=True)
                            if (
                                resolved_child.parent != resolved_directory
                                or resolved_root not in resolved_child.parents
                            ):
                                return False
                            pending.append(resolved_child)
                    except (FileNotFoundError, OSError, RuntimeError):
                        return None
        except (FileNotFoundError, OSError):
            return None
    return True


def cleanup_orphan_jobs(
    projects_root: pathlib.Path,
    *,
    now: float | None = None,
    retention_seconds: int,
    max_jobs: int,
    max_entries: int,
    max_tree_entries: int = 1_000,
) -> CleanupResult:
    """Remove only old, unmarked job directories while holding their job lock.

    Completed jobs are retained indefinitely. Directory age is measured from its
    last root-directory modification; the active-job context refreshes that time
    when generation ends, preserving partial failures for the full retention
    window. The scan and number of deletions are both bounded by caller limits.
    """
    root_input = pathlib.Path(projects_root)
    try:
        root_input.mkdir(parents=True, exist_ok=True)
        root = root_input.resolve(strict=True)
        if not root.is_dir() or root_input.is_symlink():
            return CleanupResult(errors=1)
    except OSError:
        return CleanupResult(errors=1)

    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(root, root_flags)
        opened_root_stat = os.fstat(root_descriptor)
        current_root_stat = root.stat()
        if (opened_root_stat.st_dev, opened_root_stat.st_ino) != (
            current_root_stat.st_dev,
            current_root_stat.st_ino,
        ):
            os.close(root_descriptor)
            return CleanupResult(errors=1)
    except OSError:
        return CleanupResult(errors=1)

    current_time = time.time() if now is None else now
    threshold = current_time - retention_seconds
    scanned = removed = skipped_active = skipped_oversized = errors = 0
    try:
        with _bounded_scandir(root, max_entries) as entries:
            while scanned < max_entries and removed < max_jobs:
                entry = next(entries, None)
                if entry is None:
                    break
                scanned += 1
                if not _JOB_ID_PATTERN.fullmatch(entry.name):
                    continue
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                    if (
                        not stat.S_ISDIR(entry_stat.st_mode)
                        or entry_stat.st_mtime > threshold
                    ):
                        continue
                    candidate = root / entry.name
                    resolved_candidate = candidate.resolve(strict=True)
                    if resolved_candidate.parent != root:
                        continue
                except (FileNotFoundError, OSError, RuntimeError):
                    errors += 1
                    continue

                try:
                    with _try_job_lock(root, entry.name) as (locked, lock_path):
                        if not locked:
                            skipped_active += 1
                            continue
                        completed = _has_completion_marker(root, entry.name)
                        if completed is None or completed:
                            continue
                        try:
                            current_stat = candidate.lstat()
                            if (
                                not stat.S_ISDIR(current_stat.st_mode)
                                or current_stat.st_mtime > threshold
                                or candidate.resolve(strict=True).parent != root
                            ):
                                continue
                            tree_is_bounded = _tree_within_entry_limit(
                                candidate, max_tree_entries
                            )
                            if tree_is_bounded is None:
                                errors += 1
                                continue
                            if not tree_is_bounded:
                                skipped_oversized += 1
                                continue
                            try:
                                entry_manifest = _tree_entry_manifest(
                                    candidate, max_tree_entries
                                )
                            except (FileNotFoundError, OSError, RuntimeError):
                                errors += 1
                                continue
                            if entry_manifest is None:
                                skipped_oversized += 1
                                continue
                            try:
                                _remove_tree_bounded(
                                    root_descriptor,
                                    entry.name,
                                    current_stat,
                                    entry_manifest,
                                    max_tree_entries,
                                )
                            except _DeletionBudgetExceeded:
                                skipped_oversized += 1
                                continue
                            except _ConcurrentTreeMutation:
                                skipped_oversized += 1
                                continue
                            removed += 1
                            if lock_path is not None:
                                try:
                                    lock_path.unlink(missing_ok=True)
                                except OSError:
                                    errors += 1
                        except FileNotFoundError:
                            continue
                        except (OSError, RuntimeError):
                            errors += 1
                except OSError:
                    errors += 1
    except OSError:
        errors += 1
    finally:
        os.close(root_descriptor)

    return CleanupResult(
        scanned_entries=scanned,
        removed_jobs=removed,
        skipped_active=skipped_active,
        skipped_oversized=skipped_oversized,
        errors=errors,
    )
