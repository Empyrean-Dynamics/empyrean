"""A broken data directory is named, not reported as ``[Errno 17]``.

``Path.mkdir(exist_ok=True)`` re-raises ``FileExistsError`` whenever the
path exists but is not a directory, and ``mkdir(2)`` never follows a
trailing symbolic link — so a data dir that is a link to nowhere raises a
bare ``[Errno 17] File exists`` that names neither the link nor its
target. These pin the diagnosis on :func:`empyrean._ensure_data_dir`
directly: no kernels, no network, no global context.
"""

import os
import sys

import pytest
from empyrean import _ensure_data_dir


def test_a_real_directory_is_created_and_reused(tmp_path):
    target = tmp_path / "nested" / "data"
    _ensure_data_dir(target)
    assert target.is_dir()
    # Idempotent — a second call over an existing directory is a no-op.
    _ensure_data_dir(target)
    assert target.is_dir()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_a_symlink_to_a_real_directory_resolves(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "data"
    link.symlink_to(real)
    # Must not raise: the ordinary "data dir lives behind a link" layout.
    _ensure_data_dir(link)
    assert link.is_dir()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_a_dangling_symlink_names_the_link_and_its_target(tmp_path):
    link = tmp_path / "data"
    link.symlink_to(tmp_path / "nowhere")

    with pytest.raises(NotADirectoryError) as excinfo:
        _ensure_data_dir(link)

    message = str(excinfo.value)
    assert "symbolic link" in message
    assert str(link) in message
    assert os.readlink(link) in message
    assert "Errno 17" not in message
    assert "File exists" not in message


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_a_self_referential_symlink_is_named(tmp_path):
    # The trap the engine's habit-symlink writer can leave behind: a link
    # whose target is itself. Resolution fails with ELOOP, not ENOENT, so
    # this exercises a different errno through the same guard.
    link = tmp_path / "data"
    link.symlink_to(link)

    with pytest.raises(NotADirectoryError) as excinfo:
        _ensure_data_dir(link)

    message = str(excinfo.value)
    assert "symbolic link" in message
    assert str(link) in message
    assert "Errno 17" not in message


def test_a_regular_file_at_the_data_dir_path_is_named(tmp_path):
    path = tmp_path / "data"
    path.write_text("not a directory")

    with pytest.raises(NotADirectoryError) as excinfo:
        _ensure_data_dir(path)

    message = str(excinfo.value)
    assert "is not a directory" in message
    assert str(path) in message
    assert "Errno 17" not in message
