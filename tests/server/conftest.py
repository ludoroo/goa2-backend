"""Shared server-test fixtures."""

import os

import pytest

# Every on-disk location the server writes to, and the temp dir name to give it.
# Server tests build a real app via create_app(), which reads and writes these
# for real — without isolation the suite leaks fixture games into data/games and
# loads the developer's actual saves at startup (noisy, and a bad save can
# influence a test).
_ISOLATED_DIRS = {
    "GOA2_REPLAY_DIR": "replays",
    "GOA2_SHARE_DIR": "shares",
    "GOA2_SAVE_DIR": "games",
    "GOA2_BUG_REPORT_DIR": "bug_reports",
}


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path_factory):
    """Point every server data directory at a fresh temp dir, per test."""
    previous = {var: os.environ.get(var) for var in _ISOLATED_DIRS}
    for var, name in _ISOLATED_DIRS.items():
        os.environ[var] = str(tmp_path_factory.mktemp(name))
    try:
        yield
    finally:
        for var, prev in previous.items():
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
