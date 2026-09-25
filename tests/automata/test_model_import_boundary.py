"""The core server and observation facade must not import the optional ML stack."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_core_server_and_model_observation_facades_are_torch_free() -> None:
    script = textwrap.dedent("""
        import builtins
        import os
        import sys

        real_import = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("core import attempted to load optional torch dependency")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = reject_torch
        import automata.models
        import automata.observation
        from goa2.server.app import create_app

        from fastapi.testclient import TestClient
        app = create_app()
        with TestClient(app):
            pass
        assert "torch" not in sys.modules
        assert not any(name.startswith("torch.") for name in sys.modules)
        """)
    env = {**os.environ, "PYTHONPATH": "src"}

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_only_explicit_model_or_batching_imports_load_torch() -> None:
    script = textwrap.dedent("""
        import sys

        import automata.models

        assert "torch" not in sys.modules
        import automata.models.shared_encoder.model
        assert "torch" in sys.modules
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
