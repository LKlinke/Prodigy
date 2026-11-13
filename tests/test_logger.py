"""The logger writes no file into the working directory (#284); DEBUG goes to a file only on request."""
import logging
import os
import subprocess
import sys

from prodigy.util.logger import LOG_FILE_ENV, log_setup


def test_log_setup_creates_no_file_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(LOG_FILE_ENV, raising=False)
    logger = log_setup("test_logger_default", logging.DEBUG)
    logger.debug("whole generating function")
    assert list(tmp_path.iterdir()) == []
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_debug_file_on_request(tmp_path, monkeypatch):
    target = tmp_path / "debug.log"
    monkeypatch.setenv(LOG_FILE_ENV, str(target))
    logger = log_setup("test_logger_requested", logging.DEBUG)
    logger.debug("hello")
    assert "hello" in target.read_text()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_analysis_run_creates_no_file(tmp_path):
    program = tmp_path / "p.ppy"
    program.write_text("x: nat;\nx = poisson(2);\nquery Ex[x];\n")
    env = {k: v for k, v in os.environ.items() if k != LOG_FILE_ENV}
    result = subprocess.run([sys.executable, "-m", "prodigy.cli", "analyze", str(program)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=100, check=False)
    assert result.returncode == 0, result.stderr
    assert [p.name for p in tmp_path.iterdir()] == ["p.ppy"]
