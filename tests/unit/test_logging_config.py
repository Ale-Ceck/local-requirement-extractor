import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from config.schema import LoggingConfig
from src.utils.logging_config import configure_logging, setup_logger


@pytest.fixture(autouse=True)
def restore_application_logging() -> Iterator[None]:
    """Isolate the application logger without changing pytest's root handlers."""
    logger = setup_logger().parent
    assert logger is not None
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    for handler in original_handlers:
        logger.removeHandler(handler)
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_import_and_setup_do_not_create_files(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=str(repo_root))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.cli import main; "
            "from src.utils.logging_config import setup_logger; "
            "setup_logger('test', 'custom.log').warning('before configuration')",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert not list(tmp_path.glob("*.log"))
    assert result.stdout == ""
    assert "before configuration" not in result.stderr


def test_console_logging_respects_level_and_does_not_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    config = LoggingConfig(level="WARNING", to_console=True, to_file=False)
    logger = setup_logger("console_test")
    configure_logging(config)
    configure_logging(config)

    assert setup_logger("console_test") is logger
    logger.info("hidden info")
    logger.warning("visible warning")
    output = capsys.readouterr()

    assert output.out.count("visible warning") == 1
    assert "hidden info" not in output.out
    assert output.err == ""
    assert not (tmp_path / "myapp.log").exists()


def test_file_logging_respects_configuration_and_releases_old_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "logs" / "pipeline.log"
    config = LoggingConfig(
        level="DEBUG", to_console=False, to_file=True, file_path=str(log_path)
    )
    configure_logging(config)
    logger = setup_logger("file_test")
    parent = logger.parent
    assert parent is not None
    old_file_handler = next(
        handler
        for handler in parent.handlers
        if isinstance(handler, logging.FileHandler)
    )
    configure_logging(config)
    logger.debug("requisito verificato: è valido")

    assert log_path.read_text(encoding="utf-8").count("è valido") == 1
    assert old_file_handler.stream is None
    assert capsys.readouterr().out == ""

    configure_logging(LoggingConfig(to_console=False, to_file=False))
    logger.error("disabled output")
    assert "disabled output" not in log_path.read_text(encoding="utf-8")
    assert capsys.readouterr() == ("", "")


def test_configuration_preserves_root_and_third_party_handlers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = logging.getLogger()
    third_party = logging.getLogger("third_party_logging_test")
    external_handler = logging.NullHandler()
    third_party.addHandler(external_handler)
    original_root_handlers = list(root.handlers)
    original_root_level = root.level
    original_third_party_level = third_party.level
    try:
        configure_logging(LoggingConfig(to_console=False, to_file=False))
        assert root.handlers == original_root_handlers
        assert root.level == original_root_level
        assert third_party.handlers == [external_handler]
        assert third_party.level == original_third_party_level
        third_party.warning("pytest still captures other loggers")
        assert "pytest still captures other loggers" in caplog.text
    finally:
        third_party.removeHandler(external_handler)
        external_handler.close()


def test_invalid_logging_level_does_not_replace_existing_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(LoggingConfig(level="INFO"))
    with pytest.raises(ValueError, match="logging.level"):
        configure_logging(LoggingConfig(level="no-such-level"))

    setup_logger("validation_test").info("previous configuration active")
    assert "previous configuration active" in capsys.readouterr().out
