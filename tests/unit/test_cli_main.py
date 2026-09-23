from pathlib import Path
from unittest.mock import Mock, call

import pytest

from config.schema import AppConfig, InputConfig, LoggingConfig
from src.cli import main as cli_main
from src.cli.main import parse_args
from src.evaluation.quality_evaluator import EvaluationInputError
from src.requirement_extraction.requirement_extractor import PartialExtractionError


def test_parse_args_defaults_to_extract_command():
    args = parse_args([])

    assert args.command == "extract"
    assert args.config == Path("config.yaml")


def test_parse_args_supports_vlm_service_commands():
    check_args = parse_args(["check-vlm-service", "--config", "custom.yaml"])
    start_args = parse_args(["start-vlm-service"])
    prepare_args = parse_args(["prepare-pdf"])

    assert check_args.command == "check-vlm-service"
    assert check_args.config == Path("custom.yaml")
    assert start_args.command == "start-vlm-service"
    assert prepare_args.command == "prepare-pdf"


def test_parse_args_supports_evaluate_command_options():
    args = parse_args(
        [
            "evaluate",
            "--references-dir",
            "refs",
            "--runs-dir",
            "runs",
            "--output-dir",
            "quality",
            "--document-inventory",
            "inventory.csv",
        ]
    )

    assert args.command == "evaluate"
    assert args.references_dir == Path("refs")
    assert args.runs_dir == Path("runs")
    assert args.output_dir == Path("quality")
    assert args.document_inventory == Path("inventory.csv")


def test_evaluate_command_reports_invalid_inputs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "evaluate",
            "--references-dir",
            "refs",
            "--runs-dir",
            "runs",
            "--output-dir",
            "quality",
        ],
    )

    def fail_evaluation(**_kwargs):
        raise EvaluationInputError("duplicate requirement code REQ-1")

    monkeypatch.setattr(cli_main, "evaluate_quality", fail_evaluation)

    with pytest.raises(SystemExit, match="duplicate requirement code REQ-1"):
        cli_main.main()


@pytest.mark.parametrize("command", ["extract", "prepare-pdf"])
def test_pipeline_configures_logging_before_running(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(
        input=InputConfig(path="input"), logging=LoggingConfig(level="WARNING")
    )
    operations = Mock()
    operations.load_config.return_value = config
    monkeypatch.setattr("sys.argv", ["main.py", command, "--config", "custom.yaml"])
    monkeypatch.setattr(cli_main, "load_config", operations.load_config)
    monkeypatch.setattr(cli_main, "configure_logging", operations.configure_logging)
    monkeypatch.setattr(cli_main, "VLMServiceManager", operations.vlm_service)
    monkeypatch.setattr(cli_main, "RequirementExtractor", operations.extractor)

    cli_main.main()

    assert operations.mock_calls == [
        call.load_config(Path("custom.yaml")),
        call.configure_logging(config.logging),
        call.vlm_service(config.parser),
        call.extractor(config, command_name=command),
        call.extractor().run(),
    ]


@pytest.mark.parametrize(
    "error",
    [
        PartialExtractionError("partial output saved"),
        FileExistsError("output already exists"),
        FileNotFoundError("input does not exist"),
        ValueError("invalid pipeline input"),
    ],
)
def test_pipeline_errors_return_a_readable_nonzero_exit(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(input=InputConfig(path="input"))
    extractor = Mock()
    extractor.run.side_effect = error
    monkeypatch.setattr("sys.argv", ["main.py", "extract"])
    monkeypatch.setattr(cli_main, "load_config", Mock(return_value=config))
    monkeypatch.setattr(cli_main, "configure_logging", Mock())
    monkeypatch.setattr(cli_main, "VLMServiceManager", Mock())
    monkeypatch.setattr(cli_main, "RequirementExtractor", Mock(return_value=extractor))

    with pytest.raises(SystemExit) as caught:
        cli_main.main()

    assert caught.value.code == str(error)


def test_evaluation_does_not_require_pipeline_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "evaluate",
            "--references-dir",
            "refs",
            "--runs-dir",
            "runs",
            "--output-dir",
            "quality",
        ],
    )
    load_config = Mock(side_effect=AssertionError("evaluate must be independent"))
    configure_logging = Mock()
    evaluate_quality = Mock()
    monkeypatch.setattr(cli_main, "load_config", load_config)
    monkeypatch.setattr(cli_main, "configure_logging", configure_logging)
    monkeypatch.setattr(cli_main, "evaluate_quality", evaluate_quality)

    cli_main.main()

    load_config.assert_not_called()
    configure_logging.assert_not_called()
    evaluate_quality.assert_called_once_with(
        references_dir=Path("refs"),
        runs_dir=Path("runs"),
        output_dir=Path("quality"),
        document_inventory=None,
    )
    assert "Evaluation reports written to quality" in capsys.readouterr().out
