from pathlib import Path

from src.cli.main import parse_args


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
