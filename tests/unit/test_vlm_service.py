import pytest

from config.schema import ParserConfig
from src.vlm_service import VLMServiceManager


class DummyResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_is_healthy_returns_true_for_success_status(monkeypatch):
    manager = VLMServiceManager(
        ParserConfig(vlm_backend="mlx-vlm-server", vlm_server_url="http://localhost:8111/")
    )

    monkeypatch.setattr(
        "src.vlm_service.httpx.get",
        lambda url, timeout, follow_redirects: DummyResponse(200),
    )

    assert manager.is_healthy() is True


def test_is_healthy_returns_false_on_connection_error(monkeypatch):
    manager = VLMServiceManager(
        ParserConfig(vlm_backend="mlx-vlm-server", vlm_server_url="http://localhost:8111/")
    )

    def raise_error(url, timeout, follow_redirects):
        raise RuntimeError("connection failed")

    monkeypatch.setattr("src.vlm_service.httpx.get", raise_error)

    assert manager.is_healthy() is False


def test_ensure_healthy_raises_clear_error_for_unreachable_service(monkeypatch):
    manager = VLMServiceManager(
        ParserConfig(
            vlm_backend="mlx-vlm-server",
            vlm_server_url="http://localhost:8111/",
            vlm_server_command="mlx_vlm.server --port 8111",
        )
    )

    monkeypatch.setattr(manager, "is_healthy", lambda: False)

    with pytest.raises(RuntimeError, match="mlx_vlm.server --port 8111"):
        manager.ensure_healthy()
