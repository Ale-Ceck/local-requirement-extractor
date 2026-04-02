from __future__ import annotations

import shlex
import subprocess

import httpx

from config.schema import ParserConfig


class VLMServiceManager:
    def __init__(self, config: ParserConfig):
        self.config = config

    def is_healthy(self, timeout_seconds: float = 2.0) -> bool:
        if not self.config.vlm_backend.endswith("server"):
            return True

        try:
            response = httpx.get(
                self.config.vlm_server_url,
                timeout=timeout_seconds,
                follow_redirects=True,
            )
        except Exception:
            return False

        return response.status_code < 500

    def ensure_healthy(self) -> None:
        if self.is_healthy():
            return

        raise RuntimeError(
            "VLM service is not reachable at "
            f"'{self.config.vlm_server_url}'. Start it with: {self.config.vlm_server_command}"
        )

    def start_server(self) -> int:
        command = shlex.split(self.config.vlm_server_command)
        completed = subprocess.run(command, check=False)
        return completed.returncode
