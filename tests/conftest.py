from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

from tests.mock_configurator import create_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server:
    def __init__(self, app, port: int):
        self.app = app
        self.url = f"http://127.0.0.1:{port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self):
        self._thread.start()
        deadline = time.time() + 15
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("mock configurator did not start")
            time.sleep(0.05)

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture
def configurator(monkeypatch):
    """A fresh mock LLM Configurator on a free port, with LLM_CONFIGURATOR_URL pointing at it."""
    server = _Server(create_app(), _free_port())
    server.start()
    monkeypatch.setenv("LLM_CONFIGURATOR_URL", server.url)
    try:
        yield server
    finally:
        server.stop()
