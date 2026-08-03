"""Grand Challenge HTTP readiness and invocation contracts."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from scripts.gc_submission_builder.runtime.app import AppSettings, create_app
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise
    AppSettings = None
    create_app = None


SETTINGS = (
    AppSettings(
        model_dir=Path("/fixture/model"),
        interface_manifest_path=Path("/fixture/interface.yaml"),
        runtime_profile_path=Path("/fixture/runtime.yaml"),
        device="cpu",
    )
    if AppSettings is not None
    else None
)


async def _request(app, method: str, path: str) -> tuple[int, bytes, list[tuple[bytes, bytes]]]:
    sent: list[dict] = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("algorithm", 4743),
        "state": {},
    }
    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return int(start["status"]), body, list(start.get("headers", []))


def _exercise(app, requests):
    async def run():
        async with app.router.lifespan_context(app):
            return [await _request(app, method, path) for method, path in requests]

    return asyncio.run(run())


@unittest.skipIf(create_app is None, "FastAPI is installed in the GC image, not the host test venv")
class TestGcApp(unittest.TestCase):
    def test_ready_runtime_exposes_health_200_and_invoke_201(self):
        calls = []
        runtime = SimpleNamespace(invoke=lambda **kwargs: calls.append(kwargs))
        app = create_app(settings=SETTINGS, initializer=lambda **_kwargs: runtime)

        health, invoke = _exercise(app, [("GET", "/health"), ("POST", "/invoke")])

        self.assertEqual(health[:2], (200, b"OK"))
        self.assertEqual(invoke[:2], (201, b""))
        self.assertEqual(calls, [{"input_root": "/input", "output_root": "/output"}])

    def test_initialization_failure_keeps_health_and_invoke_not_ready(self):
        def fail(**_kwargs):
            raise RuntimeError("protected/model/path")

        app = create_app(settings=SETTINGS, initializer=fail)

        health, invoke = _exercise(app, [("GET", "/health"), ("POST", "/invoke")])

        self.assertEqual(health[:2], (503, b"NOT_READY"))
        self.assertEqual(invoke[:2], (503, b"NOT_READY"))
        self.assertEqual(
            app.state.gc_state.initialization_error_type,
            "RuntimeError",
        )

    def test_runtime_failure_returns_generic_http_500(self):
        def fail_invoke(**_kwargs):
            raise RuntimeError("protected/input/file.nii.gz")

        runtime = SimpleNamespace(invoke=fail_invoke)
        app = create_app(settings=SETTINGS, initializer=lambda **_kwargs: runtime)

        (response,) = _exercise(app, [("POST", "/invoke")])

        self.assertEqual(response[:2], (500, b"INFERENCE_FAILED"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
