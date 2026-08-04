"""Grand Challenge HTTP readiness and invocation contracts."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from scripts.gc_submission_builder.runtime.app import (
        LOGGER as APP_LOGGER,
        AppSettings,
        create_app,
    )
    from scripts.gc_submission_builder.runtime.inference import GcRuntimeStageError
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise
    AppSettings = None
    APP_LOGGER = None
    create_app = None
    GcRuntimeStageError = None


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
    def test_module_execution_uses_the_configured_runtime_logger_namespace(self):
        self.assertEqual(
            APP_LOGGER.name,
            "scripts.gc_submission_builder.runtime.app",
        )

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
        with self.assertLogs(
            "scripts.gc_submission_builder.runtime.app", level="INFO"
        ) as captured:
            health, invoke = _exercise(
                app, [("GET", "/health"), ("POST", "/invoke")]
            )

        self.assertEqual(health[:2], (503, b"NOT_READY"))
        self.assertEqual(invoke[:2], (503, b"NOT_READY"))
        self.assertEqual(
            app.state.gc_state.initialization_error_type,
            "RuntimeError",
        )
        rendered = "\n".join(captured.output)
        self.assertIn("GC_EVENT", rendered)
        self.assertIn('"event":"startup_failed"', rendered)
        self.assertNotIn("protected/model/path", rendered)

    def test_runtime_failure_returns_generic_http_500(self):
        def fail_invoke(**_kwargs):
            raise RuntimeError("protected/input/file.nii.gz")

        runtime = SimpleNamespace(invoke=fail_invoke, device=None)
        app = create_app(settings=SETTINGS, initializer=lambda **_kwargs: runtime)
        with self.assertLogs(
            "scripts.gc_submission_builder.runtime.app", level="INFO"
        ) as captured:
            (response,) = _exercise(app, [("POST", "/invoke")])

        self.assertEqual(response[:2], (500, b"INFERENCE_FAILED"))
        rendered = "\n".join(captured.output)
        self.assertIn('"event":"invocation_failed"', rendered)
        self.assertNotIn("protected/input/file.nii.gz", rendered)

    def test_stage_failure_emits_partial_timing_and_stable_code(self):
        cause = ValueError("protected/input/patient-file.mha")

        def fail_invoke(**_kwargs):
            raise GcRuntimeStageError(
                stage="input_canonicalization",
                error_code="INPUT_CANONICALIZATION_FAILED",
                safe_detail="Grand Challenge input canonicalization did not complete.",
                timings_seconds={"input_canonicalization_seconds": 0.25},
                total_seconds=0.5,
                cause=cause,
            ) from cause

        runtime = SimpleNamespace(invoke=fail_invoke, device=None)
        app = create_app(settings=SETTINGS, initializer=lambda **_kwargs: runtime)
        with self.assertLogs(
            "scripts.gc_submission_builder.runtime.app", level="INFO"
        ) as captured:
            (response,) = _exercise(app, [("POST", "/invoke")])

        self.assertEqual(response[:2], (500, b"INFERENCE_FAILED"))
        rendered = "\n".join(captured.output)
        self.assertIn('"failed_stage":"input_canonicalization"', rendered)
        self.assertIn('"error_code":"INPUT_CANONICALIZATION_FAILED"', rendered)
        self.assertIn('"input_canonicalization_seconds":0.25', rendered)
        self.assertNotIn("patient-file", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
