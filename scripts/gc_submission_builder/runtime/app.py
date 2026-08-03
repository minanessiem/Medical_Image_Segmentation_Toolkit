"""Grand Challenge HTTP lifecycle for one initialized shared-inference runtime."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Response

from scripts.gc_submission_builder.runtime.inference import (
    GcInferenceRuntime,
    initialize_runtime,
)


LOGGER = logging.getLogger(__name__)
RUNTIME_LOGGER = logging.getLogger("scripts.gc_submission_builder.runtime")
RUNTIME_LOGGER.setLevel(logging.INFO)
if not RUNTIME_LOGGER.handlers:
    runtime_handler = logging.StreamHandler()
    runtime_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    RUNTIME_LOGGER.addHandler(runtime_handler)
RUNTIME_LOGGER.propagate = False
DEFAULT_INTERFACE_MANIFEST = Path("/opt/app/interface_manifest.yaml")
DEFAULT_RUNTIME_PROFILE = Path("/opt/app/runtime_profiles/gc_submission.yaml")


@dataclass(frozen=True)
class AppSettings:
    model_dir: Path
    interface_manifest_path: Path
    runtime_profile_path: Path
    device: str | None

    @classmethod
    def from_environment(cls) -> "AppSettings":
        return cls(
            model_dir=Path(os.environ.get("GC_MODEL_DIR", "/opt/ml/model")),
            interface_manifest_path=Path(
                os.environ.get(
                    "GC_INTERFACE_MANIFEST",
                    str(DEFAULT_INTERFACE_MANIFEST),
                )
            ),
            runtime_profile_path=Path(
                os.environ.get(
                    "GC_RUNTIME_PROFILE",
                    str(DEFAULT_RUNTIME_PROFILE),
                )
            ),
            device=os.environ.get("GC_DEVICE") or None,
        )


@dataclass
class AppState:
    runtime: GcInferenceRuntime | None = None
    initialization_error_type: str | None = None


def create_app(
    *,
    settings: AppSettings | None = None,
    initializer: Callable[..., GcInferenceRuntime] = initialize_runtime,
) -> FastAPI:
    """Create one process-local app whose readiness follows model initialization."""

    resolved_settings = settings or AppSettings.from_environment()
    state = AppState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            state.runtime = initializer(
                model_dir=resolved_settings.model_dir,
                interface_manifest_path=resolved_settings.interface_manifest_path,
                runtime_profile_path=resolved_settings.runtime_profile_path,
                device=resolved_settings.device,
            )
        except Exception as exc:
            state.initialization_error_type = type(exc).__name__
            LOGGER.error(
                "Grand Challenge runtime initialization failed error_type=%s",
                state.initialization_error_type,
            )
        yield
        state.runtime = None

    application = FastAPI(lifespan=lifespan)
    application.state.gc_state = state

    @application.get("/health")
    def health() -> Response:
        status = 200 if state.runtime is not None else 503
        return Response(content="OK" if status == 200 else "NOT_READY", status_code=status)

    @application.post("/invoke")
    def invoke() -> Response:
        if state.runtime is None:
            return Response(content="NOT_READY", status_code=503)
        try:
            state.runtime.invoke(input_root="/input", output_root="/output")
        except Exception as exc:
            LOGGER.error(
                "Grand Challenge invocation failed error_type=%s",
                type(exc).__name__,
            )
            return Response(content="INFERENCE_FAILED", status_code=500)
        return Response(status_code=201)

    return application


app = create_app()


def main() -> None:
    """Serve exactly one initialized model process on the platform port."""

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4743,
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
