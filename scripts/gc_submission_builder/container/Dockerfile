FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime@sha256:f894dae26e1ee8557c544f9cfdb9dc011b1552bf3c1e656b422f2e221d380e40

ARG INTERFACE_MANIFEST=scripts/gc_submission_builder/configs/interface_manifest.fixture.yaml

LABEL org.grand-challenge.api-method="invoke"

ENV PYTHONPATH=/opt/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    GC_MODEL_DIR=/opt/ml/model \
    GC_INTERFACE_MANIFEST=/opt/app/interface_manifest.yaml \
    GC_RUNTIME_PROFILE=/opt/app/runtime_profiles/gc_submission.yaml

WORKDIR /opt/app

COPY scripts/gc_submission_builder/container/requirements.lock /opt/app/requirements.lock
RUN python -m pip install --no-cache-dir --disable-pip-version-check -r /opt/app/requirements.lock

COPY src /opt/app/src
COPY scripts/gc_submission_builder/__init__.py /opt/app/scripts/gc_submission_builder/__init__.py
COPY scripts/gc_submission_builder/release_manifest.py /opt/app/scripts/gc_submission_builder/release_manifest.py
COPY scripts/gc_submission_builder/runtime /opt/app/scripts/gc_submission_builder/runtime
COPY configs/inference_runtime/gc_submission.yaml /opt/app/runtime_profiles/gc_submission.yaml
COPY configs/inference_runtime/gc_container_test.yaml /opt/app/runtime_profiles/gc_container_test.yaml
COPY ${INTERFACE_MANIFEST} /opt/app/interface_manifest.yaml

RUN groupadd --system algorithm \
    && useradd --system --gid algorithm --no-create-home algorithm \
    && mkdir -p /input /output /opt/ml/model \
    && chown -R algorithm:algorithm /opt/app /input /output \
    && chmod -R a-w /opt/app

USER algorithm:algorithm

EXPOSE 4743

CMD ["python", "-m", "scripts.gc_submission_builder.runtime.app"]
