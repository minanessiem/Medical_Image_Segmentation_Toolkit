"""Structured Grand Challenge observability contracts."""

from __future__ import annotations

import io
import json
import logging
import unittest
from unittest.mock import patch

import torch

from scripts.gc_submission_builder.runtime.observability import (
    EVENT_MARKER,
    emit_gc_event,
    failure_event_fields,
    normalized_timings,
    resource_summary,
    stage_error,
)


class TestGcObservability(unittest.TestCase):
    def test_event_is_one_stable_json_line(self):
        stream = io.StringIO()
        logger = logging.getLogger("tests.gc.observability")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))

        emit_gc_event(logger, "stage_completed", stage="preprocessing", seconds=1.25)

        line = stream.getvalue().strip()
        self.assertTrue(line.startswith(f"{EVENT_MARKER} "))
        payload = json.loads(line.removeprefix(f"{EVENT_MARKER} "))
        self.assertEqual(
            payload,
            {
                "event": "stage_completed",
                "seconds": 1.25,
                "stage": "preprocessing",
            },
        )
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_failure_fields_keep_stage_and_hide_uncontrolled_exception_text(self):
        try:
            raise ValueError("patient-name/secret-file.mha")
        except ValueError as cause:
            error = stage_error(
                cause,
                stage="input_canonicalization",
                error_code="INPUT_CANONICALIZATION_FAILED",
                safe_detail="The declared medical image could not be normalized.",
                timings_seconds={"input_canonicalization_seconds": 0.25},
                total_seconds=0.5,
            )

        fields = failure_event_fields(error)
        rendered = json.dumps(dict(fields), sort_keys=True)
        self.assertEqual(fields["failed_stage"], "input_canonicalization")
        self.assertEqual(fields["error_code"], "INPUT_CANONICALIZATION_FAILED")
        self.assertNotIn("patient-name", rendered)
        self.assertNotIn("secret-file", rendered)

    def test_timing_values_must_be_finite_and_nonnegative(self):
        self.assertEqual(
            normalized_timings({"stage_seconds": 0.0}),
            {"stage_seconds": 0.0},
        )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            normalized_timings({"stage_seconds": -0.1})

    def test_resource_collection_never_masks_a_primary_cuda_failure(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.max_memory_allocated",
            side_effect=RuntimeError("driver telemetry unavailable"),
        ):
            summary = resource_summary(device=torch.device("cuda"))

        self.assertIsNone(summary["cuda_peak_allocated_bytes"])
        self.assertIsNone(summary["cuda_peak_reserved_bytes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
