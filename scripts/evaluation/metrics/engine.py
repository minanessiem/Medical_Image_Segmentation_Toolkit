"""
Streaming metrics engine for greenfield segmentation evaluation.

This module intentionally avoids storing per-sample tensors in memory.
It updates running statistics incrementally for each incoming sample.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from scripts.evaluation.core.contracts import RunningStats, ScopeName, ScopedRunningStats, SliceSample, VolumeSample
from scripts.evaluation.metrics.registry_2d import (
    TWOD_METRIC_CLASSES,
    compute_metrics_at_threshold,
    resolve_2d_metric_class_names,
)
from scripts.evaluation.metrics.registry_3d import THREED_METRIC_CLASSES, compute_metrics_3d_at_threshold
from scripts.evaluation.io.volume_assembler import VolumeAssembler


DEFAULT_METRIC_NAMES: Sequence[str] = (
    "Dice2DForegroundOnly",
    "VoxelPrecision2D",
    "VoxelSensitivity2D",
    "VoxelSpecificity2D",
    "VoxelF1Score2D",
    "VoxelF2Score2D",
)


@dataclass
class ThresholdState:
    """Accumulator state for one threshold."""

    threshold: float
    metrics: Dict[str, ScopedRunningStats] = field(default_factory=dict)
    slice_counts: Dict[str, int] = field(
        default_factory=lambda: {"total": 0, "foreground": 0, "empty": 0}
    )

    def to_dict(self) -> Dict[str, object]:
        """Serialize threshold state to report-ready dictionary."""
        return {
            "threshold": float(self.threshold),
            "slice_counts": {
                "total": int(self.slice_counts["total"]),
                "foreground": int(self.slice_counts["foreground"]),
                "empty": int(self.slice_counts["empty"]),
            },
            "metrics": {name: scoped.to_dict() for name, scoped in self.metrics.items()},
        }


@dataclass
class VolumeThresholdState:
    """Accumulator state for one threshold at volume level."""

    threshold: float
    metrics: Dict[str, RunningStats] = field(default_factory=dict)
    volume_counts: Dict[str, int] = field(
        default_factory=lambda: {"total": 0, "foreground": 0, "empty": 0}
    )
    volume_slice_count_stats: RunningStats = field(default_factory=RunningStats)
    volume_slice_count_total: int = 0
    volume_slice_count_min: Optional[int] = None
    volume_slice_count_max: Optional[int] = None

    def update_volume_slice_count(self, num_slices: int) -> None:
        self.volume_slice_count_stats.update(float(num_slices))
        self.volume_slice_count_total += int(num_slices)
        if self.volume_slice_count_min is None or num_slices < self.volume_slice_count_min:
            self.volume_slice_count_min = int(num_slices)
        if self.volume_slice_count_max is None or num_slices > self.volume_slice_count_max:
            self.volume_slice_count_max = int(num_slices)

    def to_dict(self) -> Dict[str, object]:
        return {
            "threshold": float(self.threshold),
            "volume_counts": {
                "total": int(self.volume_counts["total"]),
                "foreground": int(self.volume_counts["foreground"]),
                "empty": int(self.volume_counts["empty"]),
            },
            "volume_slice_counts": {
                "mean": float(self.volume_slice_count_stats.mean),
                "std": float(self.volume_slice_count_stats.std),
                "count": int(self.volume_slice_count_stats.count),
                "total": int(self.volume_slice_count_total),
                "min": int(self.volume_slice_count_min) if self.volume_slice_count_min is not None else 0,
                "max": int(self.volume_slice_count_max) if self.volume_slice_count_max is not None else 0,
            },
            "metrics": {name: stats.to_dict() for name, stats in self.metrics.items()},
        }


class StreamingMetricsEngine:
    """
    Streaming evaluation engine over one or many thresholds.

    - Input is a stream of `SliceSample`
    - Output is running aggregated metrics with explicit denominator scopes
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        metric_names: Optional[Sequence[str]] = None,
    ):
        if not thresholds:
            raise ValueError("StreamingMetricsEngine requires at least one threshold.")
        self.thresholds: List[float] = [float(t) for t in thresholds]
        self.metric_names: Sequence[str] = (
            resolve_2d_metric_class_names(metric_names)
            if metric_names is not None
            else tuple(TWOD_METRIC_CLASSES.keys())
        )
        self._states: Dict[float, ThresholdState] = {
            t: ThresholdState(
                threshold=t,
                metrics={name: ScopedRunningStats() for name in self.metric_names},
            )
            for t in self.thresholds
        }

    def update(self, sample: SliceSample) -> None:
        """Consume one sample and update all threshold accumulators."""
        sample.validate()
        gt = sample.ground_truth_mask
        has_foreground = bool((gt > 0.5).sum() > 0)
        pred_input = self._resolve_prediction_input(sample)

        for threshold in self.thresholds:
            state = self._states[threshold]
            state.slice_counts["total"] += 1
            if has_foreground:
                state.slice_counts["foreground"] += 1
            else:
                state.slice_counts["empty"] += 1

            metric_values = compute_metrics_at_threshold(
                pred_input,
                gt,
                threshold=threshold,
                metric_names=self.metric_names,
            )
            self._update_metric_scopes(state, metric_values, has_foreground)

    def run(self, samples: Iterable[SliceSample]) -> Dict[float, Dict[str, object]]:
        """Consume a stream of samples and return finalized threshold results."""
        for sample in samples:
            self.update(sample)
        return self.finalize()

    def finalize(self) -> Dict[float, Dict[str, object]]:
        """Return threshold-keyed finalized metrics."""
        return {threshold: state.to_dict() for threshold, state in self._states.items()}

    @staticmethod
    def _resolve_prediction_input(sample: SliceSample):
        """
        Resolve the prediction tensor used by the metric wrappers.

        For probability sources, we pass probabilities and rely on thresholding
        in the wrapper.
        For post-threshold sources (e.g. nnU-Net masks), we pass the mask tensor.
        """
        if sample.prediction_prob is not None:
            return sample.prediction_prob
        if sample.prediction_mask is not None:
            return sample.prediction_mask
        raise ValueError("SliceSample has neither prediction_prob nor prediction_mask.")

    def _update_metric_scopes(
        self,
        state: ThresholdState,
        metric_values: Dict[str, float],
        has_foreground: bool,
    ) -> None:
        """Update all-slices and foreground-only running stats per metric."""
        for metric_name in self.metric_names:
            value = float(metric_values.get(metric_name, 0.0))
            scoped = state.metrics[metric_name]
            scoped.all_slices.update(value)
            if has_foreground:
                scoped.foreground_only.update(value)


class DualLevelStreamingMetricsEngine:
    """
    Streaming dual-level engine.

    - Updates slice-level 2D metrics per incoming SliceSample
    - Finalizes volume-level 3D metrics as soon as volume boundaries are crossed
    - Keeps memory bounded to one open volume buffer per case-engine instance
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        slice_metric_names: Optional[Sequence[str]] = None,
        volume_metric_names: Optional[Sequence[str]] = None,
        volume_metric_configs: Optional[Mapping[str, Mapping[str, object]]] = None,
        assembler_case_key: str = "default_case",
    ):
        if not thresholds:
            raise ValueError("DualLevelStreamingMetricsEngine requires at least one threshold.")
        self.thresholds: List[float] = [float(t) for t in thresholds]
        self.slice_engine = StreamingMetricsEngine(
            thresholds=self.thresholds,
            metric_names=slice_metric_names,
        )
        self.volume_metric_names: Sequence[str] = tuple(volume_metric_names) if volume_metric_names else tuple(THREED_METRIC_CLASSES.keys())
        self.volume_metric_configs = dict(volume_metric_configs or {})
        self._volume_states: Dict[float, VolumeThresholdState] = {
            t: VolumeThresholdState(
                threshold=t,
                metrics={name: RunningStats() for name in self.volume_metric_names},
            )
            for t in self.thresholds
        }
        self._assembler = VolumeAssembler()
        self._assembler_case_key = str(assembler_case_key)
        self._current_volume_id: Optional[str] = None

    def update(self, sample: SliceSample) -> List[VolumeSample]:
        """
        Consume one slice sample. Returns list of finalized volumes (0 or 1).
        """
        self.slice_engine.update(sample)
        finalized: List[VolumeSample] = []
        if sample.volume_id is None or sample.slice_index is None:
            return finalized

        volume_id = str(sample.volume_id)
        if self._current_volume_id is None:
            self._current_volume_id = volume_id
        elif volume_id != self._current_volume_id:
            prior = self._assembler.finalize_volume(
                analysis_case_key=self._assembler_case_key,
                volume_id=self._current_volume_id,
            )
            if prior is not None:
                self._update_volume_states(prior)
                finalized.append(prior)
            self._current_volume_id = volume_id

        self._assembler.add_sample(self._assembler_case_key, sample)
        return finalized

    def finalize(self) -> Dict[str, Dict[float, Dict[str, object]]]:
        """
        Finalize engine and return both slice and volume level results.
        """
        finalized_volumes = self.finalize_open_volumes()
        # finalized_volumes already reflected in volume state updates
        del finalized_volumes
        return {
            "slice_level": self.slice_engine.finalize(),
            "volume_level": {
                threshold: state.to_dict() for threshold, state in self._volume_states.items()
            },
        }

    def finalize_open_volumes(self) -> List[VolumeSample]:
        """
        Flush any remaining open volumes and return them.
        """
        finalized: List[VolumeSample] = []
        if self._current_volume_id is not None:
            prior = self._assembler.finalize_volume(
                analysis_case_key=self._assembler_case_key,
                volume_id=self._current_volume_id,
            )
            if prior is not None:
                self._update_volume_states(prior)
                finalized.append(prior)
            self._current_volume_id = None
        trailing = self._assembler.finalize_case(self._assembler_case_key)
        for volume_sample in trailing:
            self._update_volume_states(volume_sample)
        finalized.extend(trailing)
        return finalized

    def _update_volume_states(self, volume_sample: VolumeSample) -> None:
        volume_sample.validate()
        num_slices = int(volume_sample.metadata.get("num_slices", int(volume_sample.prediction_volume.shape[-1])))
        has_foreground = bool((volume_sample.ground_truth_volume > 0.5).sum() > 0)
        pred = volume_sample.prediction_volume
        gt = volume_sample.ground_truth_volume
        for threshold in self.thresholds:
            state = self._volume_states[threshold]
            state.volume_counts["total"] += 1
            if has_foreground:
                state.volume_counts["foreground"] += 1
            else:
                state.volume_counts["empty"] += 1
            state.update_volume_slice_count(num_slices)
            metric_values = compute_metrics_3d_at_threshold(
                pred=pred,
                gt=gt,
                threshold=threshold,
                metric_configs=self.volume_metric_configs,
                metric_names=self.volume_metric_names,
            )
            for metric_name in self.volume_metric_names:
                state.metrics[metric_name].update(float(metric_values.get(metric_name, 0.0)))


def evaluate_stream(
    samples: Iterable[SliceSample],
    thresholds: Sequence[float],
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[float, Dict[str, object]]:
    """Convenience function to evaluate a sample stream in one call."""
    engine = StreamingMetricsEngine(thresholds=thresholds, metric_names=metric_names)
    return engine.run(samples)


def get_scope_metric(
    finalized_results: Dict[float, Dict[str, object]],
    threshold: float,
    metric_name: str,
    scope: ScopeName,
) -> Dict[str, float]:
    """Utility accessor for tests and downstream report adapters."""
    return finalized_results[threshold]["metrics"][metric_name][scope]
