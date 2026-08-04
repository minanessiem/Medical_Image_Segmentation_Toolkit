"""Strict Grand Challenge interface-manifest and invocation dispatch."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml


class InterfaceManifestError(ValueError):
    """Raised when platform transport cannot satisfy an explicit interface."""


RESULT_KEYS = frozenset({"mask", "probability"})
TECHNICAL_FIELD_TYPES = frozenset({"string", "integer", "number"})


@dataclass(frozen=True)
class InputBinding:
    slug: str
    dataset_key: str
    relative_path: str
    file_type: str
    cardinality: str


@dataclass(frozen=True)
class TechnicalFieldSchema:
    value_type: str
    nullable: bool


@dataclass(frozen=True)
class TechnicalInputBinding:
    slug: str
    relative_path: str
    file_type: str
    required: bool
    schema: Mapping[str, TechnicalFieldSchema]


@dataclass(frozen=True)
class OutputBinding:
    slug: str
    result_key: str
    relative_path: str
    file_type: str


@dataclass(frozen=True)
class InterfaceDefinition:
    name: str
    inputs: tuple[InputBinding, ...]
    technical_inputs: tuple[TechnicalInputBinding, ...]
    outputs: tuple[OutputBinding, ...]

    @property
    def interface_key(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                [binding.slug for binding in self.inputs]
                + [binding.slug for binding in self.technical_inputs]
            )
        )


@dataclass(frozen=True)
class InterfaceManifest:
    interfaces: tuple[InterfaceDefinition, ...]


@dataclass(frozen=True)
class ResolvedInvocation:
    interface: InterfaceDefinition
    raw_modalities: Mapping[str, Path]
    technical_inputs: Mapping[str, Any]


def load_interface_manifest(path: str | Path) -> InterfaceManifest:
    """Load one strict transport-only interface manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InterfaceManifestError(
            f"Could not load interface manifest {manifest_path}: {exc}"
        ) from exc
    root = _mapping(raw, "interface manifest")
    _unknown(root, {"interfaces"}, "interface manifest")
    raw_interfaces = _sequence(root.get("interfaces"), "interfaces")
    if not raw_interfaces:
        raise InterfaceManifestError("Interface manifest must define at least one interface.")

    interfaces = tuple(
        _parse_interface(value, index=index)
        for index, value in enumerate(raw_interfaces)
    )
    names = [item.name for item in interfaces]
    if len(names) != len(set(names)):
        raise InterfaceManifestError("Interface names must be unique.")
    keys = [item.interface_key for item in interfaces]
    if len(keys) != len(set(keys)):
        raise InterfaceManifestError(
            "Each configured interface must have a unique sorted socket-slug key."
        )
    return InterfaceManifest(interfaces=interfaces)


def validate_dataset_bindings(
    manifest: InterfaceManifest,
    *,
    required_raw_keys: Sequence[str],
) -> None:
    """Require every interface to bind exactly the adapter's raw modalities."""

    required = {str(value).strip() for value in required_raw_keys}
    if not required or "" in required:
        raise InterfaceManifestError(
            "Dataset preprocessing must declare non-empty required raw keys."
        )
    for interface in manifest.interfaces:
        observed = {binding.dataset_key for binding in interface.inputs}
        missing = sorted(required - observed)
        unsupported = sorted(observed - required)
        if missing or unsupported:
            raise InterfaceManifestError(
                f"Interface {interface.name!r} does not match the selected dataset "
                f"adapter; missing={missing}, unsupported={unsupported}."
            )


def resolve_invocation(
    manifest: InterfaceManifest,
    input_root: str | Path,
) -> ResolvedInvocation:
    """Dispatch `/input/inputs.json` and return canonical raw-key paths."""

    if not isinstance(manifest, InterfaceManifest):
        raise InterfaceManifestError("manifest must be an InterfaceManifest.")
    root = Path(input_root).expanduser().resolve()
    inputs_path = root / "inputs.json"
    try:
        payload = json.loads(inputs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InterfaceManifestError("Could not read the platform inputs.json file.") from exc
    entries = _sequence(payload, "inputs.json")
    socket_entries: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(entries):
        entry = _mapping(value, f"inputs.json[{index}]")
        socket = _mapping(entry.get("socket"), f"inputs.json[{index}].socket")
        slug = _string(socket.get("slug"), f"inputs.json[{index}].socket.slug")
        if slug in socket_entries:
            raise InterfaceManifestError(
                f"inputs.json contains duplicate socket slug {slug!r}."
            )
        socket_entries[slug] = socket

    interface_key = tuple(sorted(socket_entries))
    matches = [
        interface
        for interface in manifest.interfaces
        if interface.interface_key == interface_key
    ]
    if len(matches) != 1:
        configured = [item.interface_key for item in manifest.interfaces]
        raise InterfaceManifestError(
            "No configured interface matches the supplied socket set; "
            f"observed={interface_key}, configured={configured}."
        )
    interface = matches[0]

    raw_modalities: dict[str, Path] = {}
    for binding in interface.inputs:
        _validate_platform_relative_path(socket_entries[binding.slug], binding)
        directory = _resolve_under_root(root, binding.relative_path)
        raw_modalities[binding.dataset_key] = _resolve_single_nifti(
            directory,
            input_root=root,
        )

    technical_values: dict[str, Any] = {}
    for binding in interface.technical_inputs:
        _validate_platform_relative_path(socket_entries[binding.slug], binding)
        location = _resolve_under_root(root, binding.relative_path)
        try:
            value = json.loads(
                location.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise InterfaceManifestError(
                f"Required technical JSON socket {binding.slug!r} is invalid."
            ) from exc
        _validate_technical_value(value, binding=binding)
        technical_values[binding.slug] = value

    return ResolvedInvocation(
        interface=interface,
        raw_modalities=MappingProxyType(raw_modalities),
        technical_inputs=MappingProxyType(technical_values),
    )


def _parse_interface(value: Any, *, index: int) -> InterfaceDefinition:
    path = f"interfaces[{index}]"
    raw = _mapping(value, path)
    _unknown(raw, {"name", "inputs", "technical_inputs", "outputs"}, path)
    name = _string(raw.get("name"), f"{path}.name")
    raw_inputs = _sequence(raw.get("inputs"), f"{path}.inputs")
    if not raw_inputs:
        raise InterfaceManifestError(f"{path}.inputs must not be empty.")
    inputs = tuple(
        _parse_input(item, path=f"{path}.inputs[{item_index}]")
        for item_index, item in enumerate(raw_inputs)
    )
    raw_technical = _sequence(
        raw.get("technical_inputs", []),
        f"{path}.technical_inputs",
    )
    technical = tuple(
        _parse_technical(item, path=f"{path}.technical_inputs[{item_index}]")
        for item_index, item in enumerate(raw_technical)
    )
    raw_outputs = _sequence(raw.get("outputs"), f"{path}.outputs")
    if not raw_outputs:
        raise InterfaceManifestError(f"{path}.outputs must not be empty.")
    outputs = tuple(
        _parse_output(item, path=f"{path}.outputs[{item_index}]")
        for item_index, item in enumerate(raw_outputs)
    )

    slugs = (
        [binding.slug for binding in inputs]
        + [binding.slug for binding in technical]
        + [binding.slug for binding in outputs]
    )
    if len(slugs) != len(set(slugs)):
        raise InterfaceManifestError(f"{path} input/output socket slugs must be unique.")
    keys = [binding.dataset_key for binding in inputs]
    if len(keys) != len(set(keys)):
        raise InterfaceManifestError(f"{path} input dataset_key values must be unique.")
    result_keys = [binding.result_key for binding in outputs]
    if len(result_keys) != len(set(result_keys)):
        raise InterfaceManifestError(f"{path} output result_key values must be unique.")
    relative_paths = (
        [binding.relative_path for binding in inputs]
        + [binding.relative_path for binding in technical]
        + [binding.relative_path for binding in outputs]
    )
    if len(relative_paths) != len(set(relative_paths)):
        raise InterfaceManifestError(f"{path} interface relative paths must be unique.")
    return InterfaceDefinition(
        name=name,
        inputs=inputs,
        technical_inputs=technical,
        outputs=outputs,
    )


def _parse_input(value: Any, *, path: str) -> InputBinding:
    raw = _mapping(value, path)
    _unknown(
        raw,
        {"slug", "dataset_key", "relative_path", "file_type", "cardinality"},
        path,
    )
    file_type = _string(raw.get("file_type"), f"{path}.file_type").lower()
    cardinality = _string(raw.get("cardinality"), f"{path}.cardinality").lower()
    if file_type != "nifti":
        raise InterfaceManifestError(
            f"{path}.file_type must be 'nifti' for the current certified transport."
        )
    if cardinality != "one":
        raise InterfaceManifestError(f"{path}.cardinality must be 'one'.")
    return InputBinding(
        slug=_string(raw.get("slug"), f"{path}.slug"),
        dataset_key=_string(raw.get("dataset_key"), f"{path}.dataset_key"),
        relative_path=_relative_path(raw.get("relative_path"), f"{path}.relative_path"),
        file_type=file_type,
        cardinality=cardinality,
    )


def _parse_technical(value: Any, *, path: str) -> TechnicalInputBinding:
    raw = _mapping(value, path)
    _unknown(
        raw,
        {"slug", "relative_path", "file_type", "required", "schema"},
        path,
    )
    file_type = _string(raw.get("file_type"), f"{path}.file_type").lower()
    if file_type != "json":
        raise InterfaceManifestError(f"{path}.file_type must be 'json'.")
    required = raw.get("required")
    if type(required) is not bool:
        raise InterfaceManifestError(f"{path}.required must be a boolean.")
    if not required:
        raise InterfaceManifestError(
            f"{path}.required=false is ambiguous for an exact platform interface; "
            "represent the optional socket as a separate interface."
        )
    return TechnicalInputBinding(
        slug=_string(raw.get("slug"), f"{path}.slug"),
        relative_path=_relative_path(raw.get("relative_path"), f"{path}.relative_path"),
        file_type=file_type,
        required=required,
        schema=_parse_technical_schema(raw.get("schema", {}), path=f"{path}.schema"),
    )


def _parse_output(value: Any, *, path: str) -> OutputBinding:
    raw = _mapping(value, path)
    _unknown(raw, {"slug", "result_key", "relative_path", "file_type"}, path)
    file_type = _string(raw.get("file_type"), f"{path}.file_type").lower()
    if file_type not in {"nifti", "mha"}:
        raise InterfaceManifestError(
            f"{path}.file_type must be one of ['mha', 'nifti']."
        )
    result_key = _string(raw.get("result_key"), f"{path}.result_key").lower()
    if result_key not in RESULT_KEYS:
        raise InterfaceManifestError(
            f"{path}.result_key must be one of {sorted(RESULT_KEYS)}."
        )
    return OutputBinding(
        slug=_string(raw.get("slug"), f"{path}.slug"),
        result_key=result_key,
        relative_path=_relative_path(raw.get("relative_path"), f"{path}.relative_path"),
        file_type=file_type,
    )


def _parse_technical_schema(
    value: Any,
    *,
    path: str,
) -> Mapping[str, TechnicalFieldSchema]:
    raw_schema = _mapping(value, path)
    parsed: dict[str, TechnicalFieldSchema] = {}
    for field_name, field_value in raw_schema.items():
        name = _string(field_name, f"{path} field name")
        field = _mapping(field_value, f"{path}.{name}")
        _unknown(field, {"type", "nullable"}, f"{path}.{name}")
        value_type = _string(field.get("type"), f"{path}.{name}.type").lower()
        if value_type not in TECHNICAL_FIELD_TYPES:
            raise InterfaceManifestError(
                f"{path}.{name}.type must be one of "
                f"{sorted(TECHNICAL_FIELD_TYPES)}."
            )
        nullable = field.get("nullable")
        if type(nullable) is not bool:
            raise InterfaceManifestError(f"{path}.{name}.nullable must be a boolean.")
        parsed[name] = TechnicalFieldSchema(
            value_type=value_type,
            nullable=nullable,
        )
    return MappingProxyType(parsed)


def _validate_technical_value(
    value: Any,
    *,
    binding: TechnicalInputBinding,
) -> None:
    if not isinstance(value, Mapping):
        raise InterfaceManifestError(
            f"Technical JSON socket {binding.slug!r} must be a mapping."
        )
    missing = sorted(set(binding.schema) - set(value))
    if missing:
        raise InterfaceManifestError(
            f"Technical JSON socket {binding.slug!r} is missing required fields: "
            f"{missing}."
        )
    for name, field in binding.schema.items():
        observed = value[name]
        if observed is None:
            if field.nullable:
                continue
            raise InterfaceManifestError(
                f"Technical field {name!r} is not nullable."
            )
        if field.value_type == "string":
            valid = isinstance(observed, str)
        elif field.value_type == "integer":
            valid = type(observed) is int
        else:
            valid = type(observed) in {int, float}
        if not valid:
            raise InterfaceManifestError(
                f"Technical field {name!r} must be a {field.value_type}."
            )
        if field.value_type in {"integer", "number"} and not math.isfinite(
            float(observed)
        ):
            raise InterfaceManifestError(
                f"Technical field {name!r} must be finite."
            )


def _validate_platform_relative_path(
    socket: Mapping[str, Any],
    binding: InputBinding | TechnicalInputBinding,
) -> None:
    observed = _relative_path(
        socket.get("relative_path"),
        f"inputs.json socket {binding.slug!r} relative_path",
    )
    if observed != binding.relative_path:
        raise InterfaceManifestError(
            f"Socket {binding.slug!r} relative_path disagrees with the release "
            f"manifest: observed={observed!r}, expected={binding.relative_path!r}."
        )


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise InterfaceManifestError("Resolved interface path escapes /input.") from exc
    return candidate


def _resolve_single_nifti(directory: Path, *, input_root: Path) -> Path:
    if not directory.is_dir():
        raise InterfaceManifestError("Required NIfTI socket directory does not exist.")
    matches: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file() or not path.name.lower().endswith(".nii.gz"):
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(input_root)
        except ValueError as exc:
            raise InterfaceManifestError(
                "Resolved NIfTI input file escapes /input."
            ) from exc
        matches.append(resolved)
    matches.sort()
    if len(matches) != 1:
        raise InterfaceManifestError(
            "A single-value NIfTI socket must contain exactly one .nii.gz file; "
            f"observed_count={len(matches)}."
        )
    return matches[0]


def _relative_path(value: Any, path: str) -> str:
    token = _string(value, path)
    if "\\" in token:
        raise InterfaceManifestError(f"{path} must be a POSIX relative path.")
    candidate = PurePosixPath(token)
    if candidate.is_absolute() or not candidate.parts or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise InterfaceManifestError(f"{path} must be a safe relative path.")
    return candidate.as_posix()


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterfaceManifestError(f"{path} must be a mapping.")
    return dict(value)


def _sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise InterfaceManifestError(f"{path} must be a list.")
    return list(value)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceManifestError(f"{path} must be a non-empty string.")
    return value.strip()


def _unknown(raw: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise InterfaceManifestError(f"{path} contains unknown keys: {unknown}.")


__all__ = [
    "InputBinding",
    "InterfaceDefinition",
    "InterfaceManifest",
    "InterfaceManifestError",
    "OutputBinding",
    "ResolvedInvocation",
    "TechnicalInputBinding",
    "TechnicalFieldSchema",
    "load_interface_manifest",
    "resolve_invocation",
    "validate_dataset_bindings",
]
