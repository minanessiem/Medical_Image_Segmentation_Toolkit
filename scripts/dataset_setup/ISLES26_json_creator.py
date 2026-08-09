import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


SESSION = "ses-1"
ANAT_DIR = "anat"
TRAINING_ROOT_CANDIDATES = (
    Path("atlas21_training_raw") / "Training_Raw",
    Path("Training_Raw"),
    Path("."),
)
DAYS_POST_STROKE_COLUMN = "DAYS_POST_STROKE"
DAYS_POST_STROKE_BINS = (
    -np.inf,
    7,
    30,
    90,
    180,
    365,
    730,
    np.inf,
)
DAYS_POST_STROKE_LABELS = (
    "0-7 days",
    "8-30 days",
    "31-90 days",
    "91-180 days",
    "181-365 days",
    "366-730 days",
    ">730 days",
)
VALIDATION_BALANCE_COLUMNS = ("DAYS_POST_STROKE_BIN", "CHRONICITY", "ATLAS2_DATASET")
FULL_VALIDATION_SPLITS = ("val_rest", "val_fast")


@dataclass(frozen=True)
class Isles26Case:
    case_id: str
    site_id: str
    t1_path: Path
    label_path: Path
    metadata_path: Path


def resolve_training_root(dataset_path: Path) -> Path:
    """Resolve the directory containing center folders with subject children."""
    for candidate in TRAINING_ROOT_CANDIDATES:
        training_root = dataset_path / candidate
        if _contains_site_directories(training_root):
            return training_root

    raise FileNotFoundError(
        "Could not find ISLES26 training root. Expected center folders containing "
        f"'sub-*' directories under one of: "
        f"{', '.join(str(dataset_path / c) for c in TRAINING_ROOT_CANDIDATES)}"
    )


def _is_subject_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("sub-")


def _is_site_dir(path: Path) -> bool:
    return path.is_dir() and any(_is_subject_dir(child) for child in path.iterdir())


def _contains_site_directories(path: Path) -> bool:
    return path.is_dir() and any(_is_site_dir(child) for child in path.iterdir())


def discover_cases(training_root: Path) -> List[Isles26Case]:
    """Discover ISLES26 cases and expected file paths without applying split logic."""
    cases = []

    for site_dir in sorted(_iter_site_dirs(training_root), key=_site_sort_key):
        for subject_dir in sorted(_iter_subject_dirs(site_dir), key=_subject_sort_key):
            anat_dir = subject_dir / SESSION / ANAT_DIR
            case_id = subject_dir.name

            cases.append(
                Isles26Case(
                    case_id=case_id,
                    site_id=site_dir.name,
                    t1_path=anat_dir / f"{case_id}_{SESSION}_space-orig_desc-brain_T1w.nii.gz",
                    label_path=anat_dir / f"{case_id}_{SESSION}_space-orig_label-lesion_desc-T1lesion_mask.nii.gz",
                    metadata_path=anat_dir / f"{case_id}_{SESSION}_metadata.csv",
                )
            )

    return cases


def _iter_site_dirs(training_root: Path) -> Iterable[Path]:
    return (child for child in training_root.iterdir() if _is_site_dir(child))


def _iter_subject_dirs(site_dir: Path) -> Iterable[Path]:
    return (child for child in site_dir.iterdir() if _is_subject_dir(child))


def _site_sort_key(path: Path) -> tuple[int, str]:
    return _extract_first_int(path.name), path.name.casefold()


def _subject_sort_key(path: Path) -> tuple[int, int]:
    match = re.match(r"sub-r(\d+)s(\d+)", path.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, _extract_first_int(path.name)


def _extract_first_int(value: str) -> int:
    digits = "".join(char for char in value if char.isdigit())
    return int(digits) if digits else -1


def validate_case(case: Isles26Case, check_valid_mask: bool = False) -> bool:
    """Return True when all required files exist and the mask is optionally non-empty."""
    required_paths = (case.t1_path, case.label_path, case.metadata_path)
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print(f"Skipping {case.case_id}: missing {', '.join(str(path) for path in missing_paths)}")
        return False

    if check_valid_mask and not contains_valid_mask(case.label_path):
        return False

    return True


def contains_valid_mask(mask_path: Path) -> bool:
    """Check whether a NIfTI lesion mask contains at least one positive voxel."""
    lesion_mask = nib.load(str(mask_path))
    lesion_data = lesion_mask.get_fdata()

    if np.any(lesion_data > 0):
        return True

    print(f"Skipping {mask_path}: no positive lesion labels found.")
    return False


def load_metadata_row(metadata_path: Path) -> Dict[str, Any]:
    """Load the first row of an ISLES26 metadata CSV."""
    metadata_df = pd.read_csv(metadata_path)

    if metadata_df.empty:
        return {column: None for column in metadata_df.columns}

    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        metadata_df[DAYS_POST_STROKE_COLUMN] = pd.to_numeric(
            metadata_df[DAYS_POST_STROKE_COLUMN],
            errors="coerce",
        )

    if "CHRONICITY" in metadata_df.columns:
        metadata_df["CHRONICITY"] = pd.to_numeric(
            metadata_df["CHRONICITY"].replace("", 0).fillna(0),
            errors="coerce",
        ).fillna(0).astype(int)

    return metadata_df.iloc[0].to_dict()


def _metadata_site(metadata: Dict[str, Any], fallback: str) -> str:
    value = metadata.get("SITE")
    if value is None or pd.isna(value) or not str(value).strip():
        return fallback
    return str(value)


def assign_round_robin_folds(cases: List[Isles26Case], num_folds: int) -> Dict[str, int]:
    """Assign folds by sorted case order until ISLES26 stratification is implemented."""
    return {case.case_id: index % num_folds for index, case in enumerate(cases)}


def build_metadata_dataframe(cases: List[Isles26Case], training_root: Path) -> pd.DataFrame:
    """Build the case-level metadata frame used for split assignment and reporting."""
    rows = []
    for case in cases:
        metadata = load_metadata_row(case.metadata_path)
        metadata["caseID"] = case.case_id
        metadata["siteID"] = case.site_id
        metadata["SITE"] = _metadata_site(metadata, case.site_id)
        metadata["metadata_csv"] = case.metadata_path.relative_to(training_root).as_posix()
        rows.append(metadata)

    metadata_df = pd.DataFrame(rows)

    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        metadata_df[DAYS_POST_STROKE_COLUMN] = pd.to_numeric(
            metadata_df[DAYS_POST_STROKE_COLUMN],
            errors="coerce",
        )
        metadata_df["DAYS_POST_STROKE_BIN"] = pd.cut(
            metadata_df[DAYS_POST_STROKE_COLUMN],
            bins=DAYS_POST_STROKE_BINS,
            labels=DAYS_POST_STROKE_LABELS,
        )

    if "CHRONICITY" in metadata_df.columns:
        metadata_df["CHRONICITY"] = pd.to_numeric(
            metadata_df["CHRONICITY"].replace("", 0).fillna(0),
            errors="coerce",
        ).fillna(0).astype(int)

    return metadata_df


def validate_nested_split_sizes(
    metadata_df: pd.DataFrame,
    val_full_percent: float,
    val_fast_percent: float,
) -> tuple[int, int]:
    """Validate percentages and return integer targets on the full dataset size."""
    if not 0 < val_full_percent < 100:
        raise ValueError("--val_full_size must be between 0 and 100")
    if not 0 < val_fast_percent < 100:
        raise ValueError("--val_fast_size must be between 0 and 100")
    if val_fast_percent >= val_full_percent:
        raise ValueError("--val_fast_size must be smaller than --val_full_size")

    total_cases = len(metadata_df)
    target_val_full_count = int(round(total_cases * (val_full_percent / 100.0)))
    target_val_fast_count = int(round(total_cases * (val_fast_percent / 100.0)))

    if target_val_full_count <= 0 or target_val_full_count >= total_cases:
        raise ValueError(
            f"{val_full_percent}% gives {target_val_full_count} validation cases; "
            f"it must be between 1 and {total_cases - 1}."
        )
    if target_val_fast_count <= 0:
        raise ValueError(
            f"{val_fast_percent}% gives {target_val_fast_count} val_fast cases; it must be at least 1."
        )
    if target_val_fast_count >= target_val_full_count:
        raise ValueError(
            f"Rounded counts violate nesting: val_fast={target_val_fast_count} must be smaller "
            f"than val_full (val_rest+val_fast)={target_val_full_count}."
        )

    return target_val_full_count, target_val_fast_count


def assign_site_balanced_validation_pool(
    metadata_df: pd.DataFrame,
    target_val_full_count: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train vs validation_pool, keeping singleton sites in training."""
    if "SITE" not in metadata_df.columns:
        raise ValueError("SITE column is required for site-balanced validation assignment.")

    split_df = metadata_df.copy()
    site_counts = split_df["SITE"].value_counts().sort_index()
    singleton_sites = site_counts[site_counts == 1].index.tolist()
    splittable_site_counts = site_counts[site_counts >= 2]

    min_val_count = len(splittable_site_counts)
    max_val_count = int((splittable_site_counts - 1).sum())
    if target_val_full_count < min_val_count:
        raise ValueError(
            f"Target validation size {target_val_full_count} is too small; at least {min_val_count} "
            "cases are needed to place every multi-case site in validation."
        )
    if target_val_full_count > max_val_count:
        raise ValueError(
            f"Target validation size {target_val_full_count} is too large; at most {max_val_count} "
            "cases are allowed while keeping every multi-case site in training and singleton "
            "sites assigned to training."
        )

    site_quotas = allocate_group_quotas(
        counts=splittable_site_counts,
        target=target_val_full_count,
        min_per_group=1,
        max_per_group=splittable_site_counts - 1,
    )

    rng = np.random.default_rng(seed)
    validation_pool_indices = []
    for site, quota in site_quotas.items():
        site_df = split_df[split_df["SITE"] == site]
        validation_pool_indices.extend(select_representative_site_cases(site_df, quota, rng))

    split_df["split"] = "train"
    split_df.loc[validation_pool_indices, "split"] = "validation_pool"
    validation_pool_df = split_df[split_df["split"] == "validation_pool"]

    split_df.attrs["outer_split_seed"] = seed
    split_df.attrs["singleton_training_sites"] = singleton_sites
    split_df.attrs["outer_balance_score"] = score_subset_balance(split_df, validation_pool_df)
    return split_df


def assign_site_balanced_fast_subset(
    outer_split_df: pd.DataFrame,
    target_val_fast_count: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign val_fast within validation_pool without strict one-site minimums."""
    if "SITE" not in outer_split_df.columns:
        raise ValueError("SITE column is required for site-balanced val_fast assignment.")

    validation_pool_mask = outer_split_df["split"] == "validation_pool"
    validation_pool_df = outer_split_df[validation_pool_mask]
    if validation_pool_df.empty:
        raise ValueError("validation_pool is empty; cannot assign val_fast.")
    if target_val_fast_count >= len(validation_pool_df):
        raise ValueError(
            f"val_fast target {target_val_fast_count} must be smaller than "
            f"validation_pool size {len(validation_pool_df)}."
        )

    site_counts = validation_pool_df["SITE"].value_counts().sort_index()
    site_quotas = allocate_group_quotas(
        counts=site_counts,
        target=target_val_fast_count,
        min_per_group=0,
        max_per_group=site_counts,
    )

    rng = np.random.default_rng(seed)
    val_fast_indices = []
    for site, quota in site_quotas.items():
        if quota == 0:
            continue
        site_df = validation_pool_df[validation_pool_df["SITE"] == site]
        val_fast_indices.extend(select_representative_site_cases(site_df, quota, rng))

    split_df = outer_split_df.copy()
    split_df.attrs.update(outer_split_df.attrs)
    split_df.loc[validation_pool_mask, "split"] = "val_rest"
    split_df.loc[val_fast_indices, "split"] = "val_fast"

    full_validation_df = split_df[split_df["split"].isin(FULL_VALIDATION_SPLITS)]
    val_fast_df = split_df[split_df["split"] == "val_fast"]
    score_fast_to_full_validation = score_subset_balance(full_validation_df, val_fast_df)
    score_fast_to_all = score_subset_balance(split_df, val_fast_df)

    split_df.attrs["inner_split_seed"] = seed
    split_df.attrs["inner_balance_score"] = score_fast_to_full_validation
    split_df.attrs["inner_balance_to_all_score"] = score_fast_to_all
    split_df.attrs["nested_balance_score"] = (
        split_df.attrs.get("outer_balance_score", np.inf)
        + score_fast_to_full_validation
        + score_fast_to_all
    )
    return split_df


def generate_candidate_seeds(seed: int, num_candidates: Optional[int]) -> List[int]:
    if num_candidates is None:
        return [int(seed)]

    seed_rng = np.random.default_rng(seed)
    return seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=num_candidates,
        dtype=np.uint32,
    ).astype(int).tolist()


def select_nested_validation_split(
    metadata_df: pd.DataFrame,
    val_full_percent: float,
    val_fast_percent: float,
    seed: int = 42,
    num_outer_split_seeds: Optional[int] = None,
    num_inner_split_seeds: Optional[int] = None,
) -> pd.DataFrame:
    """Choose a nested train/val_rest/val_fast split."""
    target_val_full_count, target_val_fast_count = validate_nested_split_sizes(
        metadata_df=metadata_df,
        val_full_percent=val_full_percent,
        val_fast_percent=val_fast_percent,
    )

    outer_candidate_seeds = generate_candidate_seeds(seed, num_outer_split_seeds)
    best_split = None
    best_score = np.inf
    outer_iterable = outer_candidate_seeds
    if len(outer_candidate_seeds) > 1:
        outer_iterable = tqdm(
            outer_candidate_seeds,
            desc="Evaluating outer validation seeds",
            unit="seed",
        )

    for outer_seed in outer_iterable:
        outer_split = assign_site_balanced_validation_pool(
            metadata_df=metadata_df,
            target_val_full_count=target_val_full_count,
            seed=outer_seed,
        )
        inner_candidate_seeds = generate_candidate_seeds(outer_seed, num_inner_split_seeds)
        for inner_seed in inner_candidate_seeds:
            candidate_split = assign_site_balanced_fast_subset(
                outer_split_df=outer_split,
                target_val_fast_count=target_val_fast_count,
                seed=inner_seed,
            )
            candidate_score = candidate_split.attrs["nested_balance_score"]
            if candidate_score < best_score:
                best_split = candidate_split
                best_score = candidate_score

    if best_split is None:
        raise ValueError("No nested split candidates were generated.")

    best_split.attrs["base_seed"] = seed
    best_split.attrs["num_outer_split_candidates"] = len(outer_candidate_seeds)
    best_split.attrs["num_inner_split_candidates"] = (
        len(generate_candidate_seeds(seed, num_inner_split_seeds))
        if num_inner_split_seeds is not None
        else 1
    )
    best_split.attrs["target_val_full_count"] = target_val_full_count
    best_split.attrs["target_val_fast_count"] = target_val_fast_count
    return best_split


def allocate_group_quotas(
    counts: pd.Series,
    target: int,
    min_per_group: int,
    max_per_group: pd.Series,
) -> pd.Series:
    """Allocate integer quotas proportionally while respecting group bounds."""
    proportions = counts / counts.sum()
    ideal = proportions * target
    quotas = np.floor(ideal).astype(int)
    quotas = quotas.clip(lower=min_per_group)
    quotas = pd.Series(np.minimum(quotas, max_per_group), index=counts.index)

    while quotas.sum() < target:
        eligible = quotas[quotas < max_per_group]
        if eligible.empty:
            break
        remainders = (ideal - quotas).loc[eligible.index].sort_values(ascending=False)
        quotas.loc[remainders.index[0]] += 1

    while quotas.sum() > target:
        eligible = quotas[quotas > min_per_group]
        if eligible.empty:
            break
        remainders = (ideal - quotas).loc[eligible.index].sort_values(ascending=True)
        quotas.loc[remainders.index[0]] -= 1

    return quotas.astype(int)


def select_representative_site_cases(
    site_df: pd.DataFrame,
    quota: int,
    rng: np.random.Generator,
) -> List[int]:
    """Sample cases within a site across available time/chronicity strata."""
    stratification_columns = _existing_columns(site_df, ["DAYS_POST_STROKE_BIN", "CHRONICITY"])
    if not stratification_columns:
        return rng.choice(site_df.index.to_numpy(), size=quota, replace=False).tolist()

    stratum_labels = site_df[stratification_columns].apply(
        lambda row: " | ".join(str(value) if pd.notna(value) else "Missing" for value in row),
        axis=1,
    )
    stratum_counts = stratum_labels.value_counts()
    stratum_quotas = allocate_group_quotas(
        counts=stratum_counts,
        target=quota,
        min_per_group=0,
        max_per_group=stratum_counts,
    )

    selected_indices = []
    for stratum, stratum_quota in stratum_quotas.items():
        if stratum_quota == 0:
            continue
        candidate_indices = site_df.index[stratum_labels == stratum].to_numpy()
        selected_indices.extend(
            rng.choice(candidate_indices, size=stratum_quota, replace=False).tolist()
        )

    return selected_indices


def score_subset_balance(reference_df: pd.DataFrame, subset_df: pd.DataFrame) -> float:
    """Score representativeness by summed percentage-point deviations."""
    if subset_df.empty:
        return np.inf

    score = 0.0
    for column in _existing_columns(reference_df, VALIDATION_BALANCE_COLUMNS):
        if column not in subset_df.columns:
            continue
        comparison = build_distribution_comparison(
            reference_df=reference_df,
            subset_df=subset_df,
            column=column,
            reference_label="reference",
            subset_label="subset",
        )
        score += comparison["subset_minus_reference_pp"].abs().sum()
    return float(score)


def build_split_mapping(split_df: pd.DataFrame) -> Dict[str, str]:
    """Explicit split mapping used by Option B JSON output."""
    return {
        row.caseID: row.split
        for row in split_df[["caseID", "split"]].itertuples(index=False)
    }


def build_case_entry(
    case: Isles26Case,
    training_root: Path,
    split_label: Optional[str] = None,
    fold: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one MSD-style JSON entry using paths relative to the training root."""
    metadata = load_metadata_row(case.metadata_path)
    site = _metadata_site(metadata, case.site_id)

    case_entry = {
        "caseID": case.case_id,
        "siteID": case.site_id,
        "T1": [_relative_posix(case.t1_path, training_root)],
        "label": _relative_posix(case.label_path, training_root),
        "metadata_csv": _relative_posix(case.metadata_path, training_root),
        "metadata": {
            "ATLAS2_DATASET": _json_safe_value(metadata.get("ATLAS2_DATASET")),
            "SESSION_ID": _json_safe_value(metadata.get("SESSION_ID")),
            "DAYS_POST_STROKE": _json_safe_value(metadata.get("DAYS_POST_STROKE")),
            "CHRONICITY": _json_safe_value(metadata.get("CHRONICITY")),
            "SITE": site,
        },
    }

    if fold is not None:
        case_entry["fold"] = fold
    if split_label is not None:
        case_entry["split"] = split_label

    return case_entry


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def print_nested_validation_report(
    metadata_df: pd.DataFrame,
    val_full_size: float,
    val_fast_size: float,
) -> None:
    print_header("Nested Validation Split Simulation")

    split_counts = metadata_df["split"].value_counts()
    train_count = int(split_counts.get("train", 0))
    val_rest_count = int(split_counts.get("val_rest", 0))
    val_fast_count = int(split_counts.get("val_fast", 0))
    full_validation_count = val_rest_count + val_fast_count

    outer_seed = metadata_df.attrs.get("outer_split_seed")
    inner_seed = metadata_df.attrs.get("inner_split_seed")
    base_seed = metadata_df.attrs.get("base_seed")
    num_outer_candidates = metadata_df.attrs.get("num_outer_split_candidates", 1)
    num_inner_candidates = metadata_df.attrs.get("num_inner_split_candidates", 1)

    print(f"Requested val_full size [val_rest+val_fast]: {val_full_size:.2f}%")
    print(f"Requested val_fast size: {val_fast_size:.2f}%")
    print(f"Train cases: {train_count}")
    print(f"val_rest cases: {val_rest_count}")
    print(f"val_fast cases: {val_fast_count}")
    print(f"Full validation cases (val_full = val_rest+val_fast): {full_validation_count}")
    print(f"Actual val_fast size: {(val_fast_count / len(metadata_df)) * 100:.2f}%")
    print(f"Actual val_full size: {(full_validation_count / len(metadata_df)) * 100:.2f}%")

    print(f"Base seed: {base_seed}")
    print(f"Outer candidate seeds tried: {num_outer_candidates}")
    print(f"Inner candidate seeds tried per outer seed: {num_inner_candidates}")
    print(f"Selected outer split seed: {outer_seed}")
    print(f"Selected inner split seed: {inner_seed}")

    outer_score = metadata_df.attrs.get("outer_balance_score")
    inner_score = metadata_df.attrs.get("inner_balance_score")
    inner_to_all_score = metadata_df.attrs.get("inner_balance_to_all_score")
    nested_score = metadata_df.attrs.get("nested_balance_score")
    if outer_score is not None:
        print(f"Outer score [dist(val_full, all)]: {outer_score:.4f}")
    if inner_score is not None:
        print(f"Inner score [dist(val_fast, val_full)]: {inner_score:.4f}")
    if inner_to_all_score is not None:
        print(f"Inner score [dist(val_fast, all)]: {inner_to_all_score:.4f}")
    if nested_score is not None:
        print(f"Total nested score (lower is better): {nested_score:.4f}")

    singleton_training_sites = metadata_df.attrs.get("singleton_training_sites", [])
    print(
        "Assignment policy: outer split keeps every singleton site in training and keeps at least "
        "one case from every multi-case site in both train and full validation; inner split uses "
        "proportional site quotas without strict one-case-per-site requirements."
    )
    print(f"Singleton sites assigned to training only: {singleton_training_sites}")
    print("JSON output policy: split labels only (train/val_rest/val_fast), no fold field.")

    print("\nSITE coverage:")
    site_split_counts = pd.crosstab(metadata_df["SITE"], metadata_df["split"])
    site_split_counts["total"] = site_split_counts.sum(axis=1)
    site_split_counts["full_validation_count"] = (
        site_split_counts.get("val_rest", 0) + site_split_counts.get("val_fast", 0)
    )
    site_split_counts["full_validation_percent"] = (
        site_split_counts["full_validation_count"] / site_split_counts["total"] * 100
    ).round(2)
    site_split_counts["val_fast_percent"] = (
        site_split_counts.get("val_fast", 0) / site_split_counts["total"] * 100
    ).round(2)
    print(site_split_counts.to_string())

    full_validation_df = metadata_df[metadata_df["split"].isin(FULL_VALIDATION_SPLITS)]
    val_fast_df = metadata_df[metadata_df["split"] == "val_fast"]
    comparison_columns = _existing_columns(
        metadata_df,
        ["SITE", "DAYS_POST_STROKE_BIN", "CHRONICITY", "ATLAS2_DATASET"],
    )

    print("\nRepresentativeness checks: val_full (val_rest+val_fast) vs all")
    for column in comparison_columns:
        print_distribution_comparison(
            reference_df=metadata_df,
            subset_df=full_validation_df,
            column=column,
            reference_label="all",
            subset_label="val_full",
        )

    print("\nRepresentativeness checks: val_fast vs val_full (val_rest+val_fast)")
    for column in comparison_columns:
        print_distribution_comparison(
            reference_df=full_validation_df,
            subset_df=val_fast_df,
            column=column,
            reference_label="val_full",
            subset_label="val_fast",
        )

    print("\nRepresentativeness checks: val_fast vs all")
    for column in comparison_columns:
        print_distribution_comparison(
            reference_df=metadata_df,
            subset_df=val_fast_df,
            column=column,
            reference_label="all",
            subset_label="val_fast",
        )

    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        print("\nDAYS_POST_STROKE by split:")
        print(metadata_df.groupby("split")[DAYS_POST_STROKE_COLUMN].describe().round(2).to_string())


def print_distribution_comparison(
    reference_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    column: str,
    reference_label: str,
    subset_label: str,
) -> None:
    comparison = build_distribution_comparison(
        reference_df=reference_df,
        subset_df=subset_df,
        column=column,
        reference_label=reference_label,
        subset_label=subset_label,
    )
    print(f"\n{column}: {subset_label} vs {reference_label}")
    print(comparison.to_string(index=False))


def build_distribution_comparison(
    reference_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    column: str,
    reference_label: str,
    subset_label: str,
) -> pd.DataFrame:
    reference_values = _display_series(reference_df[column])
    subset_values = _display_series(subset_df[column])
    reference_total = len(reference_df)
    subset_total = len(subset_df)
    rows = []

    value_order = reference_values.value_counts().index.tolist()
    for value in subset_values.value_counts().index.tolist():
        if value not in value_order:
            value_order.append(value)

    for value in value_order:
        reference_count = int((reference_values == value).sum())
        subset_count = int((subset_values == value).sum())
        reference_percent = (reference_count / reference_total) * 100 if reference_total else 0.0
        subset_percent = (subset_count / subset_total) * 100 if subset_total else 0.0
        rows.append(
            {
                column: value,
                f"{reference_label}_count": reference_count,
                f"{reference_label}_percent": round(reference_percent, 2),
                f"{subset_label}_count": subset_count,
                f"{subset_label}_percent": round(subset_percent, 2),
                "subset_minus_reference_pp": round(subset_percent - reference_percent, 2),
            }
        )

    return pd.DataFrame(rows)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _existing_columns(metadata_df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column in metadata_df.columns]


def _display_series(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "Missing")


def create_directory_list(
    dataset_path: Path,
    num_folds: Optional[int] = None,
    check_valid_mask: bool = False,
    val_full_size: float = 15.0,
    val_fast_size: float = 5.0,
    seed: int = 42,
    num_outer_split_seeds: Optional[int] = None,
    num_inner_split_seeds: Optional[int] = None,
) -> Dict[str, Any]:
    """Create an ISLES26 dataset JSON description.

    Modes:
    - Fold mode (--num_folds): emit fold field only.
    - Split mode (default): emit split labels only (train/val_rest/val_fast).
    """
    training_root = resolve_training_root(dataset_path)
    discovered_cases = discover_cases(training_root)
    valid_cases = [case for case in discovered_cases if validate_case(case, check_valid_mask)]

    if num_folds is not None:
        fold_by_case_id = assign_round_robin_folds(valid_cases, num_folds)
        training_entries = [
            build_case_entry(
                case=case,
                training_root=training_root,
                fold=fold_by_case_id[case.case_id],
            )
            for case in valid_cases
        ]

        print(f"Resolved training root: {training_root}")
        print(f"Discovered cases: {len(discovered_cases)}")
        print(f"Valid cases: {len(valid_cases)}")
        print(f"Fold assignment: round-robin over {num_folds} folds")

        return {"training": training_entries}

    metadata_df = build_metadata_dataframe(valid_cases, training_root)
    split_df = select_nested_validation_split(
        metadata_df=metadata_df,
        val_full_percent=val_full_size,
        val_fast_percent=val_fast_size,
        seed=seed,
        num_outer_split_seeds=num_outer_split_seeds,
        num_inner_split_seeds=num_inner_split_seeds,
    )
    split_by_case_id = build_split_mapping(split_df)

    training_entries = [
        build_case_entry(
            case=case,
            training_root=training_root,
            split_label=split_by_case_id[case.case_id],
        )
        for case in valid_cases
    ]

    print(f"Resolved training root: {training_root}")
    print(f"Discovered cases: {len(discovered_cases)}")
    print(f"Valid cases: {len(valid_cases)}")
    print("Split assignment: train/val_rest/val_fast (no fold field in split mode)")
    print_nested_validation_report(split_df, val_full_size, val_fast_size)

    split_metadata = {
        "fast_validation_split_label": "val_fast",
        "rest_validation_split_label": "val_rest",
        "full_validation_group_label": "val_full",
        "full_validation_split_labels": list(FULL_VALIDATION_SPLITS),
        "full_validation_union_description": "val_full is defined as val_rest + val_fast.",
        "requested_val_full_percent": val_full_size,
        "requested_val_fast_percent": val_fast_size,
        "selected_outer_split_seed": split_df.attrs.get("outer_split_seed"),
        "selected_inner_split_seed": split_df.attrs.get("inner_split_seed"),
        "base_seed": split_df.attrs.get("base_seed"),
        "outer_candidates": split_df.attrs.get("num_outer_split_candidates", 1),
        "inner_candidates_per_outer": split_df.attrs.get("num_inner_split_candidates", 1),
        "singleton_site_policy": "Singleton sites are assigned to training only.",
        "singleton_training_sites": split_df.attrs.get("singleton_training_sites", []),
    }

    return {
        "validation_policy": split_metadata,
        "training": training_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an ISLES26 JSON dataset description.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset root or Training_Raw directory")
    parser.add_argument("output_file", type=Path, help="Path to the output JSON file")
    parser.add_argument(
        "--num_folds",
        type=int,
        help="Fold mode: emit fold assignments only (no split labels)",
    )
    parser.add_argument(
        "--val_full_size",
        type=float,
        default=15.0,
        help="Split mode: percentage assigned to val_full (= val_rest + val_fast)",
    )
    parser.add_argument(
        "--val_fast_size",
        type=float,
        default=5.0,
        help="Split mode: percentage assigned to val_fast (sampled from val_full = val_rest + val_fast)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split mode assignment")
    parser.add_argument(
        "--num_outer_split_seeds",
        type=int,
        help="Generate and score N outer split seeds from --seed, then keep the best nested split",
    )
    parser.add_argument(
        "--num_inner_split_seeds",
        type=int,
        help="For each outer split candidate, generate and score N inner val_fast seeds",
    )
    parser.add_argument("--check_valid_mask", action="store_true", help="Filter out cases with empty lesion masks")
    args = parser.parse_args()

    if args.num_folds is not None:
        if args.num_folds < 1:
            parser.error("--num_folds must be at least 1")
        split_mode_customized = (
            args.val_full_size != 15.0
            or args.val_fast_size != 5.0
            or args.num_outer_split_seeds is not None
            or args.num_inner_split_seeds is not None
        )
        if split_mode_customized:
            parser.error(
                "--num_folds cannot be combined with split-mode options "
                "(--val_full_size/--val_fast_size/--num_outer_split_seeds/--num_inner_split_seeds)."
            )
    else:
        if not 0 < args.val_full_size < 100:
            parser.error("--val_full_size must be between 0 and 100")
        if not 0 < args.val_fast_size < 100:
            parser.error("--val_fast_size must be between 0 and 100")
        if args.val_fast_size >= args.val_full_size:
            parser.error("--val_fast_size must be smaller than --val_full_size")
        if args.num_outer_split_seeds is not None and args.num_outer_split_seeds < 1:
            parser.error("--num_outer_split_seeds must be at least 1")
        if args.num_inner_split_seeds is not None and args.num_inner_split_seeds < 1:
            parser.error("--num_inner_split_seeds must be at least 1")

    directory_list = create_directory_list(
        dataset_path=args.dataset_path,
        num_folds=args.num_folds,
        check_valid_mask=args.check_valid_mask,
        val_full_size=args.val_full_size,
        val_fast_size=args.val_fast_size,
        seed=args.seed,
        num_outer_split_seeds=args.num_outer_split_seeds,
        num_inner_split_seeds=args.num_inner_split_seeds,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output_file:
        json.dump(directory_list, output_file, indent=4)

    print(f"Directory structure saved to {args.output_file}")


if __name__ == "__main__":
    main()
