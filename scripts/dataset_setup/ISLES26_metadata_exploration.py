import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__:
    from .ISLES26_json_creator import discover_cases, resolve_training_root, validate_case
else:
    from ISLES26_json_creator import discover_cases, resolve_training_root, validate_case


DAYS_POST_STROKE_COLUMN = "DAYS_POST_STROKE"
CATEGORICAL_COLUMNS = ("ATLAS2_DATASET", "CHRONICITY", "SITE")
VALIDATION_BALANCE_COLUMNS = ("DAYS_POST_STROKE_BIN", "CHRONICITY", "ATLAS2_DATASET")
FULL_VALIDATION_SPLITS = ("val_rest", "val_fast")
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


def load_metadata_data(dataset_path: Path, check_valid_mask: bool = False) -> pd.DataFrame:
    """Load all ISLES26 metadata CSVs into one case-level dataframe."""
    training_root = resolve_training_root(dataset_path)
    discovered_cases = discover_cases(training_root)
    valid_cases = [case for case in discovered_cases if validate_case(case, check_valid_mask)]

    rows = []
    for case in valid_cases:
        try:
            metadata = pd.read_csv(case.metadata_path).iloc[0].to_dict()
        except Exception as exc:
            print(f"Skipping {case.case_id}: could not read metadata CSV ({exc})")
            continue

        metadata["caseID"] = case.case_id
        metadata["siteID"] = case.site_id
        metadata["metadata_csv"] = case.metadata_path.relative_to(training_root).as_posix()
        rows.append(metadata)

    if not rows:
        raise ValueError("No ISLES26 metadata CSV files were successfully loaded.")

    metadata_df = pd.DataFrame(rows)
    metadata_df = normalize_metadata(metadata_df)

    print(f"Resolved training root: {training_root}")
    print(f"Discovered cases: {len(discovered_cases)}")
    print(f"Cases with complete required files: {len(valid_cases)}")
    print(f"Metadata rows loaded: {len(metadata_df)}")

    return metadata_df


def normalize_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and add split-planning helper columns."""
    metadata_df = metadata_df.copy()

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

    for column in CATEGORICAL_COLUMNS:
        if column in metadata_df.columns:
            metadata_df[column] = metadata_df[column].replace("", pd.NA)

    return metadata_df


def print_metadata_report(
    metadata_df: pd.DataFrame,
    num_folds: Optional[int] = None,
    val_full_size: Optional[float] = 15.0,
    val_fast_size: Optional[float] = 5.0,
    seed: int = 42,
    num_outer_split_seeds: Optional[int] = None,
    num_inner_split_seeds: Optional[int] = None,
) -> pd.DataFrame:
    """Print descriptive metadata statistics, with optional fold-oriented hints."""
    print_header("Dataset Overview")
    print(f"Total cases: {len(metadata_df)}")
    print(f"Columns: {', '.join(metadata_df.columns)}")
    print(f"Unique case IDs: {metadata_df['caseID'].nunique()}")
    print(f"Unique session IDs: {_safe_nunique(metadata_df, 'SESSION_ID')}")
    print(f"Unique sites: {_safe_nunique(metadata_df, 'SITE')}")

    print_header("Missing Values")
    missing_summary = build_missing_summary(metadata_df)
    if missing_summary.empty:
        print("No missing values detected.")
    else:
        print(missing_summary.to_string())

    print_header("Categorical Distributions")
    for column in _existing_columns(metadata_df, ["ATLAS2_DATASET", "SITE", "siteID", "CHRONICITY"]):
        print_value_counts(metadata_df, column)

    print_header("Days Post Stroke")
    if DAYS_POST_STROKE_COLUMN in metadata_df.columns:
        print_numeric_summary(metadata_df[DAYS_POST_STROKE_COLUMN], DAYS_POST_STROKE_COLUMN)
        print_value_counts(metadata_df, "DAYS_POST_STROKE_BIN")
    else:
        print(f"{DAYS_POST_STROKE_COLUMN} column not found.")

    print_header("Cross Tabulations")
    print_cross_tab(metadata_df, "SITE", "DAYS_POST_STROKE_BIN")
    print_cross_tab(metadata_df, "siteID", "DAYS_POST_STROKE_BIN")
    print_cross_tab(metadata_df, "SITE", "CHRONICITY")

    if num_folds is not None:
        print_header("Fold Planning Hints")
        print_fold_planning_hints(metadata_df, num_folds)

    if val_full_size is not None and val_fast_size is not None:
        metadata_df = select_nested_validation_split(
            metadata_df=metadata_df,
            val_full_percent=val_full_size,
            val_fast_percent=val_fast_size,
            seed=seed,
            num_outer_split_seeds=num_outer_split_seeds,
            num_inner_split_seeds=num_inner_split_seeds,
        )
        print_nested_validation_report(metadata_df, val_full_size, val_fast_size)

    return metadata_df


def build_missing_summary(metadata_df: pd.DataFrame) -> pd.DataFrame:
    missing_count = metadata_df.isna().sum()
    missing_percent = (missing_count / len(metadata_df)) * 100
    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent.round(2),
        }
    )
    return summary[summary["missing_count"] > 0].sort_values(
        by=["missing_percent", "missing_count"],
        ascending=False,
    )


def print_numeric_summary(series: pd.Series, label: str) -> None:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    print(f"\n{label}:")
    print(f"  valid_count: {len(numeric_series)}")
    print(f"  missing_count: {series.isna().sum()}")

    if numeric_series.empty:
        print("  No valid numeric values.")
        return

    summary = numeric_series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    for stat_name, value in summary.items():
        print(f"  {stat_name:>8}: {value:.2f}")


def print_value_counts(metadata_df: pd.DataFrame, column: str) -> None:
    if column not in metadata_df.columns:
        return

    display_values = _display_series(metadata_df[column])
    counts = display_values.value_counts(dropna=False)
    percentages = display_values.value_counts(dropna=False, normalize=True) * 100
    summary = pd.DataFrame(
        {
            "count": counts,
            "percent": percentages.round(2),
        }
    )

    print(f"\n{column}:")
    print(summary.to_string())


def print_cross_tab(metadata_df: pd.DataFrame, row_column: str, column_column: str) -> None:
    if row_column not in metadata_df.columns or column_column not in metadata_df.columns:
        print(f"\nSkipping {row_column} x {column_column}: required columns not found.")
        return

    print(f"\n{row_column} x {column_column}:")
    crosstab = pd.crosstab(
        _display_series(metadata_df[row_column]),
        _display_series(metadata_df[column_column]),
        dropna=False,
    )
    print(crosstab.to_string())


def print_fold_planning_hints(metadata_df: pd.DataFrame, num_folds: int) -> None:
    print(f"Requested folds: {num_folds}")
    print(f"Ideal cases per fold: {len(metadata_df) / num_folds:.2f}")

    for column in _existing_columns(metadata_df, ["SITE", "siteID", "DAYS_POST_STROKE_BIN", "CHRONICITY"]):
        counts = _display_series(metadata_df[column]).value_counts(dropna=False)
        scarce_groups = counts[counts < num_folds]

        print(f"\n{column}:")
        print(f"  groups: {len(counts)}")
        print(f"  largest_group: {counts.index[0]} ({counts.iloc[0]} cases)")
        if scarce_groups.empty:
            print("  every group has at least one case per fold in principle")
        else:
            print("  groups with fewer cases than requested folds:")
            for value, count in scarce_groups.items():
                print(f"    {value}: {count}")

    print("\nInterpretation:")
    print("- `SITE`/`siteID` are likely important because scanner/site effects can leak into model behavior.")
    print("- `DAYS_POST_STROKE_BIN` is the clearest continuous clinical axis available in this metadata.")
    print("- `CHRONICITY` is normalized as 0/1, with missing or blank values treated as 0.")
    print("- Small site groups may need grouped, site-aware, or repeated split strategies instead of strict stratification.")


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
    """Assign train vs validation_pool with site coverage in both sets."""
    if "SITE" not in metadata_df.columns:
        raise ValueError("SITE column is required for site-balanced validation assignment.")

    split_df = metadata_df.copy()
    total_cases = len(split_df)
    site_counts = split_df["SITE"].value_counts().sort_index()

    min_val_count = len(site_counts)
    max_val_count = total_cases - len(site_counts)
    if target_val_full_count < min_val_count:
        raise ValueError(
            f"Target validation size {target_val_full_count} is too small; at least {min_val_count} "
            "cases are needed to place every site in validation."
        )
    if target_val_full_count > max_val_count:
        raise ValueError(
            f"Target validation size {target_val_full_count} is too large; at most {max_val_count} "
            "cases are allowed while keeping every site in training."
        )
    if (site_counts < 2).any():
        sparse_sites = site_counts[site_counts < 2].index.tolist()
        raise ValueError(
            "Cannot keep each site in both train and validation for these sites: "
            f"{sparse_sites}"
        )

    site_quotas = allocate_group_quotas(
        counts=site_counts,
        target=target_val_full_count,
        min_per_group=1,
        max_per_group=site_counts - 1,
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

    print(
        "Assignment policy: outer split keeps one case per site in both train and full validation; "
        "inner split uses proportional site quotas without strict one-case-per-site requirements."
    )

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


def _safe_nunique(metadata_df: pd.DataFrame, column: str) -> str:
    if column not in metadata_df.columns:
        return "column not found"
    return str(metadata_df[column].nunique(dropna=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore combined ISLES26 metadata with nested val_rest/val_fast split analysis."
    )
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset root or Training_Raw directory")
    parser.add_argument("--fold", type=int, help="Optional number of folds to consider for split planning")
    parser.add_argument(
        "--val_full_size",
        type=float,
        default=15.0,
        help="Percentage assigned to full validation group val_full (= val_rest + val_fast)",
    )
    parser.add_argument(
        "--val_fast_size",
        type=float,
        default=5.0,
        help="Percentage assigned to val_fast (sampled from val_full = val_rest + val_fast)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for nested validation assignment")
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
    parser.add_argument("--check_valid_mask", action="store_true", help="Exclude cases with empty lesion masks")
    parser.add_argument("--output", type=Path, help="Optional path to save the combined metadata CSV")
    args = parser.parse_args()

    if args.fold is not None and args.fold < 1:
        parser.error("--fold must be at least 1")
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

    metadata_df = load_metadata_data(
        dataset_path=args.dataset_path,
        check_valid_mask=args.check_valid_mask,
    )
    metadata_df = print_metadata_report(
        metadata_df,
        num_folds=args.fold,
        val_full_size=args.val_full_size,
        val_fast_size=args.val_fast_size,
        seed=args.seed,
        num_outer_split_seeds=args.num_outer_split_seeds,
        num_inner_split_seeds=args.num_inner_split_seeds,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(args.output, index=False)
        print(f"\nCombined metadata saved to {args.output}")


if __name__ == "__main__":
    main()
