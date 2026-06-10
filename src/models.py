from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Thresholds:
    baseline_mean: float
    baseline_std: float
    warning_threshold: float
    high_threshold: float
    very_high_threshold: float
    baseline_window: int


@dataclass(frozen=True)
class ChangePointResult:
    boundaries: list[int]
    bic_by_segments: dict[int, float]
    chosen_segments: int
    max_segments: int
    min_segment_size: int
    at_max_segments: bool
    bic_converged: bool


@dataclass(frozen=True)
class CrossingSummary:
    threshold_column: str
    value_column: str
    n_samples: int
    n_crossing_samples: int
    crossing_ratio: float
    first_crossing_time_s: float | None
    last_crossing_time_s: float | None
    longest_run_samples: int
    longest_run_duration_s: float
    longest_run_start_time_s: float | None
    longest_run_end_time_s: float | None


GMM_FEATURES = ["ma30_count_rate", "std30_count_rate", "diff_ma30_count_rate"]
IFOREST_FEATURES = ["count_rate", "ma30_count_rate", "std30_count_rate", "diff_ma30_count_rate"]


def compute_domain_thresholds(df: pd.DataFrame, baseline_window: int = 200) -> Thresholds:
    """Estimate exploratory count-level thresholds from an early baseline window."""
    ma = df["ma30_count_rate"].dropna()
    baseline = ma.iloc[:baseline_window] if len(ma) >= baseline_window else ma
    baseline_mean = float(baseline.mean())
    baseline_std = float(baseline.std(ddof=1))
    return Thresholds(
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        warning_threshold=baseline_mean + 1.96 * baseline_std,
        high_threshold=baseline_mean + 3.0 * baseline_std,
        very_high_threshold=baseline_mean + 4.0 * baseline_std,
        baseline_window=int(len(baseline)),
    )


def assign_threshold_states(df: pd.DataFrame, thresholds: Thresholds) -> pd.Series:
    """Assign exploratory count-level bands without mapping them to coal/gangue labels."""
    states = []
    values = df["ma30_count_rate"].fillna(df["ma15_count_rate"])
    for value in values:
        if value >= thresholds.very_high_threshold:
            states.append("very_high_count")
        elif value >= thresholds.high_threshold:
            states.append("high_count")
        elif value >= thresholds.warning_threshold:
            states.append("elevated_count")
        else:
            states.append("baseline_like_count")
    return pd.Series(states, index=df.index, name="threshold_state")


def estimate_background(
    series: pd.Series,
    method: str = "rolling_quantile",
    initial_seconds: float = 5.0,
    sample_interval_s: float = 0.1,
    window_seconds: float = 10.0,
    quantile: float = 0.2,
    alpha: float = 0.03,
) -> pd.Series:
    """Estimate slow-varying background count rate from a time series."""
    values = series.astype(float)
    if method == "initial":
        n_initial = max(1, int(round(initial_seconds / sample_interval_s)))
        baseline = float(values.iloc[:n_initial].median())
        return pd.Series(baseline, index=values.index, name="background_estimate")
    if method == "rolling_quantile":
        window = max(3, int(round(window_seconds / sample_interval_s)))
        background = values.rolling(window, min_periods=1, center=True).quantile(quantile)
        return background.bfill().ffill().rename("background_estimate")
    if method == "exponential":
        return values.ewm(alpha=alpha, adjust=False).mean().rename("background_estimate")
    raise ValueError(f"Unsupported background estimation method: {method}")


def separate_components(
    df: pd.DataFrame,
    value_column: str = "ma10_count_rate",
    method: str = "rolling_quantile",
    sample_interval_s: float = 0.1,
    window_seconds: float = 10.0,
    quantile: float = 0.2,
    initial_seconds: float = 5.0,
    alpha: float = 0.03,
) -> pd.DataFrame:
    """Split observed count rate into background, radiation contribution, and residual."""
    out = df.copy()
    background = estimate_background(
        out[value_column],
        method=method,
        initial_seconds=initial_seconds,
        sample_interval_s=sample_interval_s,
        window_seconds=window_seconds,
        quantile=quantile,
        alpha=alpha,
    )
    out["background_estimate"] = background
    out["radiation_component"] = out[value_column] - out["background_estimate"]
    out["noise_residual"] = out["count_rate"] - out[value_column]
    return out


def add_dynamic_threshold(
    df: pd.DataFrame,
    delta: float = 25.4,
    k_sigma: float | None = None,
    sigma_window_seconds: float = 5.0,
    sample_interval_s: float = 0.1,
    value_column: str = "ma10_count_rate",
) -> pd.DataFrame:
    """Add a background-relative threshold and crossing flag."""
    out = df.copy()
    if "background_estimate" not in out.columns:
        raise ValueError("background_estimate column is required before adding dynamic threshold")
    if k_sigma is None:
        out["dynamic_threshold"] = out["background_estimate"] + delta
    else:
        window = max(3, int(round(sigma_window_seconds / sample_interval_s)))
        sigma = (out[value_column] - out["background_estimate"]).rolling(window, min_periods=3).std(ddof=1)
        sigma = sigma.bfill().fillna(0.0)
        out["background_sigma"] = sigma
        out["dynamic_threshold"] = out["background_estimate"] + k_sigma * sigma
    out["above_dynamic_threshold"] = out[value_column] >= out["dynamic_threshold"]
    return out


def add_poisson_confidence_bands(
    df: pd.DataFrame,
    background_column: str = "background_estimate",
    value_column: str = "ma10_count_rate",
    effective_window_s: float = 1.0,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """Add one-sided Poisson upper confidence bands for background count-rate fluctuation."""
    if effective_window_s <= 0:
        raise ValueError("effective_window_s must be positive")
    out = df.copy()
    if background_column not in out.columns:
        raise ValueError(f"{background_column} column is required before adding Poisson bands")
    if value_column not in out.columns:
        raise ValueError(f"{value_column} column is required before adding Poisson bands")

    background_rate = out[background_column].astype(float).clip(lower=0.0)
    observed_rate = out[value_column].astype(float).clip(lower=0.0)
    expected_counts = background_rate * effective_window_s
    observed_counts = observed_rate * effective_window_s
    observed_count_ceiling = np.ceil(observed_counts).astype(int)

    out["poisson_effective_window_s"] = float(effective_window_s)
    out["poisson_expected_counts"] = expected_counts
    out["poisson_observed_counts"] = observed_counts
    out["poisson_sigma_rate"] = np.sqrt(background_rate / effective_window_s)
    out["poisson_upper_tail_p"] = poisson.sf(observed_count_ceiling - 1, expected_counts)

    for level in confidence_levels:
        suffix = f"{int(round(level * 100))}"
        upper_counts = poisson.ppf(level, expected_counts)
        upper_rate = upper_counts / effective_window_s
        out[f"poisson_upper_{suffix}"] = upper_rate
        out[f"poisson_excess_{suffix}"] = observed_rate - upper_rate
        out[f"above_poisson_{suffix}"] = observed_rate > upper_rate
    return out


def _longest_true_run(mask: pd.Series) -> tuple[int | None, int | None, int]:
    best_start: int | None = None
    best_end: int | None = None
    best_len = 0
    start: int | None = None
    values = mask.fillna(False).to_numpy(dtype=bool)
    for idx, flag in enumerate(values):
        if flag and start is None:
            start = idx
        if start is not None and ((not flag) or idx == len(values) - 1):
            end = idx - 1 if not flag else idx
            run_len = end - start + 1
            if run_len > best_len:
                best_start = start
                best_end = end
                best_len = run_len
            start = None
    return best_start, best_end, best_len


def summarize_threshold_crossing(
    df: pd.DataFrame,
    flag_column: str,
    value_column: str,
    sample_interval_s: float = 0.1,
) -> CrossingSummary:
    """Summarize threshold-crossing timing and longest continuous run."""
    mask = df[flag_column].fillna(False).astype(bool)
    crossing_indices = np.flatnonzero(mask.to_numpy())
    first_time = None
    last_time = None
    if len(crossing_indices) > 0:
        first_time = float(df["time_s"].iloc[int(crossing_indices[0])])
        last_time = float(df["time_s"].iloc[int(crossing_indices[-1])])
    run_start, run_end, run_len = _longest_true_run(mask)
    return CrossingSummary(
        threshold_column=flag_column,
        value_column=value_column,
        n_samples=int(len(df)),
        n_crossing_samples=int(mask.sum()),
        crossing_ratio=float(mask.mean()),
        first_crossing_time_s=first_time,
        last_crossing_time_s=last_time,
        longest_run_samples=int(run_len),
        longest_run_duration_s=float(run_len * sample_interval_s),
        longest_run_start_time_s=None if run_start is None else float(df["time_s"].iloc[run_start]),
        longest_run_end_time_s=None if run_end is None else float(df["time_s"].iloc[run_end]),
    )


def summarize_background_drift(
    df: pd.DataFrame,
    sample_interval_s: float = 0.1,
) -> dict[str, float]:
    """Summarize the estimated background level and its slow drift."""
    background = df["background_estimate"].astype(float)
    time = df["time_s"].astype(float)
    slope = 0.0
    if len(df) >= 2 and float(time.max()) > float(time.min()):
        slope = float(np.polyfit(time, background, deg=1)[0])
    return {
        "background_mean": float(background.mean()),
        "background_min": float(background.min()),
        "background_max": float(background.max()),
        "background_range": float(background.max() - background.min()),
        "background_slope_per_s": slope,
    }


def _scaled_model_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    model_df = df.dropna(subset=feature_cols).copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(model_df[feature_cols])
    return model_df, x


def select_gmm_components(
    df: pd.DataFrame,
    component_range: range = range(2, 9),
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate GMM component counts using AIC, BIC, and silhouette score."""
    feature_cols = feature_cols or GMM_FEATURES
    model_df, x = _scaled_model_matrix(df, feature_cols)
    rows = []
    for n_components in component_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
        labels = gmm.fit_predict(x)
        silhouette = float(silhouette_score(x, labels)) if len(set(labels)) > 1 else float("nan")
        rows.append(
            {
                "n_components": n_components,
                "n_samples": int(len(model_df)),
                "aic": float(gmm.aic(x)),
                "bic": float(gmm.bic(x)),
                "silhouette": silhouette,
            }
        )
    return pd.DataFrame(rows)


def fit_gmm_states(
    df: pd.DataFrame,
    n_states: int | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[pd.Series, float, pd.DataFrame]:
    """Fit exploratory GMM states; choose component count by BIC when unspecified."""
    feature_cols = feature_cols or GMM_FEATURES
    selection = select_gmm_components(df, feature_cols=feature_cols)
    if n_states is None:
        n_states = int(selection.loc[selection["bic"].idxmin(), "n_components"])

    model_df, x = _scaled_model_matrix(df, feature_cols)
    gmm = GaussianMixture(n_components=n_states, covariance_type="full", random_state=42)
    raw_labels = gmm.fit_predict(x)

    label_means = (
        pd.DataFrame({"raw": raw_labels, "ma": model_df["ma30_count_rate"].values})
        .groupby("raw")["ma"]
        .mean()
        .sort_values()
    )
    remap = {raw: ordered for ordered, raw in enumerate(label_means.index)}
    ordered_labels = np.array([remap[label] for label in raw_labels], dtype=int)
    states = pd.Series(index=df.index, dtype="object", name="gmm_state")
    states.loc[model_df.index] = [f"gmm_count_state_{label + 1}" for label in ordered_labels]
    score = float(silhouette_score(x, ordered_labels)) if len(set(ordered_labels)) > 1 else float("nan")
    return states, score, selection


def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.06) -> pd.Series:
    """Flag high-count anomalies as exploratory outliers, not validated gangue labels."""
    model_df, x = _scaled_model_matrix(df, IFOREST_FEATURES)
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(x)
    out = pd.Series(False, index=df.index, name="is_high_count_anomaly")
    high_count = model_df["ma30_count_rate"] >= model_df["ma30_count_rate"].quantile(0.75)
    out.loc[model_df.index] = (labels == -1) & high_count
    return out


def _segment_sse(prefix: np.ndarray, prefix_sq: np.ndarray, start: int, end: int) -> float:
    length = end - start
    if length <= 0:
        return float("inf")
    total = prefix[end] - prefix[start]
    total_sq = prefix_sq[end] - prefix_sq[start]
    return float(total_sq - total * total / length)


def _fit_piecewise_constant(values: np.ndarray, min_segment_size: int, max_segments: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    prefix = np.r_[0.0, np.cumsum(values)]
    prefix_sq = np.r_[0.0, np.cumsum(values**2)]
    cost = np.full((max_segments + 1, n + 1), np.inf)
    prev = np.full((max_segments + 1, n + 1), -1, dtype=int)

    for end in range(min_segment_size, n + 1):
        cost[1, end] = _segment_sse(prefix, prefix_sq, 0, end)

    for k in range(2, max_segments + 1):
        for end in range(k * min_segment_size, n + 1):
            starts = np.arange((k - 1) * min_segment_size, end - min_segment_size + 1)
            lengths = end - starts
            totals = prefix[end] - prefix[starts]
            totals_sq = prefix_sq[end] - prefix_sq[starts]
            segment_sse = totals_sq - totals * totals / lengths
            candidates = cost[k - 1, starts] + segment_sse
            best_idx = int(np.argmin(candidates))
            cost[k, end] = float(candidates[best_idx])
            prev[k, end] = int(starts[best_idx])
    return cost, prev


def detect_changepoints(
    series: pd.Series,
    min_segment_size: int = 40,
    max_segments: int = 15,
) -> ChangePointResult:
    """Piecewise-constant dynamic programming segmentation with corrected BIC."""
    clean = series.dropna().astype(float)
    values = clean.to_numpy()
    n = len(values)
    cost, prev = _fit_piecewise_constant(values, min_segment_size, max_segments)

    bic_by_segments: dict[int, float] = {}
    for k in range(2, max_segments + 1):
        segment_sse = max(cost[k, n], 1e-9)
        parameter_count = 2 * k - 1
        bic_by_segments[k] = float(n * np.log(segment_sse / n) + parameter_count * np.log(n))
    chosen_segments = min(bic_by_segments, key=bic_by_segments.get)
    at_max_segments = chosen_segments == max_segments
    bic_values = [bic_by_segments[k] for k in sorted(bic_by_segments)]
    bic_converged = not at_max_segments and any(
        later > earlier for earlier, later in zip(bic_values, bic_values[1:])
    )

    boundaries = [n]
    k = chosen_segments
    end = n
    while k > 1:
        start = int(prev[k, end])
        boundaries.append(start)
        end = start
        k -= 1
    boundaries.append(0)
    boundaries = sorted(boundaries)
    row_boundaries = [int(clean.index[pos]) for pos in boundaries[1:-1]]
    return ChangePointResult(
        boundaries=row_boundaries,
        bic_by_segments=bic_by_segments,
        chosen_segments=chosen_segments,
        max_segments=max_segments,
        min_segment_size=min_segment_size,
        at_max_segments=at_max_segments,
        bic_converged=bic_converged,
    )


def summarize_segments(df: pd.DataFrame, boundaries: list[int]) -> pd.DataFrame:
    cuts = [0] + boundaries + [len(df)]
    rows = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        seg = df.iloc[start:end]
        rows.append(
            {
                "segment": i + 1,
                "start_row": int(seg["row_number"].iloc[0]),
                "end_row": int(seg["row_number"].iloc[-1]),
                "start_sample": int(seg["sample_index"].iloc[0]),
                "end_sample": int(seg["sample_index"].iloc[-1]),
                "n_samples": int(len(seg)),
                "mean_count_rate": float(seg["count_rate"].mean()),
                "std_count_rate": float(seg["count_rate"].std(ddof=1)),
                "mean_ma30_count_rate": float(seg["ma30_count_rate"].mean()),
                "max_count_rate": float(seg["count_rate"].max()),
            }
        )
    return pd.DataFrame(rows)
