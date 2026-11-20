"""
Pooled DiDint port for Python.

This module mirrors the Stata workflow structure:
1) First-stage residualization (optional covariates + ccc spec)
2) Second-stage ATT / ATTg / ATTgt estimation
3) Conventional, jackknife, and randomization inference

As of 2026-06-02, the first stage uses an absorbed state-time fixed-effect
projection that matches the verified `merit_example` benchmark much more
closely than the original plain least-squares-on-CCC-only implementation.
"""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .undid import run_undid
except ImportError:
    from undid import run_undid


VALID_CCC = {"hom", "state", "time", "int", "add"}
VALID_SLOPE_SAMPLE = {"full", "untreated_pre"}


def _to_numeric_or_raise(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        raise ValueError(f"Column '{name}' must be numeric for DiDint.")
    return out


def _build_covariate_design(
    work: pd.DataFrame,
    covariates: List[str],
    ccc: str,
    state_col: str,
    time_col: str,
) -> np.ndarray:
    if ccc not in VALID_CCC:
        raise ValueError(f"ccc must be one of {sorted(VALID_CCC)}.")

    cov_mat = work[covariates].to_numpy(dtype=float)
    n_obs, n_cov = cov_mat.shape
    if n_cov == 0:
        return np.zeros((n_obs, 0), dtype=float)

    if ccc == "hom":
        return cov_mat

    state_d = pd.get_dummies(work[state_col], drop_first=False, dtype=float).to_numpy()
    time_d = pd.get_dummies(work[time_col], drop_first=False, dtype=float).to_numpy()

    blocks: List[np.ndarray] = []
    if ccc == "state":
        for j in range(n_cov):
            blocks.append(state_d * cov_mat[:, [j]])
    elif ccc == "time":
        for j in range(n_cov):
            blocks.append(time_d * cov_mat[:, [j]])
    elif ccc == "int":
        state_time_d = np.einsum("ij,ik->ijk", state_d, time_d).reshape(n_obs, -1)
        for j in range(n_cov):
            blocks.append(state_time_d * cov_mat[:, [j]])
    elif ccc == "add":
        for j in range(n_cov):
            blocks.append(state_d * cov_mat[:, [j]])
            blocks.append(time_d * cov_mat[:, [j]])

    if not blocks:
        return np.zeros((n_obs, 0), dtype=float)
    return np.concatenate(blocks, axis=1)


def _state_time_codes(
    state_series: pd.Series,
    time_series: pd.Series,
) -> tuple[np.ndarray, int]:
    pairs = pd.MultiIndex.from_arrays([state_series.to_numpy(), time_series.to_numpy()])
    codes, uniques = pd.factorize(pairs, sort=False)
    return codes.astype(np.int64, copy=False), len(uniques)


def _demean_vector(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    sums = np.bincount(codes, weights=values, minlength=n_groups).astype(float)
    means = sums / counts
    return values - means[codes]


def _demean_matrix(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    out = np.empty_like(values, dtype=float)
    for j in range(values.shape[1]):
        sums = np.bincount(codes, weights=values[:, j], minlength=n_groups).astype(float)
        means = sums / counts
        out[:, j] = values[:, j] - means[codes]
    return out


def _prepare_panel_data_int_full_fast(
    work: pd.DataFrame,
    *,
    outcome_col: str,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str],
    covariates: List[str],
) -> pd.DataFrame:
    """
    Fast first stage for DID-INT with ccc="int" and slope_sample="full".

    With state-time-specific slopes and state-time intercepts, the least-squares
    problem is separable by state-time cell:

        y_ist = lambda_st + x_ist' beta_st + e_ist

    so we can estimate one small intercept-plus-slopes regression per cell and
    keep only the implied lambda_st and cell size n_t.
    """
    records: List[Dict[str, Any]] = []
    for (state_val, time_val), block in work.groupby([state_col, time_col], sort=False):
        y = block[outcome_col].to_numpy(dtype=float)
        n_obs = int(len(block))
        if covariates:
            x = block[covariates].to_numpy(dtype=float)
            y_mean = float(y.mean())
            x_mean = x.mean(axis=0)
            x_tilde = x - x_mean
            y_tilde = y - y_mean
            beta = np.linalg.lstsq(x_tilde, y_tilde, rcond=None)[0]
            lambda_st = float(y_mean - x_mean @ beta)
        else:
            lambda_st = float(y.mean())

        row: Dict[str, Any] = {
            state_col: state_val,
            time_col: float(time_val),
            gvar_col: float(block[gvar_col].iloc[0]) if gvar_col in block.columns and n_obs else 0.0,
            "lambda_st": lambda_st,
            "n_t": n_obs,
        }
        if pg_col:
            row[pg_col] = str(block[pg_col].iloc[0]) if pg_col in block.columns and n_obs else "0"
        records.append(row)

    panel = pd.DataFrame.from_records(records)
    out_cols = [state_col, time_col, gvar_col, "lambda_st", "n_t"]
    if pg_col:
        panel[pg_col] = panel[pg_col].fillna("0").astype(str)
        out_cols.insert(3, pg_col)
    return panel[out_cols]


def prepare_panel_data(
    df: pd.DataFrame,
    outcome_col: str,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    ccc: str = "int",
    slope_sample: str = "full",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    First-stage collapse for DiDint.

    If covariates are supplied, estimate CCC slopes after partialling out the
    full set of state-time intercepts via the within transformation implied by:

        y_ist = lambda_st + Z_ist beta + e_ist

    The collapsed state-time means of the residualized outcome are then used as
    the DID-INT `lambda_st` object passed to the second stage.
    """
    required = {outcome_col, state_col, time_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    covariates = covariates or []
    missing_cov = [c for c in covariates if c not in df.columns]
    if missing_cov:
        raise ValueError(f"Missing covariate columns: {missing_cov}")

    if ccc not in VALID_CCC:
        raise ValueError(f"ccc must be one of {sorted(VALID_CCC)}.")
    if slope_sample not in VALID_SLOPE_SAMPLE:
        raise ValueError(
            f"slope_sample must be one of {sorted(VALID_SLOPE_SAMPLE)}."
        )

    keep_cols = [state_col, time_col, outcome_col]
    if gvar_col in df.columns:
        keep_cols.append(gvar_col)
    if pg_col and pg_col in df.columns:
        keep_cols.append(pg_col)
    keep_cols.extend(covariates)

    work = df[keep_cols].copy()
    work = work.dropna(subset=[state_col, time_col, outcome_col])
    work[time_col] = _to_numeric_or_raise(work[time_col], time_col)

    if gvar_col in work.columns:
        work[gvar_col] = pd.to_numeric(work[gvar_col], errors="coerce").fillna(0.0)
    else:
        work[gvar_col] = 0.0
    if pg_col:
        if pg_col in work.columns:
            work[pg_col] = work[pg_col].fillna("0").astype(str)
        else:
            work[pg_col] = "0"

    for cov in covariates:
        work[cov] = pd.to_numeric(work[cov], errors="coerce")
    if covariates:
        work = work.dropna(subset=covariates)

    y = work[outcome_col].to_numpy(dtype=float)

    if ccc == "int" and slope_sample == "full":
        panel_fast = _prepare_panel_data_int_full_fast(
            work=work,
            outcome_col=outcome_col,
            state_col=state_col,
            time_col=time_col,
            gvar_col=gvar_col,
            pg_col=pg_col,
            covariates=covariates,
        )
        if cache_path:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            panel_fast.to_csv(cache_file, index=False)
        return panel_fast

    if covariates:
        x_all = _build_covariate_design(
            work=work,
            covariates=covariates,
            ccc=ccc,
            state_col=state_col,
            time_col=time_col,
        )

        if x_all.shape[1] == 0:
            y_adj = y
        else:
            if slope_sample == "untreated_pre":
                fit_mask = (work[gvar_col] == 0) | (work[time_col] < work[gvar_col])
                fit_mask = fit_mask.to_numpy(dtype=bool)
            else:
                fit_mask = np.ones(len(work), dtype=bool)

            if fit_mask.sum() <= x_all.shape[1]:
                fit_mask = np.ones(len(work), dtype=bool)

            fit_work = work.loc[fit_mask]
            fit_y = y[fit_mask]
            fit_x = x_all[fit_mask, :]
            fit_codes, n_fit_groups = _state_time_codes(
                fit_work[state_col],
                fit_work[time_col],
            )
            fit_y_tilde = _demean_vector(fit_y, fit_codes, n_fit_groups)
            fit_x_tilde = _demean_matrix(fit_x, fit_codes, n_fit_groups)

            beta = np.linalg.lstsq(fit_x_tilde, fit_y_tilde, rcond=None)[0]
            y_adj = y - x_all @ beta
    else:
        y_adj = y

    work["_lambda_source"] = y_adj

    panel = (
        work.groupby([state_col, time_col], as_index=False)
        .agg(lambda_st=("_lambda_source", "mean"), n_t=("_lambda_source", "size"))
        .reset_index(drop=True)
    )

    g_map = (
        work.groupby(state_col, as_index=False)[gvar_col]
        .agg(lambda x: float(x.iloc[0]) if len(x) else 0.0)
        .reset_index(drop=True)
    )

    panel = panel.merge(g_map, on=state_col, how="left")
    panel[gvar_col] = panel[gvar_col].fillna(0.0)
    out_cols = [state_col, time_col, gvar_col, "lambda_st", "n_t"]

    if pg_col:
        pg_map = (
            work.groupby(state_col, as_index=False)[pg_col]
            .agg(lambda x: str(x.iloc[0]) if len(x) else "0")
            .reset_index(drop=True)
        )
        panel = panel.merge(pg_map, on=state_col, how="left")
        panel[pg_col] = panel[pg_col].fillna("0").astype(str)
        out_cols.insert(3, pg_col)

    out = panel[out_cols]
    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_file, index=False)
    return out


def run_didint(
    panel_df: pd.DataFrame,
    *,
    state_col: str = "state",
    time_col: str = "time",
    gvar_col: str = "gvar",
    pg_col: Optional[str] = None,
    weighting: str = "both",
    not_yet_treated: bool = False,
    n_perm: int = 399,
    do_jackknife: bool = True,
    include_state_level_jackknife: bool = True,
    randomization_targets: str = "all",
    random_state: Optional[Any] = None,
    include_aggregation_weights: bool = True,
    include_legacy_aggregates: bool = True,
    jackknife_support: str = "changing",
) -> Dict[str, Any]:
    """
    Second-stage DiDint estimation on a prepared panel.
    """
    warnings.warn(
        'Legacy intercept-based DID-INT: with covariates this targets fitted values '
        'at X=0 and is not location/reference invariant. Use Design for '
        'target-standardized fitted DID-INT.', FutureWarning, stacklevel=2
    )
    out = run_undid(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        n_perm=n_perm,
        do_jackknife=do_jackknife,
        include_state_level_jackknife=include_state_level_jackknife,
        randomization_targets=randomization_targets,
        random_state=random_state,
        include_aggregation_weights=include_aggregation_weights,
        include_legacy_aggregates=include_legacy_aggregates,
        jackknife_support=jackknife_support,
    )
    out["method"] = "didint"
    out["estimator_variant"] = "legacy_intercept"
    out["target_standardized"] = False
    return out


def run_didint_from_raw(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    ccc: str = "int",
    slope_sample: str = "full",
    weighting: str = "both",
    not_yet_treated: bool = False,
    n_perm: int = 399,
    do_jackknife: bool = True,
    include_state_level_jackknife: bool = True,
    randomization_targets: str = "all",
    random_state: Optional[Any] = None,
    include_aggregation_weights: bool = True,
    include_legacy_aggregates: bool = True,
    jackknife_support: str = "changing",
) -> Dict[str, Any]:
    panel_df = prepare_panel_data(
        df=df,
        outcome_col=outcome_col,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
        covariates=covariates,
        ccc=ccc,
        slope_sample=slope_sample,
    )

    out = run_didint(
        panel_df=panel_df,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
        weighting=weighting,
        not_yet_treated=not_yet_treated,
        n_perm=n_perm,
        do_jackknife=do_jackknife,
        include_state_level_jackknife=include_state_level_jackknife,
        randomization_targets=randomization_targets,
        random_state=random_state,
        include_aggregation_weights=include_aggregation_weights,
        include_legacy_aggregates=include_legacy_aggregates,
        jackknife_support=jackknife_support,
    )
    out["ccc"] = ccc
    out["covariates"] = covariates or []
    out["slope_sample"] = slope_sample
    return out


