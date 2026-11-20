"""Target-standardized fitted-value extension of the fast DID-INT estimator.

This module is deliberately separate from :mod:`didint`: it leaves the current
intercept-based estimator unchanged while retaining the nuisance coefficients
and observation-level target designs that its fast first stage normally drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import undid
    from .canonical_aggregation import aggregate_from_att_st
    from ._inference import InferenceError, strict_lstsq
except ImportError:
    import undid
    from canonical_aggregation import aggregate_from_att_st
    from _inference import InferenceError, strict_lstsq


Cell = Tuple[Any, float]


@dataclass
class FittedFirstStage:
    """Objects needed by both old and fitted DID-INT second stages."""

    panel: pd.DataFrame
    slopes: Dict[Any, np.ndarray]
    target_designs: Dict[Cell, np.ndarray]
    intercepts: Dict[Cell, float]
    outcome_means: Dict[Cell, float]
    covariates: List[str]
    ccc: str
    state_col: str
    time_col: str
    gvar_col: str
    pg_col: Optional[str]

    def slope(self, state: Any, time: float) -> np.ndarray:
        if self.ccc == "hom":
            key: Any = "hom"
        elif self.ccc == "state":
            key = state
        elif self.ccc == "time":
            key = float(time)
        else:
            key = (state, float(time))
        return self.slopes[key]

    def prediction_vector(
        self,
        source_state: Any,
        source_time: float,
        target_state: Any,
        target_time: float,
    ) -> np.ndarray:
        """Evaluate a source cell model on every actual target observation."""
        target_key = (target_state, float(target_time))
        source_key = (source_state, float(source_time))
        x_target = self.target_designs[target_key]
        beta = self.slope(source_state, source_time)
        intercept = self.intercepts[source_key]
        # Keep this as observation-level prediction even though a linear model
        # would permit replacing X by its column means.
        return intercept + x_target @ beta

    def prediction_mean(
        self,
        source_state: Any,
        source_time: float,
        target_state: Any,
        target_time: float,
    ) -> float:
        return float(
            self.prediction_vector(
                source_state, source_time, target_state, target_time
            ).mean()
        )


def _clean_first_stage_data(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str],
    covariates: List[str],
) -> pd.DataFrame:
    required = [outcome_col, state_col, time_col] + list(covariates)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep = [state_col, time_col, outcome_col]
    if gvar_col in df.columns:
        keep.append(gvar_col)
    if pg_col and pg_col in df.columns:
        keep.append(pg_col)
    keep.extend(covariates)
    work = df[keep].copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    for cov in covariates:
        work[cov] = pd.to_numeric(work[cov], errors="coerce")
    if work[[state_col, time_col, outcome_col] + covariates].isna().any().any():
        raise InferenceError('invalid_input', 'missing values are not silently dropped')
    if not np.isfinite(work[[time_col, outcome_col] + covariates].to_numpy(float)).all():
        raise InferenceError('invalid_input', 'input contains infinity')

    if gvar_col in work.columns:
        work[gvar_col] = pd.to_numeric(work[gvar_col], errors="coerce").fillna(0.0)
    else:
        work[gvar_col] = 0.0
    if pg_col:
        if pg_col in work.columns:
            work[pg_col] = work[pg_col].fillna("0").astype(str)
        else:
            work[pg_col] = "0"
    return work


def prepare_fitted_first_stage(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    ccc: str = "int",
) -> FittedFirstStage:
    """Fit the existing full-sample CCC first stage.

    ``hom``, ``state``, and ``time`` are solved after removing state-time
    intercepts within the relevant slope block. ``int`` is the existing fast
    independent state-time-cell fit. These are the same least-squares problems
    used by the current absorbed/sparse implementation.
    """
    if ccc not in {"hom", "state", "time", "int"}:
        raise ValueError("Fitted DID-INT currently supports ccc='hom', 'state', 'time', or 'int'.")
    covariates = list(covariates or [])
    work = _clean_first_stage_data(
        df,
        outcome_col=outcome_col,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
        covariates=covariates,
    )

    slopes: Dict[Any, np.ndarray] = {}
    target_designs: Dict[Cell, np.ndarray] = {}
    intercepts: Dict[Cell, float] = {}
    outcome_means: Dict[Cell, float] = {}
    records: List[Dict[str, Any]] = []
    k = len(covariates)

    if ccc == "hom":
        x_all = work[covariates].to_numpy(dtype=float) if k else np.zeros((len(work), 0))
        y_all = work[outcome_col].to_numpy(dtype=float)
        if k:
            cell_codes, _ = pd.factorize(
                pd.MultiIndex.from_arrays([work[state_col], work[time_col]]),
                sort=False,
            )
            counts = np.bincount(cell_codes).astype(float)
            y_means = np.bincount(cell_codes, weights=y_all) / counts
            y_tilde = y_all - y_means[cell_codes]
            x_tilde = np.empty_like(x_all)
            for j in range(k):
                x_means = np.bincount(cell_codes, weights=x_all[:, j]) / counts
                x_tilde[:, j] = x_all[:, j] - x_means[cell_codes]
            beta = strict_lstsq(x_tilde, y_tilde, absorbed_df=len(counts))
        else:
            beta = np.zeros(0, dtype=float)
        slopes["hom"] = beta

        for (state, time), block in work.groupby([state_col, time_col], sort=False):
            x = block[covariates].to_numpy(dtype=float) if k else np.zeros((len(block), 0))
            y = block[outcome_col].to_numpy(dtype=float)
            cell = (state, float(time))
            lam = float(y.mean() - x.mean(axis=0) @ beta) if k else float(y.mean())
            target_designs[cell] = x
            intercepts[cell] = lam
            outcome_means[cell] = float(y.mean())
            records.append(_panel_record(block, state, time, lam, state_col, time_col, gvar_col, pg_col))
    elif ccc == "state":
        for state, state_block in work.groupby(state_col, sort=False):
            x_state = state_block[covariates].to_numpy(dtype=float) if k else np.zeros((len(state_block), 0))
            y_state = state_block[outcome_col].to_numpy(dtype=float)
            if k:
                cell_codes, _ = pd.factorize(state_block[time_col], sort=False)
                counts = np.bincount(cell_codes).astype(float)
                y_means = np.bincount(cell_codes, weights=y_state) / counts
                y_tilde = y_state - y_means[cell_codes]
                x_tilde = np.empty_like(x_state)
                for j in range(k):
                    x_means = np.bincount(cell_codes, weights=x_state[:, j]) / counts
                    x_tilde[:, j] = x_state[:, j] - x_means[cell_codes]
                beta = strict_lstsq(x_tilde, y_tilde, absorbed_df=len(counts))
            else:
                beta = np.zeros(0, dtype=float)
            slopes[state] = beta

            for time, block in state_block.groupby(time_col, sort=False):
                x = block[covariates].to_numpy(dtype=float) if k else np.zeros((len(block), 0))
                y = block[outcome_col].to_numpy(dtype=float)
                cell = (state, float(time))
                lam = float(y.mean() - x.mean(axis=0) @ beta) if k else float(y.mean())
                target_designs[cell] = x
                intercepts[cell] = lam
                outcome_means[cell] = float(y.mean())
                records.append(_panel_record(block, state, time, lam, state_col, time_col, gvar_col, pg_col))
    elif ccc == "time":
        for time, time_block in work.groupby(time_col, sort=False):
            x_time = time_block[covariates].to_numpy(dtype=float) if k else np.zeros((len(time_block), 0))
            y_time = time_block[outcome_col].to_numpy(dtype=float)
            if k:
                cell_codes, _ = pd.factorize(time_block[state_col], sort=False)
                counts = np.bincount(cell_codes).astype(float)
                y_means = np.bincount(cell_codes, weights=y_time) / counts
                y_tilde = y_time - y_means[cell_codes]
                x_tilde = np.empty_like(x_time)
                for j in range(k):
                    x_means = np.bincount(cell_codes, weights=x_time[:, j]) / counts
                    x_tilde[:, j] = x_time[:, j] - x_means[cell_codes]
                beta = strict_lstsq(x_tilde, y_tilde, absorbed_df=len(counts))
            else:
                beta = np.zeros(0, dtype=float)
            slopes[float(time)] = beta

            for state, block in time_block.groupby(state_col, sort=False):
                x = block[covariates].to_numpy(dtype=float) if k else np.zeros((len(block), 0))
                y = block[outcome_col].to_numpy(dtype=float)
                cell = (state, float(time))
                lam = float(y.mean() - x.mean(axis=0) @ beta) if k else float(y.mean())
                target_designs[cell] = x
                intercepts[cell] = lam
                outcome_means[cell] = float(y.mean())
                records.append(_panel_record(block, state, time, lam, state_col, time_col, gvar_col, pg_col))
    else:
        for (state, time), block in work.groupby([state_col, time_col], sort=False):
            x = block[covariates].to_numpy(dtype=float) if k else np.zeros((len(block), 0))
            y = block[outcome_col].to_numpy(dtype=float)
            if k:
                x_mean = x.mean(axis=0)
                y_mean = float(y.mean())
                beta = strict_lstsq(x - x_mean, y - y_mean, absorbed_df=1)
                lam = float(y_mean - x_mean @ beta)
            else:
                beta = np.zeros(0, dtype=float)
                lam = float(y.mean())
            cell = (state, float(time))
            slopes[cell] = beta
            target_designs[cell] = x
            intercepts[cell] = lam
            outcome_means[cell] = float(y.mean())
            records.append(_panel_record(block, state, time, lam, state_col, time_col, gvar_col, pg_col))

    panel = pd.DataFrame.from_records(records)
    cols = [state_col, time_col, gvar_col, "lambda_st", "n_t"]
    if pg_col:
        cols.insert(3, pg_col)
    panel = panel[cols]
    return FittedFirstStage(
        panel=panel,
        slopes=slopes,
        target_designs=target_designs,
        intercepts=intercepts,
        outcome_means=outcome_means,
        covariates=covariates,
        ccc=ccc,
        state_col=state_col,
        time_col=time_col,
        gvar_col=gvar_col,
        pg_col=pg_col,
    )


def _panel_record(
    block: pd.DataFrame,
    state: Any,
    time: Any,
    lam: float,
    state_col: str,
    time_col: str,
    gvar_col: str,
    pg_col: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        state_col: state,
        time_col: float(time),
        gvar_col: float(block[gvar_col].iloc[0]),
        "lambda_st": lam,
        "n_t": int(len(block)),
    }
    if pg_col:
        row[pg_col] = str(block[pg_col].iloc[0])
    return row


def subset_first_stage(
    fit: FittedFirstStage,
    selected_states: List[Any],
    selected_times: List[float],
    g_map: Dict[Any, float],
) -> pd.DataFrame:
    out = fit.panel.loc[
        fit.panel[fit.state_col].isin(selected_states)
        & fit.panel[fit.time_col].isin(selected_times)
    ].copy()
    out[fit.gvar_col] = out[fit.state_col].map(g_map).fillna(0.0)
    return out


def _aggregate_targets(
    frame: pd.DataFrame,
    group_cols: List[str],
    weight_col: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["estimate", "se", "p_value", "weight"])
    records: List[Dict[str, Any]] = []
    grouped = frame.groupby(group_cols, sort=True, dropna=False) if group_cols else [((), frame)]
    for key, block in grouped:
        weights = block[weight_col].to_numpy(dtype=float)
        values = block["estimate"].to_numpy(dtype=float)
        est, se = undid._weighted_mean_and_se(values, weights)
        rec: Dict[str, Any] = {}
        if group_cols:
            keys = key if isinstance(key, tuple) else (key,)
            rec.update(dict(zip(group_cols, keys)))
        rec.update(
            estimate=float(est),
            se=float(se),
            p_value=np.nan,
            weight=float(block["weight"].sum()),
            n_treated=float(len(block)),
            n_control=np.nan,
        )
        records.append(rec)
    return pd.DataFrame.from_records(records)


def run_fitted_core(
    fit: FittedFirstStage,
    panel_df: pd.DataFrame,
    *,
    weighting: str = "both",
    not_yet_treated: bool = False,
) -> Dict[str, Any]:
    """Construct target-standardized ATT(s,t), then aggregate those targets."""
    state_col, time_col, gvar_col = fit.state_col, fit.time_col, fit.gvar_col
    times = sorted(float(x) for x in panel_df[time_col].unique())
    state_g = panel_df[[state_col, gvar_col]].drop_duplicates(state_col)
    g_map = state_g.set_index(state_col)[gvar_col]
    n_wide = panel_df.pivot(index=state_col, columns=time_col, values="n_t")
    cohorts = sorted(float(g) for g in g_map.unique() if np.isfinite(g) and g > 0)
    records: List[Dict[str, Any]] = []

    for g in cohorts:
        pre = [tt for tt in times if tt < g]
        if not pre:
            continue
        baseline = pre[-1]
        treated_states = [s for s in g_map.index if float(g_map[s]) == g]
        for target_time in [tt for tt in times if tt >= g]:
            controls = [
                s for s in g_map.index
                if float(g_map[s]) == 0.0
                or (not_yet_treated and float(g_map[s]) > target_time)
            ]
            for target_state in treated_states:
                target_key = (target_state, float(target_time))
                if target_key not in fit.target_designs:
                    continue
                # Every source model below uses this exact same ndarray object.
                common_target_design = fit.target_designs[target_key]
                comparison_rows: List[Dict[str, Any]] = []
                for source_state, treat in [(target_state, 1)] + [(c, 0) for c in controls]:
                    if (source_state, float(target_time)) not in fit.intercepts or (source_state, float(baseline)) not in fit.intercepts:
                        continue
                    assert fit.target_designs[target_key] is common_target_design
                    post = fit.prediction_mean(source_state, target_time, target_state, target_time)
                    assert fit.target_designs[target_key] is common_target_design
                    pre_mean = fit.prediction_mean(source_state, baseline, target_state, target_time)
                    n_post = float(n_wide.get(target_time, pd.Series(dtype=float)).get(source_state, np.nan))
                    n_pre = float(n_wide.get(baseline, pd.Series(dtype=float)).get(source_state, np.nan))
                    w_diff = n_post + n_pre if weighting in {"both", "diff"} else 1.0
                    w_att = n_post if weighting in {"both", "att"} else 1.0
                    comparison_rows.append(
                        {"delta": post - pre_mean, "treat": treat, "w_diff": w_diff, "w_att": w_att}
                    )
                stats = undid._weighted_diff_stats(pd.DataFrame.from_records(comparison_rows))
                if stats is None:
                    continue
                records.append(
                    {
                        "state": target_state,
                        "g": g,
                        "t": target_time,
                        **stats,
                        "w_diff_target": float(
                            n_wide.loc[target_state, target_time] + n_wide.loc[target_state, baseline]
                            if weighting in {"both", "diff"} else 1.0
                        ),
                        "target_design_id": id(common_target_design),
                    }
                )

    att_st = pd.DataFrame.from_records(records)
    if att_st.empty:
        empty = pd.DataFrame()
        return {"att": {"estimate": np.nan}, "att_g": empty, "att_gt": empty, "att_s": empty, "att_st": empty}

    canonical = aggregate_from_att_st(att_st, weighting=weighting, include_policy=False)
    att = canonical["att"]
    att_g = canonical["att_g"]
    att_gt = canonical["att_gt"]
    att_s = canonical["att_s"]
    att_st = canonical["att_st"]
    return {
        "att": att,
        "att_g": att_g,
        "att_gt": att_gt,
        "att_s": att_s,
        "att_st": att_st,
        "aggregation_weights": canonical.get("aggregation_weights", pd.DataFrame()),
        "aggregation_note": "ATT(s,t)-first canonical aggregation",
    }
