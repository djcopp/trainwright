"""Event log -> one row of features per entity (customers from transactions, users from sessions).

Only events strictly before the cutoff are used, so the features can be computed the same way
at prediction time without seeing the future. Per entity:

* counts: events, distinct days, events in the last 7/30/90/365 days
* time: days since the last and the first event, mean days between events
* numerics (and ``--expr`` columns such as ``revenue=Quantity*Price``): sum, mean, std, min, max,
  and the sum in each window
* categoricals: number of distinct values, share of the most common value
* the slope of activity: last-90-day events / previous-90-day events

Entities with no events before the cutoff get zero counts and NaN statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WINDOWS = (7, 30, 90, 365)


def _read(path: Path) -> pd.DataFrame:
    from .data.records import read_table

    return read_table(path)


def rollup(
    events: pd.DataFrame,
    key: str,
    time: str,
    cutoff: pd.Timestamp | None = None,
    exprs: list[str] | None = None,
    where: str | None = None,
    max_categoricals: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ev = events.copy()
    ev[time] = pd.to_datetime(ev[time], errors="coerce")
    ev = ev.dropna(subset=[key, time])
    for e in exprs or []:
        name, _, formula = e.partition("=")
        ev[name.strip()] = ev.eval(formula.strip())
    if where:
        ev = ev.query(where)
    cutoff = (
        pd.Timestamp(cutoff) if cutoff is not None else ev[time].max() + pd.Timedelta(seconds=1)
    )
    ev = ev[ev[time] < cutoff]
    ev["_days_ago"] = (cutoff - ev[time]).dt.total_seconds() / 86400.0
    g = ev.groupby(key, sort=False)
    out = pd.DataFrame(index=pd.Index(g.size().index, name=key))
    out["n_events"] = g.size()
    out["n_active_days"] = g[time].apply(lambda s: s.dt.normalize().nunique())
    out["days_since_last"] = g["_days_ago"].min()
    out["days_since_first"] = g["_days_ago"].max()
    out["mean_days_between"] = (out["days_since_first"] - out["days_since_last"]) / (
        out["n_active_days"] - 1
    ).where(out["n_active_days"] > 1)
    for w in WINDOWS:
        out[f"n_events_{w}d"] = ev[ev["_days_ago"] <= w].groupby(key).size()
    prev = ev[(ev["_days_ago"] > 90) & (ev["_days_ago"] <= 180)].groupby(key).size()
    out["trend_90d"] = out["n_events_90d"].fillna(0) / (prev.reindex(out.index).fillna(0) + 1)
    skip = {key, time, "_days_ago"}
    num = [c for c in ev.columns if c not in skip and pd.api.types.is_numeric_dtype(ev[c])]
    cat = [
        c
        for c in ev.columns
        if c not in skip and c not in num and 1 < ev[c].nunique() <= max(50, len(ev) // 20)
    ][:max_categoricals]
    for c in num:
        s = g[c]
        out[f"{c}_sum"] = s.sum()
        out[f"{c}_mean"] = s.mean()
        out[f"{c}_std"] = s.std()
        out[f"{c}_min"] = s.min()
        out[f"{c}_max"] = s.max()
        for w in (30, 90, 365):
            out[f"{c}_sum_{w}d"] = ev[ev["_days_ago"] <= w].groupby(key)[c].sum()
    for c in cat:
        out[f"{c}_n_unique"] = g[c].nunique()
        out[f"{c}_top_share"] = g[c].agg(
            lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan
        )
    counts = [c for c in out.columns if c.startswith("n_events")]
    out[counts] = out[counts].fillna(0)
    info = {
        "cutoff": str(cutoff),
        "events_used": int(len(ev)),
        "entities": int(len(out)),
        "numeric_columns": num,
        "categorical_columns": cat,
        "features": int(out.shape[1]),
    }
    return out.reset_index(), info


def rollup_files(
    events_path: Path,
    key: str,
    time: str,
    out_path: Path,
    entities_path: Path | None = None,
    entity_key: str | None = None,
    cutoff: str | None = None,
    exprs: list[str] | None = None,
    where: str | None = None,
) -> dict[str, Any]:
    feats, info = rollup(
        _read(events_path), key, time, pd.Timestamp(cutoff) if cutoff else None, exprs, where
    )
    if entities_path is not None:
        ent = _read(entities_path)
        ek = entity_key or key
        feats = feats.rename(columns={key: ek})
        # same dtype on both sides of the join (ids read as float in one file, int in another)
        feats[ek] = _norm_key(feats[ek])
        left = ent.assign(**{ek: _norm_key(ent[ek])})
        merged = left.merge(feats, on=ek, how="left")
        counts = [c for c in merged.columns if c.startswith("n_events")]
        merged[counts] = merged[counts].fillna(0)
        info["entities_without_events"] = int((merged["n_events"] == 0).sum())
        feats = merged
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix in (".parquet", ".pq"):
        feats.to_parquet(out_path, index=False)
    else:
        feats.to_csv(out_path, index=False)
    info.update(out=str(out_path), rows=int(len(feats)), columns=int(feats.shape[1]))
    return info


def _norm_key(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all() and np.all(np.mod(num, 1) == 0):
        return num.astype("int64").astype(str)
    return s.astype(str)
