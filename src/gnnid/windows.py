"""Sliding event-time windows over a run's events table."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Window:
    run_id: str
    w_idx: int
    t0: float
    t1: float
    events: pd.DataFrame   # events with t0 <= ts < t1


def iter_windows(events: pd.DataFrame, width_s: float, stride_s: float,
                 min_tail_s: float, run_id: str):
    """Yield fixed-width sliding windows. The last (partial) window is kept
    iff it spans >= min_tail_s. Events must be ts-sorted."""
    if events.empty:
        return
    ts = events["ts"].to_numpy()
    start, end = float(ts[0]), float(ts[-1])
    w_idx = 0
    t0 = start
    while t0 <= end:
        t1 = t0 + width_s
        span_end = min(t1, end)
        # A trailing partial window is dropped only if it is too short AND we
        # already emitted a full window; a run shorter than min_tail still
        # yields its single window.
        if t1 > end and w_idx > 0 and (span_end - t0) < min_tail_s:
            break
        mask = (ts >= t0) & (ts < t1)
        if mask.any():
            yield Window(run_id, w_idx, t0, t1,
                         events.iloc[mask.nonzero()[0]].reset_index(drop=True))
            w_idx += 1
        if t1 > end:
            break
        t0 += stride_s
