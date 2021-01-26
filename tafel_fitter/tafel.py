from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

R = 8.3145  # J/mol.K
F = 96485  # C/mol
T = 293  # K

def fit_all(x: np.array, y: np.array, windows: np.array = np.arange(0.01, 0.05, 0.001), R2_thresh: float = 0.9) -> pd.DataFrame:

    df = []
    for windowsize in windows:
        df.append(fit_windows(x, y, windowsize, n=1))

    df = pd.concat(df)

    df = df[(df["R2_tafel"] > R2_thresh) & (df["R2_lsv"] > R2_thresh)]

    return df


def fit_windows(potential: np.array, current: np.array, window: float, n: int = 1, scan_type: str = "cathodic") -> pd.DataFrame:
    """ L137 in ba3cc165515cc335578db76cf6fff4672afacb29 """

    window_samples = int(np.round(window / np.median(np.diff(potential))))
    log_current = np.log10(np.abs(current))

    # check for transport effects
    # numerically differentiate the LSV curve
    # find the max and truncate data...
    # note: do this outside the hot loop

    # fit on all intervals of size window
    results = []
    for idx in range(len(potential)-window_samples):
        if scan_type == "cathodic":
            mask = slice(-(idx+1)-window_samples, -(idx+1))
        elif scan_type == "anodic":
            mask = slice(idx, idx+window_samples)
        # mask = (potential >= x_start) & (potential < x_start + window)

        m_tafel, ic_tafel, r_tafel, *rest = stats.linregress(
            potential[mask], log_current[mask]
        )
        m_lsv, _, r_lsv, *rest = stats.linregress(
            potential[mask], current[mask]
        )

        results.append({
            "j0": 10**ic_tafel,
            "dj/dV": abs(m_lsv),
            "window_start": potential[idx],
            "window_min": potential[mask].min(),
            "window_max": potential[mask].max(),
            "R2_tafel": r_tafel**2,
            "R2_lsv": r_lsv**2,
            "dlog(j)/dV": 1000/m_tafel
        })

    r = pd.DataFrame(results)

    # now L423
    r["residue"] = np.abs(r["dj/dV"] - r["j0"] * (F / n * R * T))
    r["window"] = window

    return r




def filter_r2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    r_thresh = np.arange(0.9, 1.0, 0.001)
    for rr in r_thresh:
        sel = (df["R2_tafel"] > rr) & (df["R2_lsv"] > rr)
        for w, group in df[sel].groupby("window"):
            if group.size > 0:
                idx = group["residue"].argmin()
                rows.append(group.iloc[idx])

    return pd.DataFrame(rows)

def find_best_fit(df: pd.DataFrame, tafel_binsize: float = 1) -> tuple[pd.Series, pd.DataFrame]:

    # bin the tafel slopes
    tslope = df["dlog(j)/dV"]
    nbins = int(np.round(tslope.max() - tslope.min() / tafel_binsize))

    # find the tafel slope bin with the most unique values of `window`
    def count_windows(x: np.array) -> int:
        return np.unique(x).size

    nwindows, bins, counts = stats.binned_statistic(tslope, df["window"], statistic=count_windows, bins=nbins)

    id_bin = nwindows.argmax()
    left, right = bins[id_bin], bins[id_bin+1]

    subset = df[(tslope > left) & (tslope < right)]

    best_fit = subset.sort_values(by="R2_tafel").iloc[-1]
    return best_fit, subset
