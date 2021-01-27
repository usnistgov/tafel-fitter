from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

R = 8.3145  # J/mol.K
F = 96485  # C/mol
T = 293  # K


def fit_all(
    x: np.array,
    y: np.array,
    windows: np.array = np.arange(0.01, 0.05, 0.001),
    R2_thresh: float = 0.9,
) -> pd.DataFrame:
    """ fit tafel model on all sub-windows for each window size in `windows` """

    df = []
    for windowsize in windows:
        df.append(fit_windows(x, y, windowsize, n=1))

    df = pd.concat(df)

    df = df[(df["R2_tafel"] > R2_thresh) & (df["R2_lsv"] > R2_thresh)]

    return df


def fit_windows(
    potential: np.array,
    current: np.array,
    window: float,
    n: int = 1,
    scan_type: str = "cathodic",
) -> pd.DataFrame:
    """fit a tafel model on each sub-window of size `window`
    L137 in ba3cc165515cc335578db76cf6fff4672afacb29

    """

    # explicit loop is over LSV samples, not raw potential values
    window_samples = int(np.round(window / np.median(np.diff(potential))))
    log_current = np.log10(np.abs(current))

    # check for transport effects
    # numerically differentiate the LSV curve
    # find the max and truncate data...
    # note: do this outside the hot loop

    # fit on all intervals of size `window`
    results = []
    for idx in range(len(potential) - window_samples):

        if scan_type == "cathodic":
            # cathodic scans run "backwards" in time
            # start at open circuit and stride backwards towards more negative potential
            mask = slice(-(idx + 1) - window_samples, -(idx + 1))
        elif scan_type == "anodic":
            # anodic scans run forwards/intuitively
            # start at open circuit and stride forwards towards more positive potential
            mask = slice(idx, idx + window_samples)

        # fit Tafel data
        slope_tafel, intercept_tafel, r_tafel, *rest = stats.linregress(
            potential[mask], log_current[mask]
        )

        # fit LSV data
        slope_lsv, _, r_lsv, *rest = stats.linregress(potential[mask], current[mask])

        results.append(
            {
                "j0": 10 ** intercept_tafel,  # exchange current
                "dj/dV": abs(slope_lsv),  # LSV slope ~ j0 in the Tafel regime
                "dlog(j)/dV": 1000 / slope_tafel,  # tafel slope (mV/decade)
                "window_start": potential[idx],
                "window_min": potential[mask].min(),
                "window_max": potential[mask].max(),
                "R2_tafel": r_tafel ** 2,
                "R2_lsv": r_lsv ** 2,
            }
        )

    r = pd.DataFrame(results)

    # Tafel residue -- quantify the quality of the Tafel/linearity assumption
    r["residue"] = np.abs(r["dj/dV"] - r["j0"] * (F / n * R * T))

    # save the fitting window size for downstream analysis
    r["window"] = window

    return r


def filter_r2(
    df: pd.DataFrame, r2_threshold: np.array = np.arange(0.9, 1.0, 0.001)
) -> pd.DataFrame:
    """ record minimal-tafel-residue fits as a function of R^2 threshold """
    rows = []

    for threshold in r2_threshold:
        sel = (df["R2_tafel"] > threshold) & (df["R2_lsv"] > threshold)

        # record the fit with minimal tafel residue for each fitting window size
        for w, group in df[sel].groupby("window"):
            if group.size > 0:
                idx = group["residue"].argmin()
                rows.append(group.iloc[idx])

    return pd.DataFrame(rows)


def find_best_fit(
    df: pd.DataFrame, tafel_binsize: float = 1
) -> tuple[pd.Series, pd.DataFrame]:
    """select the most stable Tafel fit

    1. discretize the tafel fits by tafel slope
    2. select tafel slope bin with greatest range of fitting window size
    3. select from this tafel slope bin the fit with the best R^2 value
    """

    # bin the tafel slopes
    tslope = df["dlog(j)/dV"]
    nbins = int(np.round(tslope.max() - tslope.min() / tafel_binsize))

    def count_unique_values(x: np.array) -> int:
        return np.unique(x).size

    # nwindows: number of unique window sizes in each tafel slope bin
    # bins: tafel bin edges
    nwindows, bins, counts = stats.binned_statistic(
        tslope, df["window"], statistic=count_unique_values, bins=nbins
    )

    # select the tafel slope bin with the most unique window size values
    # then select all the fits in this tafel slope bin
    id_bin = nwindows.argmax()
    bin_min, bin_max = bins[id_bin], bins[id_bin + 1]
    subset = df[(tslope > bin_min) & (tslope < bin_max)]

    # the "best" fit has the highest Tafel R^2 value in this tafel slope bin
    best_fit = subset.sort_values(by="R2_tafel").iloc[-1]

    return best_fit, subset
