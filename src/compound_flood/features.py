"""
Feature engineering module for CFRI computation.

This module handles:
    - Percentile normalization of Q and WL
    - River-to-estuary lag estimation
    - Compound Flood Risk Index (CFRI) calculation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from loguru import logger

from .utils import compute_percentile_rank, robust_zscore
from .events import FloodEvent, get_event_q_wl_lags


def normalize_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame = None,
    method: str = 'percentile',
    robust: bool = False
) -> pd.DataFrame:
    """
    Normalize Q and WL_MHHW to [0, 1] scale.

    IMPORTANT: Uses train_df distribution to avoid data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Data to normalize
    train_df : pd.DataFrame, optional
        Training data for computing normalization parameters.
        If None, uses df itself (only for exploration, not final model).
    method : str
        'percentile' or 'zscore'
    robust : bool
        If True and method='zscore', use median/MAD

    Returns
    -------
    pd.DataFrame
        DataFrame with added Q_norm and WL_norm columns
    """
    df = df.copy()

    if train_df is None:
        logger.warning("No training data provided. Using self for normalization.")
        train_df = df

    for col, norm_col in [('Q', 'Q_norm'), ('WL_MHHW', 'WL_norm')]:
        if col not in df.columns:
            continue

        train_values = train_df[col].dropna().values
        values = df[col].values

        if method == 'percentile':
            df[norm_col] = compute_percentile_rank(values, train_values)
        elif method == 'zscore':
            if robust:
                z = robust_zscore(values, train_values)
            else:
                mean = np.nanmean(train_values)
                std = np.nanstd(train_values)
                z = (values - mean) / (std + 1e-10)
            # Convert to [0, 1] using sigmoid
            df[norm_col] = 1 / (1 + np.exp(-z))

    return df


def estimate_lag_xcorr(
    df: pd.DataFrame,
    lag_range: Tuple[int, int] = (0, 120),
    step: int = 1
) -> Tuple[int, float, np.ndarray]:
    """
    Estimate river-to-estuary lag using cross-correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q and WL_MHHW columns
    lag_range : tuple
        (min_lag, max_lag) in hours to search
    step : int
        Step size for lag search

    Returns
    -------
    tuple
        (optimal_lag, max_correlation, correlation_array)
    """
    q = df['Q'].values
    wl = df['WL_MHHW'].values

    # Handle NaN
    valid = ~np.isnan(q) & ~np.isnan(wl)
    if valid.sum() < 100:
        logger.warning("Insufficient valid data for cross-correlation")
        return 0, 0.0, np.array([])

    q_valid = q.copy()
    wl_valid = wl.copy()
    q_valid[~valid] = np.nanmean(q)
    wl_valid[~valid] = np.nanmean(wl)

    # Normalize
    q_norm = (q_valid - np.mean(q_valid)) / (np.std(q_valid) + 1e-10)
    wl_norm = (wl_valid - np.mean(wl_valid)) / (np.std(wl_valid) + 1e-10)

    # Compute cross-correlation for each lag
    lags = np.arange(lag_range[0], lag_range[1] + 1, step)
    correlations = []

    for lag in lags:
        if lag >= 0:
            # Q leads WL by lag hours
            q_shifted = q_norm[:-lag] if lag > 0 else q_norm
            wl_target = wl_norm[lag:] if lag > 0 else wl_norm
        else:
            # WL leads Q (unlikely for river-coast)
            q_shifted = q_norm[-lag:]
            wl_target = wl_norm[:lag]

        if len(q_shifted) > 0:
            corr = np.corrcoef(q_shifted, wl_target)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)

    correlations = np.array(correlations)

    # Find optimal lag
    valid_corr = ~np.isnan(correlations)
    if not valid_corr.any():
        return 0, 0.0, correlations

    best_idx = np.nanargmax(correlations)
    optimal_lag = lags[best_idx]
    max_corr = correlations[best_idx]

    logger.info(f"Cross-correlation lag: {optimal_lag} hours (r = {max_corr:.3f})")

    return int(optimal_lag), float(max_corr), correlations


def estimate_lag_events(
    events: List[FloodEvent],
    min_events: int = 10
) -> Tuple[Optional[float], List[float]]:
    """
    Estimate lag from event Q-WL peak timing.

    Parameters
    ----------
    events : list
        List of FloodEvent objects
    min_events : int
        Minimum events required for reliable estimate

    Returns
    -------
    tuple
        (median_lag, list_of_lags)
    """
    lags = get_event_q_wl_lags(events)

    if len(lags) < min_events:
        logger.warning(
            f"Only {len(lags)} events with Q peaks (need {min_events}). "
            "Event-based lag unreliable."
        )
        return None, lags

    median_lag = np.median(lags)
    logger.info(
        f"Event-based lag: {median_lag:.1f} hours "
        f"(IQR: {np.percentile(lags, 25):.1f} - {np.percentile(lags, 75):.1f})"
    )

    return median_lag, lags


def estimate_lag(
    df: pd.DataFrame,
    events: List[FloodEvent],
    config: Dict
) -> Dict:
    """
    Estimate optimal river-to-estuary lag using hybrid method.

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q and WL_MHHW
    events : list
        List of flood events
    config : dict
        Lag estimation configuration

    Returns
    -------
    dict
        Lag estimation results
    """
    lag_config = config.get('lag', {})
    lag_range = tuple(lag_config.get('search_range_hours', [0, 120]))
    step = lag_config.get('step_hours', 1)
    method = lag_config.get('method', 'hybrid')
    min_events = lag_config.get('min_events_for_event_lag', 10)

    results = {
        'method': method,
        'xcorr_lag': None,
        'xcorr_r': None,
        'event_lag': None,
        'event_lags': [],
        'optimal_lag': None
    }

    # Cross-correlation method
    xcorr_lag, xcorr_r, _ = estimate_lag_xcorr(df, lag_range, step)
    results['xcorr_lag'] = xcorr_lag
    results['xcorr_r'] = xcorr_r

    # Event-based method
    event_lag, event_lags = estimate_lag_events(events, min_events)
    results['event_lag'] = event_lag
    results['event_lags'] = event_lags

    # Choose optimal based on method
    if method == 'xcorr':
        results['optimal_lag'] = xcorr_lag
    elif method == 'event':
        results['optimal_lag'] = event_lag if event_lag else xcorr_lag
    else:  # hybrid
        if event_lag is not None:
            # Use event-based if available and consistent
            if abs(event_lag - xcorr_lag) < 24:
                results['optimal_lag'] = int(round(event_lag))
            else:
                # Prefer event-based
                results['optimal_lag'] = int(round(event_lag))
                logger.warning(
                    f"Event lag ({event_lag:.0f}h) differs from xcorr ({xcorr_lag}h)"
                )
        else:
            results['optimal_lag'] = xcorr_lag

    logger.info(f"Optimal lag selected: {results['optimal_lag']} hours")

    return results


def compute_lagged_features(
    df: pd.DataFrame,
    lag_hours: int
) -> pd.DataFrame:
    """
    Compute lagged river discharge feature.

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q_norm column
    lag_hours : int
        Lag in hours (positive = Q leads)

    Returns
    -------
    pd.DataFrame
        DataFrame with added Q_lag column
    """
    df = df.copy()

    if 'Q_norm' not in df.columns:
        raise ValueError("Must normalize features first (Q_norm missing)")

    # Shift Q forward by lag (Q at t-lag predicts WL at t)
    df['Q_lag'] = df['Q_norm'].shift(lag_hours)

    logger.debug(f"Created lagged Q feature with {lag_hours} hour lag")

    return df


def compute_overlap(
    df: pd.DataFrame,
    method: str = 'multiplicative',
    thresholds: Dict = None
) -> pd.DataFrame:
    """
    Compute compound overlap term.

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q_lag and WL_norm columns
    method : str
        'multiplicative' or 'threshold'
    thresholds : dict, optional
        For threshold method: river_percentile and coastal_percentile

    Returns
    -------
    pd.DataFrame
        DataFrame with added Overlap column
    """
    df = df.copy()

    if 'Q_lag' not in df.columns or 'WL_norm' not in df.columns:
        raise ValueError("Must have Q_lag and WL_norm columns")

    if method == 'multiplicative':
        # Continuous overlap: product of normalized values
        df['Overlap'] = df['Q_lag'] * df['WL_norm']

    elif method == 'threshold':
        # Binary overlap: both above threshold
        thresholds = thresholds or {}
        pQ = thresholds.get('river_percentile', 0.90)
        pWL = thresholds.get('coastal_percentile', 0.90)

        q_above = (df['Q_lag'] > pQ).astype(float)
        wl_above = (df['WL_norm'] > pWL).astype(float)
        df['Overlap'] = q_above * wl_above

    else:
        raise ValueError(f"Unknown overlap method: {method}")

    return df


def compute_cfri(
    df: pd.DataFrame,
    weights: Dict = None,
    normalize_output: bool = True
) -> pd.DataFrame:
    """
    Compute Compound Flood Risk Index (CFRI).

    CFRI(t) = (w1 * Q_lag(t) + w2 * WL_norm(t) + w3 * Overlap(t)) / (w1 + w2 + w3)

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q_lag, WL_norm, and Overlap columns
    weights : dict, optional
        Weights for each component. Default: equal weights (1/3 each)
    normalize_output : bool
        If True, normalize CFRI to [0, 1]

    Returns
    -------
    pd.DataFrame
        DataFrame with added CFRI column
    """
    df = df.copy()

    # Default weights
    if weights is None:
        weights = {'river': 1/3, 'coastal': 1/3, 'overlap': 1/3}

    w1 = weights.get('river', 1/3)
    w2 = weights.get('coastal', 1/3)
    w3 = weights.get('overlap', 1/3)
    w_sum = w1 + w2 + w3

    # Compute CFRI
    cfri = (
        w1 * df['Q_lag'].fillna(0) +
        w2 * df['WL_norm'].fillna(0) +
        w3 * df['Overlap'].fillna(0)
    ) / w_sum

    # Handle missing data
    missing = df['Q_lag'].isna() | df['WL_norm'].isna()
    cfri[missing] = np.nan

    if normalize_output:
        # Clip to [0, 1]
        cfri = cfri.clip(0, 1)

    df['CFRI'] = cfri

    # Store weights as attribute
    df.attrs['cfri_weights'] = {'w1': w1, 'w2': w2, 'w3': w3}

    logger.info(
        f"CFRI computed with weights: river={w1:.3f}, coastal={w2:.3f}, overlap={w3:.3f}"
    )

    return df


def compute_baseline_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute baseline indices for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Data with normalized features

    Returns
    -------
    pd.DataFrame
        DataFrame with baseline index columns
    """
    df = df.copy()

    # WL-only index (reactive baseline)
    if 'WL_norm' in df.columns:
        df['WL_only'] = df['WL_norm']

    # Q-only index
    if 'Q_lag' in df.columns:
        df['Q_only'] = df['Q_lag']

    # Persistence baseline (previous hour flood state)
    if 'FloodNow' in df.columns:
        df['Persistence'] = df['FloodNow'].shift(1)

    return df


def build_features_pipeline(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    lag_hours: int,
    config: Dict
) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Data to process
    train_df : pd.DataFrame
        Training data for normalization
    lag_hours : int
        River-to-estuary lag
    config : dict
        Feature configuration

    Returns
    -------
    pd.DataFrame
        DataFrame with all features
    """
    feat_config = config.get('features', {})
    cfri_config = feat_config.get('cfri', {})

    # Step 1: Normalize
    norm_method = feat_config.get('normalization', {}).get('method', 'percentile')
    robust = feat_config.get('normalization', {}).get('robust_zscore', False)

    df = normalize_features(
        df, train_df,
        method=norm_method,
        robust=robust
    )

    # Step 2: Compute lagged features
    df = compute_lagged_features(df, lag_hours)

    # Step 3: Compute overlap
    overlap_method = cfri_config.get('overlap_method', 'multiplicative')
    overlap_thresh = cfri_config.get('overlap_thresholds', {})

    df = compute_overlap(df, method=overlap_method, thresholds=overlap_thresh)

    # Step 4: Compute CFRI
    weights = cfri_config.get('default_weights', None)
    df = compute_cfri(df, weights=weights)

    # Step 5: Compute baselines
    df = compute_baseline_indices(df)

    return df


def fit_optimal_weights(
    df: pd.DataFrame,
    target_col: str = 'Y'
) -> Dict:
    """
    Fit optimal CFRI weights using logistic regression.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with features and target
    target_col : str
        Target column name

    Returns
    -------
    dict
        Fitted weights
    """
    from scipy.optimize import minimize

    # Prepare data
    features = df[['Q_lag', 'WL_norm', 'Overlap']].dropna()
    target = df.loc[features.index, target_col].dropna()

    # Align indices
    common_idx = features.index.intersection(target.index)
    X = features.loc[common_idx].values
    y = target.loc[common_idx].values

    if len(y) < 100:
        logger.warning("Insufficient data for weight optimization")
        return {'river': 1/3, 'coastal': 1/3, 'overlap': 1/3}

    def neg_log_likelihood(w):
        w = np.abs(w)  # Enforce non-negative
        w = w / w.sum()  # Normalize
        cfri = X @ w
        cfri = np.clip(cfri, 1e-10, 1 - 1e-10)
        ll = y * np.log(cfri) + (1 - y) * np.log(1 - cfri)
        return -ll.mean()

    # Optimize
    w0 = np.array([1/3, 1/3, 1/3])
    result = minimize(neg_log_likelihood, w0, method='Nelder-Mead')

    w_opt = np.abs(result.x)
    w_opt = w_opt / w_opt.sum()

    weights = {
        'river': float(w_opt[0]),
        'coastal': float(w_opt[1]),
        'overlap': float(w_opt[2])
    }

    logger.info(f"Optimized weights: {weights}")

    return weights
