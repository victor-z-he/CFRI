"""
Compound Flood Risk Index v2 (CFRI-v2)

Improved CFRI formulation that emphasizes the compound interaction effect.
The key insight: flood risk is amplified when BOTH river discharge AND
water level are elevated simultaneously.

Formula:
    CFRI_v2 = w1*Q_lag + w2*WL + w3*Compound_term

    where:
    - Compound_term = Q_lag × WL × I(both elevated)
    - I(both elevated) = 1 if Q_lag > Q_median AND WL > WL_median, else 0
    - Default weights: w1=0.15, w2=0.15, w3=0.70

This gives 70% weight to the compound interaction, emphasizing that
the combination of high river discharge AND high water level creates
disproportionately higher flood risk than either alone.

Author: NCSEF Student Researcher
Version: 2.0
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


def compute_cfri_v2(
    df: pd.DataFrame,
    weights: Dict[str, float] = None,
    q_threshold_pct: float = 50,
    wl_threshold_pct: float = 50,
    train_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute Compound Flood Risk Index v2 (compound-focused).

    The v2 formula emphasizes the compound interaction effect:
    CFRI = w1*Q_lag + w2*WL + w3*(Q_lag × WL × I_compound)

    where I_compound = 1 when BOTH Q and WL exceed their thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q_lag and WL_norm columns
    weights : dict, optional
        Weights for each component. Default: {river: 0.15, coastal: 0.15, compound: 0.70}
    q_threshold_pct : float
        Percentile threshold for Q to be considered "elevated" (default: 50)
    wl_threshold_pct : float
        Percentile threshold for WL to be considered "elevated" (default: 50)
    train_df : pd.DataFrame, optional
        Training data for computing thresholds (prevents data leakage)

    Returns
    -------
    pd.DataFrame
        DataFrame with CFRI_v2 and component columns added
    """
    df = df.copy()

    # Default weights emphasize compound interaction
    if weights is None:
        weights = {'river': 0.15, 'coastal': 0.15, 'compound': 0.70}

    w1 = weights.get('river', 0.15)
    w2 = weights.get('coastal', 0.15)
    w3 = weights.get('compound', 0.70)

    # Use training data for thresholds if provided (prevents data leakage)
    ref_df = train_df if train_df is not None else df

    # Compute thresholds
    Q_threshold = ref_df['Q_lag'].quantile(q_threshold_pct / 100)
    WL_threshold = ref_df['WL_norm'].quantile(wl_threshold_pct / 100)

    logger.info(f"CFRI-v2 thresholds: Q_lag > {Q_threshold:.3f} ({q_threshold_pct}th pct), "
                f"WL_norm > {WL_threshold:.3f} ({wl_threshold_pct}th pct)")

    # Identify when BOTH drivers are elevated
    Q_elevated = (df['Q_lag'] > Q_threshold).astype(float)
    WL_elevated = (df['WL_norm'] > WL_threshold).astype(float)
    Both_elevated = Q_elevated * WL_elevated

    # Store indicators
    df['Q_elevated'] = Q_elevated
    df['WL_elevated'] = WL_elevated
    df['Both_elevated'] = Both_elevated

    # Compute compound term (only active when both elevated)
    df['Compound_term'] = df['Q_lag'].fillna(0) * df['WL_norm'].fillna(0) * Both_elevated

    # Compute CFRI-v2
    cfri_v2 = (
        w1 * df['Q_lag'].fillna(0) +
        w2 * df['WL_norm'].fillna(0) +
        w3 * df['Compound_term'].fillna(0)
    )

    # Handle missing data
    missing = df['Q_lag'].isna() | df['WL_norm'].isna()
    cfri_v2[missing] = np.nan

    # Normalize to [0, 1]
    cfri_v2 = cfri_v2.clip(0, 1)

    df['CFRI_v2'] = cfri_v2

    # Store metadata
    df.attrs['cfri_v2_weights'] = {'w1': w1, 'w2': w2, 'w3': w3}
    df.attrs['cfri_v2_thresholds'] = {'Q': Q_threshold, 'WL': WL_threshold}

    # Log statistics
    compound_hours = Both_elevated.sum()
    total_hours = len(df)
    logger.info(f"CFRI-v2 computed: {compound_hours} compound hours "
                f"({compound_hours/total_hours*100:.1f}% of data)")

    return df


def find_optimal_threshold_v2(
    df: pd.DataFrame,
    target_col: str = 'Y',
    cfri_col: str = 'CFRI_v2',
    far_limit: float = 0.50,
    threshold_range: Tuple[float, float] = (0.05, 0.95),
    n_steps: int = 50
) -> Dict:
    """
    Find optimal CFRI-v2 threshold for flood prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Data with CFRI_v2 and target columns
    target_col : str
        Target variable column name
    cfri_col : str
        CFRI column name
    far_limit : float
        Maximum acceptable false alarm ratio
    threshold_range : tuple
        (min, max) threshold to search
    n_steps : int
        Number of threshold values to test

    Returns
    -------
    dict
        Optimal threshold and performance metrics
    """
    valid = ~(df[cfri_col].isna() | df[target_col].isna())
    cfri = df.loc[valid, cfri_col].values
    y_true = df.loc[valid, target_col].values

    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

    best_f1 = 0
    best_result = None

    results = []

    for T in thresholds:
        y_pred = (cfri > T).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0

        results.append({
            'threshold': T, 'precision': precision, 'recall': recall,
            'f1': f1, 'far': far, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })

        # Check if this is the best within FAR constraint
        if far <= far_limit and f1 > best_f1:
            best_f1 = f1
            best_result = results[-1].copy()

    # If no threshold meets FAR constraint, find best F1 overall
    if best_result is None:
        best_idx = np.argmax([r['f1'] for r in results])
        best_result = results[best_idx].copy()
        logger.warning(f"No threshold meets FAR <= {far_limit}. Using best F1.")

    best_result['all_results'] = pd.DataFrame(results)

    logger.info(f"Optimal CFRI-v2 threshold: {best_result['threshold']:.3f} "
                f"(F1={best_result['f1']:.3f}, FAR={best_result['far']:.3f})")

    return best_result


def compute_lead_times_v2(
    df: pd.DataFrame,
    events: list,
    cfri_threshold: float,
    cfri_col: str = 'CFRI_v2',
    max_lead_hours: int = 72
) -> Dict:
    """
    Compute lead times for CFRI-v2 warnings before flood events.

    Parameters
    ----------
    df : pd.DataFrame
        Data with CFRI-v2 column
    events : list
        List of FloodEvent objects
    cfri_threshold : float
        CFRI threshold for triggering warning
    cfri_col : str
        CFRI column name
    max_lead_hours : int
        Maximum lead time to consider

    Returns
    -------
    dict
        Lead time statistics and per-event results
    """
    lead_times = []
    event_results = []

    for event in events:
        # Look back from event start
        pre_start = event.start_time - pd.Timedelta(hours=max_lead_hours)
        pre_event = df.loc[pre_start:event.start_time, cfri_col]

        if len(pre_event) == 0:
            continue

        # Find first time CFRI exceeds threshold
        triggers = pre_event > cfri_threshold

        if triggers.any():
            first_trigger = triggers.idxmax()
            lead_hours = (event.start_time - first_trigger).total_seconds() / 3600

            if 0 < lead_hours <= max_lead_hours:
                lead_times.append(lead_hours)
                event_results.append({
                    'event_id': event.event_id,
                    'event_start': event.start_time,
                    'trigger_time': first_trigger,
                    'lead_hours': lead_hours,
                    'detected': True
                })
            else:
                event_results.append({
                    'event_id': event.event_id,
                    'event_start': event.start_time,
                    'trigger_time': None,
                    'lead_hours': None,
                    'detected': False
                })
        else:
            event_results.append({
                'event_id': event.event_id,
                'event_start': event.start_time,
                'trigger_time': None,
                'lead_hours': None,
                'detected': False
            })

    # Compute statistics
    if lead_times:
        stats = {
            'mean_lead_hours': np.mean(lead_times),
            'median_lead_hours': np.median(lead_times),
            'min_lead_hours': np.min(lead_times),
            'max_lead_hours': np.max(lead_times),
            'std_lead_hours': np.std(lead_times),
            'n_detected': len(lead_times),
            'n_total': len(events),
            'detection_rate': len(lead_times) / len(events) if events else 0
        }
    else:
        stats = {
            'mean_lead_hours': None,
            'median_lead_hours': None,
            'min_lead_hours': None,
            'max_lead_hours': None,
            'std_lead_hours': None,
            'n_detected': 0,
            'n_total': len(events),
            'detection_rate': 0
        }

    return {
        'statistics': stats,
        'lead_times': lead_times,
        'event_results': pd.DataFrame(event_results)
    }
