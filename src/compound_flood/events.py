"""
Flood event detection module.

This module handles:
    - Instantaneous flood state detection
    - Flood event identification and characterization
    - Predictive label generation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FloodEvent:
    """Container for flood event information."""
    event_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: float
    peak_time: pd.Timestamp
    peak_wl: float
    peak_exceedance: float  # Above threshold
    pre_event_wl: float  # Baseline before event
    peak_q: Optional[float] = None
    q_peak_time: Optional[pd.Timestamp] = None


def detect_flood_state(
    df: pd.DataFrame,
    threshold: float,
    wl_col: str = 'WL_MHHW'
) -> pd.DataFrame:
    """
    Detect instantaneous flood state.

    Parameters
    ----------
    df : pd.DataFrame
        Data with water level column
    threshold : float
        Flood threshold (meters above MHHW)
    wl_col : str
        Water level column name

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'FloodNow' column
    """
    df = df.copy()

    # Instantaneous flood state
    df['FloodNow'] = (df[wl_col] > threshold).astype(int)

    # Handle NaN - flood state undefined when WL is missing
    df.loc[df[wl_col].isna(), 'FloodNow'] = np.nan

    n_flood = (df['FloodNow'] == 1).sum()
    pct_flood = 100 * n_flood / df['FloodNow'].notna().sum()

    logger.info(f"Flood state: {n_flood:,} hours ({pct_flood:.2f}%) above threshold {threshold:.2f}m")

    return df


def create_predictive_labels(
    df: pd.DataFrame,
    lookahead_hours: int = 24,
    wl_col: str = 'WL_MHHW',
    threshold: float = None
) -> pd.DataFrame:
    """
    Create predictive flood labels for model training.

    Y(t) = 1 if max WL in [t, t+L] > threshold

    Parameters
    ----------
    df : pd.DataFrame
        Data with water level and FloodNow columns
    lookahead_hours : int
        Prediction window in hours
    wl_col : str
        Water level column name
    threshold : float, optional
        Flood threshold. If None, uses existing FloodNow column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Y' (predictive label) column
    """
    df = df.copy()

    if threshold is not None:
        flood_state = (df[wl_col] > threshold).astype(float)
    elif 'FloodNow' in df.columns:
        flood_state = df['FloodNow']
    else:
        raise ValueError("Must provide threshold or have FloodNow column")

    # Rolling maximum over lookahead window
    # Use forward-looking window: [t, t+L]
    y = flood_state.rolling(
        window=lookahead_hours,
        min_periods=1
    ).max().shift(-lookahead_hours + 1)

    # Y(t) = 1 if any flooding in next L hours
    df['Y'] = y

    # Mark end of series where lookahead is incomplete
    df.loc[df.index[-lookahead_hours:], 'Y'] = np.nan

    n_positive = (df['Y'] == 1).sum()
    pct_positive = 100 * n_positive / df['Y'].notna().sum() if df['Y'].notna().sum() > 0 else 0

    logger.info(
        f"Predictive labels (L={lookahead_hours}h): "
        f"{n_positive:,} positive ({pct_positive:.1f}%)"
    )

    return df


def detect_flood_events(
    df: pd.DataFrame,
    config: Dict,
    wl_col: str = 'WL_MHHW'
) -> List[FloodEvent]:
    """
    Identify and characterize individual flood events.

    Parameters
    ----------
    df : pd.DataFrame
        Data with FloodNow column
    config : dict
        Event detection configuration:
        - min_separation_hours: Minimum gap between events
        - min_duration_hours: Minimum event duration
        - pre_event_window_hours: Window for baseline calculation
    wl_col : str
        Water level column name

    Returns
    -------
    list
        List of FloodEvent objects
    """
    if 'FloodNow' not in df.columns:
        raise ValueError("Must detect flood state first")

    min_sep = config.get('min_separation_hours', 12)
    min_dur = config.get('min_duration_hours', 1)
    pre_window = config.get('pre_event_window_hours', 48)

    events = []
    event_id = 0

    # Find contiguous flood periods
    flood_mask = df['FloodNow'] == 1
    if not flood_mask.any():
        logger.info("No flood events detected")
        return events

    # Group contiguous periods
    flood_groups = (flood_mask != flood_mask.shift()).cumsum()
    # Only keep flood period group IDs
    flood_group_ids = flood_groups[flood_mask].unique()

    for group_id in flood_group_ids:
        group_idx = df.index[(flood_groups == group_id) & flood_mask]

        if len(group_idx) < min_dur:
            continue

        start_time = group_idx[0]
        end_time = group_idx[-1]
        duration = (end_time - start_time).total_seconds() / 3600 + 1

        # Check separation from previous event
        if events:
            gap = (start_time - events[-1].end_time).total_seconds() / 3600
            if gap < min_sep:
                # Merge with previous event
                events[-1].end_time = end_time
                events[-1].duration_hours = (
                    (end_time - events[-1].start_time).total_seconds() / 3600 + 1
                )
                # Update peak if higher
                event_data = df.loc[start_time:end_time]
                if event_data[wl_col].max() > events[-1].peak_wl:
                    events[-1].peak_wl = event_data[wl_col].max()
                    events[-1].peak_time = event_data[wl_col].idxmax()
                continue

        # Extract event data
        event_data = df.loc[start_time:end_time]
        peak_wl = event_data[wl_col].max()
        peak_time = event_data[wl_col].idxmax()

        # Pre-event baseline
        pre_start = start_time - pd.Timedelta(hours=pre_window)
        pre_data = df.loc[pre_start:start_time - pd.Timedelta(hours=1), wl_col]
        pre_event_wl = pre_data.median() if len(pre_data) > 0 else np.nan

        # Get threshold from data attributes if available
        threshold = df.attrs.get('flood_threshold', 0)
        peak_exceedance = peak_wl - threshold

        # Peak discharge if available
        peak_q = None
        q_peak_time = None
        if 'Q' in df.columns:
            q_data = df.loc[pre_start:end_time, 'Q']
            if not q_data.isna().all():
                peak_q = q_data.max()
                q_peak_time = q_data.idxmax()

        event = FloodEvent(
            event_id=event_id,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration,
            peak_time=peak_time,
            peak_wl=peak_wl,
            peak_exceedance=peak_exceedance,
            pre_event_wl=pre_event_wl,
            peak_q=peak_q,
            q_peak_time=q_peak_time
        )
        events.append(event)
        event_id += 1

    logger.info(f"Detected {len(events)} flood events")

    return events


def events_to_dataframe(events: List[FloodEvent]) -> pd.DataFrame:
    """
    Convert list of flood events to DataFrame.

    Parameters
    ----------
    events : list
        List of FloodEvent objects

    Returns
    -------
    pd.DataFrame
        DataFrame with event information
    """
    if not events:
        return pd.DataFrame()

    records = []
    for e in events:
        records.append({
            'event_id': e.event_id,
            'start_time': e.start_time,
            'end_time': e.end_time,
            'duration_hours': e.duration_hours,
            'peak_time': e.peak_time,
            'peak_wl': e.peak_wl,
            'peak_exceedance': e.peak_exceedance,
            'pre_event_wl': e.pre_event_wl,
            'peak_q': e.peak_q,
            'q_peak_time': e.q_peak_time
        })

    return pd.DataFrame(records)


def compute_event_statistics(events: List[FloodEvent]) -> Dict:
    """
    Compute summary statistics for flood events.

    Parameters
    ----------
    events : list
        List of FloodEvent objects

    Returns
    -------
    dict
        Summary statistics
    """
    if not events:
        return {
            'n_events': 0,
            'total_flood_hours': 0
        }

    durations = [e.duration_hours for e in events]
    peaks = [e.peak_wl for e in events]
    exceedances = [e.peak_exceedance for e in events]

    # Events per year
    years = set()
    for e in events:
        years.add(e.start_time.year)
    n_years = len(years) if years else 1
    events_per_year = len(events) / n_years

    stats = {
        'n_events': len(events),
        'events_per_year': events_per_year,
        'total_flood_hours': sum(durations),
        'mean_duration_hours': np.mean(durations),
        'median_duration_hours': np.median(durations),
        'max_duration_hours': max(durations),
        'mean_peak_wl': np.mean(peaks),
        'max_peak_wl': max(peaks),
        'mean_exceedance': np.mean(exceedances),
        'max_exceedance': max(exceedances)
    }

    return stats


def get_event_q_wl_lags(events: List[FloodEvent]) -> List[float]:
    """
    Compute Q-to-WL lag for each event.

    Parameters
    ----------
    events : list
        List of FloodEvent objects with Q peak information

    Returns
    -------
    list
        List of lag values in hours (negative = Q leads WL)
    """
    lags = []

    for e in events:
        if e.q_peak_time is not None and e.peak_time is not None:
            lag = (e.peak_time - e.q_peak_time).total_seconds() / 3600
            lags.append(lag)

    return lags


def filter_events_by_period(
    events: List[FloodEvent],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> List[FloodEvent]:
    """
    Filter events to those within a date range.

    Parameters
    ----------
    events : list
        List of FloodEvent objects
    start_date : pd.Timestamp
        Start of period
    end_date : pd.Timestamp
        End of period

    Returns
    -------
    list
        Filtered list of events
    """
    filtered = [
        e for e in events
        if start_date <= e.start_time <= end_date
    ]
    return filtered


def get_event_windows(
    events: List[FloodEvent],
    pre_hours: int = 48,
    post_hours: int = 24
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get time windows around each event for analysis.

    Parameters
    ----------
    events : list
        List of FloodEvent objects
    pre_hours : int
        Hours before event start
    post_hours : int
        Hours after event end

    Returns
    -------
    list
        List of (start, end) tuples
    """
    windows = []

    for e in events:
        win_start = e.start_time - pd.Timedelta(hours=pre_hours)
        win_end = e.end_time + pd.Timedelta(hours=post_hours)
        windows.append((win_start, win_end))

    return windows
