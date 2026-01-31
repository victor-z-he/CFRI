"""
Data preprocessing module for compound flood analysis.

This module handles:
    - Time alignment of river and tide data
    - Datum conversion to MHHW reference
    - Missing data handling
    - Outlier detection and handling
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .utils import detect_outliers_iqr


def preprocess_site(
    river_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    site_config: Dict,
    preprocess_config: Dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess river and tide data for a site.

    Performs:
        1. Time alignment to common hourly index
        2. Missing data handling
        3. Outlier detection
        4. Datum conversion to MHHW

    Parameters
    ----------
    river_df : pd.DataFrame
        River discharge data with 'Q' column
    tide_df : pd.DataFrame
        Tide data with 'WL' column
    site_config : dict
        Site-specific configuration
    preprocess_config : dict
        Preprocessing parameters
    start_date : str, optional
        Analysis start date
    end_date : str, optional
        Analysis end date

    Returns
    -------
    tuple
        (merged_df, preprocessing_stats)
        - merged_df: Aligned DataFrame with Q, WL_MHHW columns
        - preprocessing_stats: Dict with processing statistics
    """
    site_name = site_config['name']
    logger.info(f"Preprocessing data for {site_name}")

    stats = {
        'site': site_name,
        'original_river_count': len(river_df),
        'original_tide_count': len(tide_df),
        'interpolated_river_pct': 0.0,
        'interpolated_tide_pct': 0.0,
        'outliers_river': 0,
        'outliers_tide': 0
    }

    # Step 1: Align to common hourly index
    merged_df, align_stats = align_timeseries(
        river_df, tide_df,
        start_date=start_date,
        end_date=end_date
    )
    stats.update(align_stats)

    # Step 2: Handle missing data
    merged_df, missing_stats = handle_missing_data(
        merged_df,
        preprocess_config.get('missing_data', {})
    )
    stats.update(missing_stats)

    # Step 3: Handle outliers
    merged_df, outlier_stats = handle_outliers(
        merged_df,
        preprocess_config.get('outliers', {})
    )
    stats.update(outlier_stats)

    # Step 4: Convert WL to MHHW reference
    merged_df = convert_to_mhhw(
        merged_df,
        site_config['datums'],
        site_config['data']['tide'].get('datum', 'MLLW')
    )

    # Log summary
    logger.info(f"  Aligned records: {len(merged_df)}")
    logger.info(f"  Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    logger.info(f"  Valid Q: {merged_df['Q'].notna().sum()}")
    logger.info(f"  Valid WL_MHHW: {merged_df['WL_MHHW'].notna().sum()}")

    stats['final_count'] = len(merged_df)
    stats['valid_q_count'] = merged_df['Q'].notna().sum()
    stats['valid_wl_count'] = merged_df['WL_MHHW'].notna().sum()

    return merged_df, stats


def align_timeseries(
    river_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: str = 'H'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Align river and tide time series to common index.

    Parameters
    ----------
    river_df : pd.DataFrame
        River discharge data
    tide_df : pd.DataFrame
        Tide gauge data
    start_date : str, optional
        Start date for alignment
    end_date : str, optional
        End date for alignment
    freq : str
        Target frequency (default: 'H' for hourly)

    Returns
    -------
    tuple
        (aligned_df, alignment_stats)
    """
    stats = {}

    # Determine common date range
    common_start = max(river_df.index.min(), tide_df.index.min())
    common_end = min(river_df.index.max(), tide_df.index.max())

    if start_date:
        common_start = max(common_start, pd.Timestamp(start_date, tz='UTC'))
    if end_date:
        common_end = min(common_end, pd.Timestamp(end_date, tz='UTC'))

    if common_start >= common_end:
        raise ValueError("No valid overlapping date range")

    stats['aligned_start'] = common_start.isoformat()
    stats['aligned_end'] = common_end.isoformat()

    # Create regular hourly index
    target_index = pd.date_range(
        start=common_start.floor('H'),
        end=common_end.ceil('H'),
        freq=freq,
        tz='UTC'
    )

    # Resample to regular hourly index
    river_resampled = _resample_to_hourly(river_df, target_index, 'Q')
    tide_resampled = _resample_to_hourly(tide_df, target_index, 'WL')

    # Merge
    merged = pd.DataFrame(index=target_index)
    merged['Q'] = river_resampled['Q']
    merged['WL'] = tide_resampled['WL']

    stats['n_hours'] = len(merged)

    return merged, stats


def _resample_to_hourly(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    col: str
) -> pd.DataFrame:
    """
    Resample data to hourly index.

    For sub-hourly data, takes the mean within each hour.
    For data already hourly or coarser, uses nearest neighbor.
    """
    # Check if data is already hourly
    if len(df) > 1:
        median_diff = df.index.to_series().diff().median()

        if median_diff <= pd.Timedelta(hours=1):
            # Sub-hourly or hourly: resample with mean
            hourly = df.resample('H').mean()
        else:
            # Coarser than hourly: forward fill or interpolate
            hourly = df.reindex(target_index, method='ffill', limit=2)
    else:
        hourly = df.reindex(target_index)

    # Align to target index
    result = pd.DataFrame(index=target_index)
    result[col] = hourly.reindex(target_index)[col]

    return result


def handle_missing_data(
    df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values in the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with potential missing values
    config : dict
        Missing data configuration:
        - max_gap_hours: Maximum gap size to interpolate
        - interpolation_method: Method for interpolation

    Returns
    -------
    tuple
        (processed_df, missing_stats)
    """
    df = df.copy()
    stats = {}

    max_gap = config.get('max_gap_hours', 6)
    method = config.get('interpolation_method', 'linear')

    for col in ['Q', 'WL']:
        if col not in df.columns:
            continue

        original_missing = df[col].isna().sum()

        # Identify gaps
        is_missing = df[col].isna()
        gap_groups = (is_missing != is_missing.shift()).cumsum()
        gap_sizes = is_missing.groupby(gap_groups).transform('sum')

        # Only interpolate small gaps
        small_gaps = is_missing & (gap_sizes <= max_gap)

        if small_gaps.any():
            # Mark positions to interpolate
            df[f'{col}_interpolated'] = False
            df.loc[small_gaps, f'{col}_interpolated'] = True

            # Interpolate
            df[col] = df[col].interpolate(method=method, limit=max_gap)

        # Count what was interpolated
        interpolated = df.get(f'{col}_interpolated', pd.Series(False, index=df.index)).sum()
        final_missing = df[col].isna().sum()

        stats[f'{col.lower()}_original_missing'] = int(original_missing)
        stats[f'{col.lower()}_interpolated'] = int(interpolated)
        stats[f'{col.lower()}_final_missing'] = int(final_missing)
        stats[f'{col.lower()}_interpolated_pct'] = (
            100 * interpolated / len(df) if len(df) > 0 else 0
        )

        logger.debug(
            f"  {col}: {original_missing} missing -> {interpolated} interpolated "
            f"-> {final_missing} remaining"
        )

    return df, stats


def handle_outliers(
    df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and handle outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    config : dict
        Outlier configuration:
        - method: Detection method ('iqr', 'zscore', or 'none')
        - iqr_factor: IQR multiplier for outlier threshold
        - action: What to do with outliers ('flag', 'remove', 'clip')

    Returns
    -------
    tuple
        (processed_df, outlier_stats)
    """
    df = df.copy()
    stats = {}

    method = config.get('method', 'iqr')
    action = config.get('action', 'flag')
    iqr_factor = config.get('iqr_factor', 3.0)

    if method == 'none':
        return df, stats

    for col in ['Q', 'WL']:
        if col not in df.columns:
            continue

        values = df[col].values

        if method == 'iqr':
            outliers = detect_outliers_iqr(values, factor=iqr_factor)
        else:
            outliers = np.zeros(len(values), dtype=bool)

        n_outliers = outliers.sum()
        stats[f'{col.lower()}_outliers'] = int(n_outliers)

        if n_outliers > 0:
            logger.info(f"  {col}: {n_outliers} outliers detected")

            # Store outlier flag
            df[f'{col}_outlier'] = outliers

            if action == 'remove':
                df.loc[outliers, col] = np.nan
            elif action == 'clip':
                valid = values[~np.isnan(values) & ~outliers]
                if len(valid) > 0:
                    lower = np.percentile(valid, 1)
                    upper = np.percentile(valid, 99)
                    df[col] = df[col].clip(lower, upper)

    return df, stats


def convert_to_mhhw(
    df: pd.DataFrame,
    datums: Dict,
    source_datum: str = 'MLLW'
) -> pd.DataFrame:
    """
    Convert water level to MHHW reference.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'WL' column
    datums : dict
        Datum conversion parameters:
        - mhhw_above_mllw: MHHW - MLLW offset (meters)
        - navd88_to_mllw: NAVD88 to MLLW offset (if applicable)
    source_datum : str
        Original datum of WL data ('MLLW', 'NAVD88', or 'MHHW')

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'WL_MHHW' column
    """
    df = df.copy()

    mhhw_above_mllw = datums.get('mhhw_above_mllw', 0)
    navd88_to_mllw = datums.get('navd88_to_mllw', 0)

    if source_datum.upper() == 'MLLW':
        # WL_MHHW = WL_MLLW - (MHHW - MLLW)
        df['WL_MHHW'] = df['WL'] - mhhw_above_mllw

    elif source_datum.upper() == 'NAVD88':
        # First convert to MLLW, then to MHHW
        wl_mllw = df['WL'] + navd88_to_mllw
        df['WL_MHHW'] = wl_mllw - mhhw_above_mllw

    elif source_datum.upper() == 'MHHW':
        # Already in MHHW
        df['WL_MHHW'] = df['WL']

    else:
        logger.warning(f"Unknown source datum: {source_datum}. Assuming MLLW.")
        df['WL_MHHW'] = df['WL'] - mhhw_above_mllw

    # Store conversion info
    df.attrs['wl_datum'] = 'MHHW'
    df.attrs['mhhw_offset_used'] = mhhw_above_mllw

    logger.info(f"  Converted WL from {source_datum} to MHHW (offset: {mhhw_above_mllw:.3f} m)")

    return df


def compute_water_level_anomaly(
    df: pd.DataFrame,
    reference_period: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Compute water level anomaly from climatological mean.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'WL_MHHW' column
    reference_period : tuple, optional
        (start_date, end_date) for computing climatology

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'WL_anom' column
    """
    df = df.copy()

    if reference_period:
        start, end = reference_period
        ref_data = df.loc[start:end, 'WL_MHHW']
    else:
        ref_data = df['WL_MHHW']

    # Compute hourly climatology
    hourly_mean = ref_data.groupby([
        ref_data.index.month,
        ref_data.index.day,
        ref_data.index.hour
    ]).mean()

    # Create anomaly
    df['WL_anom'] = df['WL_MHHW'].copy()

    for idx in df.index:
        key = (idx.month, idx.day, idx.hour)
        if key in hourly_mean.index:
            df.loc[idx, 'WL_anom'] -= hourly_mean[key]

    return df


def get_preprocessing_summary(stats: Dict) -> str:
    """
    Generate human-readable preprocessing summary.

    Parameters
    ----------
    stats : dict
        Preprocessing statistics

    Returns
    -------
    str
        Formatted summary string
    """
    lines = [
        f"Preprocessing Summary for {stats.get('site', 'Unknown Site')}",
        "=" * 50,
        f"Date range: {stats.get('aligned_start', 'N/A')} to {stats.get('aligned_end', 'N/A')}",
        f"Total hours: {stats.get('n_hours', 'N/A'):,}",
        "",
        "Missing Data:",
        f"  River (Q): {stats.get('q_original_missing', 0):,} -> "
        f"{stats.get('q_final_missing', 0):,} ({stats.get('q_interpolated', 0)} interpolated)",
        f"  Tide (WL): {stats.get('wl_original_missing', 0):,} -> "
        f"{stats.get('wl_final_missing', 0):,} ({stats.get('wl_interpolated', 0)} interpolated)",
        "",
        "Outliers:",
        f"  River (Q): {stats.get('q_outliers', 0)} detected",
        f"  Tide (WL): {stats.get('wl_outliers', 0)} detected",
        "",
        "Final Valid Data:",
        f"  River (Q): {stats.get('valid_q_count', 0):,} records",
        f"  Tide (WL): {stats.get('valid_wl_count', 0):,} records",
    ]

    return "\n".join(lines)
