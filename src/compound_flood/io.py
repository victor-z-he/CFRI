"""
Data I/O module for compound flood analysis.

This module handles:
    - Configuration file loading
    - Time series data loading (CSV/Parquet)
    - Data validation and quality checks
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .utils import load_yaml_config, validate_datetime_index, summarize_missing_data


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate project configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Validated configuration dictionary
    """
    config = load_yaml_config(config_path)

    # Validate site configurations
    for site_name, site_cfg in config['sites'].items():
        _validate_site_config(site_name, site_cfg)

    logger.info(f"Configuration loaded with {len(config['sites'])} sites")
    return config


def _validate_site_config(site_name: str, site_cfg: Dict) -> None:
    """Validate individual site configuration."""
    required_keys = ['name', 'data', 'stations', 'datums', 'flood_thresholds']

    for key in required_keys:
        if key not in site_cfg:
            raise ValueError(f"Site '{site_name}' missing required key: {key}")

    # Validate data paths
    for data_type in ['river', 'tide']:
        if data_type not in site_cfg['data']:
            raise ValueError(f"Site '{site_name}' missing {data_type} data config")

        data_cfg = site_cfg['data'][data_type]
        required_data_keys = ['file', 'datetime_col', 'value_col']
        for key in required_data_keys:
            if key not in data_cfg:
                raise ValueError(
                    f"Site '{site_name}' {data_type} data missing: {key}"
                )


def load_site_data(
    site_config: Dict,
    base_path: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load river and tide data for a site.

    Parameters
    ----------
    site_config : dict
        Site configuration from YAML
    base_path : Path, optional
        Base path for relative file paths
    start_date : str, optional
        Start date for filtering (YYYY-MM-DD)
    end_date : str, optional
        End date for filtering (YYYY-MM-DD)

    Returns
    -------
    tuple
        (river_df, tide_df, metadata)
        - river_df: DataFrame with 'Q' column (discharge)
        - tide_df: DataFrame with 'WL' column (water level)
        - metadata: Dict with loading statistics
    """
    site_name = site_config['name']
    logger.info(f"Loading data for site: {site_name}")

    base_path = Path(base_path) if base_path else Path('.')
    metadata = {'site': site_name, 'issues': []}

    # Load river data
    river_cfg = site_config['data']['river']
    river_df, river_meta = _load_timeseries(
        file_path=base_path / river_cfg['file'],
        datetime_col=river_cfg['datetime_col'],
        value_col=river_cfg['value_col'],
        output_col='Q',
        start_date=start_date,
        end_date=end_date
    )
    metadata['river'] = river_meta

    # Load tide data
    tide_cfg = site_config['data']['tide']
    tide_df, tide_meta = _load_timeseries(
        file_path=base_path / tide_cfg['file'],
        datetime_col=tide_cfg['datetime_col'],
        value_col=tide_cfg['value_col'],
        output_col='WL',
        start_date=start_date,
        end_date=end_date
    )
    metadata['tide'] = tide_meta

    # Add datum info
    metadata['datum'] = site_config['data']['tide'].get('datum', 'unknown')

    # Log summary
    logger.info(
        f"  River: {len(river_df)} records, "
        f"{river_meta['missing_pct']:.1f}% missing"
    )
    logger.info(
        f"  Tide: {len(tide_df)} records, "
        f"{tide_meta['missing_pct']:.1f}% missing"
    )

    return river_df, tide_df, metadata


def _load_timeseries(
    file_path: Path,
    datetime_col: str,
    value_col: str,
    output_col: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a single time series file.

    Parameters
    ----------
    file_path : Path
        Path to data file (CSV or Parquet)
    datetime_col : str
        Name of datetime column
    value_col : str
        Name of value column
    output_col : str
        Name for output column
    start_date : str, optional
        Start date filter
    end_date : str, optional
        End date filter

    Returns
    -------
    tuple
        (DataFrame, metadata dict)
    """
    file_path = Path(file_path)
    metadata = {
        'file': str(file_path),
        'original_count': 0,
        'final_count': 0,
        'missing_count': 0,
        'missing_pct': 0.0,
        'duplicates_removed': 0,
        'date_range': None
    }

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load based on file type
    logger.debug(f"Loading {file_path}")

    if file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
    else:  # Assume CSV
        df = pd.read_csv(file_path)

    metadata['original_count'] = len(df)

    # Validate required columns
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not in {file_path}")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not in {file_path}")

    # Parse datetime and set as index
    df = validate_datetime_index(df, datetime_col=datetime_col)

    # Track duplicates removed
    original_len = len(df)
    df = df[~df.index.duplicated(keep='first')]
    metadata['duplicates_removed'] = original_len - len(df)

    # Rename value column
    df = df[[value_col]].rename(columns={value_col: output_col})

    # Filter date range
    if start_date:
        start_dt = pd.Timestamp(start_date, tz='UTC')
        df = df[df.index >= start_dt]
    if end_date:
        end_dt = pd.Timestamp(end_date, tz='UTC')
        df = df[df.index <= end_dt]

    # Compute missing statistics
    metadata['final_count'] = len(df)
    metadata['missing_count'] = df[output_col].isna().sum()
    metadata['missing_pct'] = 100 * metadata['missing_count'] / max(len(df), 1)

    if len(df) > 0:
        metadata['date_range'] = (
            df.index.min().strftime('%Y-%m-%d'),
            df.index.max().strftime('%Y-%m-%d')
        )

    return df, metadata


def load_or_create_synthetic_data(
    site_config: Dict,
    base_path: Optional[Path] = None,
    n_years: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load data or create synthetic data for testing.

    If data files don't exist, creates realistic synthetic data
    for development and testing purposes.

    Parameters
    ----------
    site_config : dict
        Site configuration
    base_path : Path, optional
        Base path for data files
    n_years : int
        Number of years for synthetic data

    Returns
    -------
    tuple
        (river_df, tide_df, metadata)
    """
    base_path = Path(base_path) if base_path else Path('.')

    river_file = base_path / site_config['data']['river']['file']
    tide_file = base_path / site_config['data']['tide']['file']

    # Try loading real data first
    if river_file.exists() and tide_file.exists():
        return load_site_data(site_config, base_path)

    # Create synthetic data
    logger.warning("Data files not found. Creating synthetic data for testing.")

    site_name = site_config['name']
    metadata = {'site': site_name, 'synthetic': True, 'issues': []}

    # Generate date range
    end_date = pd.Timestamp('2023-12-31 23:00:00', tz='UTC')
    start_date = end_date - pd.DateOffset(years=n_years)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H', tz='UTC')

    n_hours = len(date_range)
    np.random.seed(42)  # Reproducibility

    # Generate river discharge with seasonality and events
    # Base seasonal pattern (higher in winter/spring)
    day_of_year = date_range.dayofyear.values
    seasonal = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 90) / 365)

    # Add random events (storms)
    n_events = n_years * 8  # ~8 events per year
    event_times = np.random.choice(n_hours, n_events, replace=False)
    event_magnitudes = np.random.exponential(2, n_events)

    events = np.zeros(n_hours)
    for t, mag in zip(event_times, event_magnitudes):
        # Event rises over 12 hours, falls over 48 hours
        for dt in range(72):
            if t + dt < n_hours:
                if dt < 12:
                    events[t + dt] += mag * (dt / 12)
                else:
                    events[t + dt] += mag * np.exp(-(dt - 12) / 24)

    # Combine components
    base_q = 1000 + 500 * seasonal + 100 * np.random.randn(n_hours)
    q_values = base_q + 2000 * events
    q_values = np.maximum(q_values, 100)  # Minimum flow

    river_df = pd.DataFrame({'Q': q_values}, index=date_range)

    # Generate water level with tides and surge
    # Semi-diurnal tide
    hours = np.arange(n_hours)
    tide = 0.5 * np.sin(2 * np.pi * hours / 12.42)  # M2 constituent
    tide += 0.15 * np.sin(2 * np.pi * hours / 12.0)  # S2 constituent

    # Add surge correlated with river events (with lag)
    lag_hours = 24  # River-to-estuary lag
    lagged_events = np.roll(events, lag_hours)
    lagged_events[:lag_hours] = 0

    surge = 0.3 * lagged_events + 0.1 * np.random.randn(n_hours)

    # Combine: tide + surge + noise
    wl_values = tide + surge + 0.05 * np.random.randn(n_hours)

    # Convert to MLLW reference (add offset to make positive)
    mhhw_offset = site_config['datums']['mhhw_above_mllw']
    wl_values = wl_values + mhhw_offset + 0.3  # Center around MLLW

    tide_df = pd.DataFrame({'WL': wl_values}, index=date_range)

    # Add some missing data (realistic gaps)
    missing_pct = 0.02  # 2% missing
    n_missing = int(n_hours * missing_pct)
    missing_idx_q = np.random.choice(n_hours, n_missing, replace=False)
    missing_idx_wl = np.random.choice(n_hours, n_missing, replace=False)

    river_df.iloc[missing_idx_q, 0] = np.nan
    tide_df.iloc[missing_idx_wl, 0] = np.nan

    metadata['river'] = {
        'synthetic': True,
        'final_count': len(river_df),
        'missing_pct': 100 * river_df['Q'].isna().mean()
    }
    metadata['tide'] = {
        'synthetic': True,
        'final_count': len(tide_df),
        'missing_pct': 100 * tide_df['WL'].isna().mean()
    }
    metadata['datum'] = 'MLLW'

    logger.info(f"Created {n_years} years of synthetic data for {site_name}")

    return river_df, tide_df, metadata


def save_dataframe(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = 'csv'
) -> None:
    """
    Save DataFrame to file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str or Path
        Output file path
    format : str
        Output format ('csv' or 'parquet')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'parquet':
        df.to_parquet(output_path)
    else:
        df.to_csv(output_path)

    logger.info(f"Saved: {output_path}")


def validate_loaded_data(
    river_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    config: Dict
) -> Dict:
    """
    Validate loaded data quality.

    Parameters
    ----------
    river_df : pd.DataFrame
        River discharge data
    tide_df : pd.DataFrame
        Tide gauge data
    config : dict
        Preprocessing configuration

    Returns
    -------
    dict
        Validation results and warnings
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }

    missing_threshold = config.get('missing_data', {}).get('report_threshold', 0.05)

    # Check missing data
    river_missing = river_df['Q'].isna().mean()
    tide_missing = tide_df['WL'].isna().mean()

    if river_missing > missing_threshold:
        results['warnings'].append(
            f"River data has {100*river_missing:.1f}% missing values"
        )

    if tide_missing > missing_threshold:
        results['warnings'].append(
            f"Tide data has {100*tide_missing:.1f}% missing values"
        )

    # Check for negative discharge
    if (river_df['Q'] < 0).any():
        n_negative = (river_df['Q'] < 0).sum()
        results['warnings'].append(
            f"River discharge has {n_negative} negative values"
        )

    # Check date overlap
    river_range = (river_df.index.min(), river_df.index.max())
    tide_range = (tide_df.index.min(), tide_df.index.max())

    overlap_start = max(river_range[0], tide_range[0])
    overlap_end = min(river_range[1], tide_range[1])

    if overlap_start >= overlap_end:
        results['valid'] = False
        results['errors'].append("No overlapping date range between river and tide data")
    else:
        overlap_days = (overlap_end - overlap_start).days
        results['overlap_days'] = overlap_days
        logger.info(f"Data overlap: {overlap_days} days")

    # Log warnings
    for warning in results['warnings']:
        logger.warning(warning)

    for error in results['errors']:
        logger.error(error)

    return results
