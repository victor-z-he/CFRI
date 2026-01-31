"""
Utility functions for compound flood analysis.

This module provides:
    - Logging configuration
    - Configuration management
    - Common helper functions
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB"
) -> None:
    """
    Configure loguru logger for the project.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    log_file : Path, optional
        Path to log file. If None, logs only to stderr.
    rotation : str
        Log rotation size
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=log_format, level=log_level, colorize=True)

    # File handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention="30 days"
        )

    logger.info(f"Logging initialized at level {log_level}")


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    FileNotFoundError
        If config file does not exist
    yaml.YAMLError
        If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['global', 'sites', 'preprocessing', 'features', 'thresholding']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def ensure_output_dirs(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create output directory structure.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory

    Returns
    -------
    dict
        Dictionary mapping output types to paths
    """
    output_dir = Path(output_dir)

    dirs = {
        'base': output_dir,
        'figures': output_dir / 'figures',
        'tables': output_dir / 'tables',
        'reports': output_dir / 'reports'
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory: {path}")

    return dirs


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value
    """
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def validate_datetime_index(
    df: pd.DataFrame,
    datetime_col: str = None
) -> pd.DataFrame:
    """
    Validate and standardize datetime index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    datetime_col : str, optional
        Column name containing datetime. If None, assumes index is datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with validated DatetimeIndex in UTC

    Raises
    ------
    ValueError
        If datetime validation fails
    """
    df = df.copy()

    # Set datetime as index if specified
    if datetime_col is not None:
        if datetime_col not in df.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found")
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
        df = df.set_index(datetime_col)

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Ensure UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # Check for monotonicity
    if not df.index.is_monotonic_increasing:
        logger.warning("Datetime index is not monotonic. Sorting...")
        df = df.sort_index()

    # Check for duplicates
    n_duplicates = df.index.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate timestamps. Keeping first.")
        df = df[~df.index.duplicated(keep='first')]

    return df


def compute_percentile_rank(
    values: np.ndarray,
    reference_values: np.ndarray = None
) -> np.ndarray:
    """
    Compute percentile rank of values.

    Parameters
    ----------
    values : np.ndarray
        Values to rank
    reference_values : np.ndarray, optional
        Reference distribution for ranking. If None, uses values.

    Returns
    -------
    np.ndarray
        Percentile ranks in [0, 1]
    """
    if reference_values is None:
        reference_values = values

    # Remove NaN from reference
    ref_valid = reference_values[~np.isnan(reference_values)]

    if len(ref_valid) == 0:
        return np.full_like(values, np.nan, dtype=float)

    # Compute ranks using searchsorted
    ranks = np.searchsorted(np.sort(ref_valid), values, side='right')
    percentiles = ranks / len(ref_valid)

    # Handle NaN in input
    percentiles = np.where(np.isnan(values), np.nan, percentiles)

    return np.clip(percentiles, 0, 1)


def robust_zscore(
    values: np.ndarray,
    reference_values: np.ndarray = None
) -> np.ndarray:
    """
    Compute robust z-score using median and MAD.

    Parameters
    ----------
    values : np.ndarray
        Values to normalize
    reference_values : np.ndarray, optional
        Reference distribution. If None, uses values.

    Returns
    -------
    np.ndarray
        Robust z-scores
    """
    if reference_values is None:
        reference_values = values

    ref_valid = reference_values[~np.isnan(reference_values)]

    if len(ref_valid) == 0:
        return np.full_like(values, np.nan, dtype=float)

    median = np.median(ref_valid)
    mad = np.median(np.abs(ref_valid - median))

    # Avoid division by zero
    if mad == 0:
        mad = np.std(ref_valid)
    if mad == 0:
        return np.zeros_like(values, dtype=float)

    # Scale factor for MAD to approximate std
    k = 1.4826

    zscores = (values - median) / (k * mad)
    return np.where(np.isnan(values), np.nan, zscores)


def format_duration(hours: float) -> str:
    """
    Format duration in hours to human-readable string.

    Parameters
    ----------
    hours : float
        Duration in hours

    Returns
    -------
    str
        Formatted duration string
    """
    if hours < 1:
        return f"{hours * 60:.0f} min"
    elif hours < 24:
        return f"{hours:.1f} hr"
    else:
        days = hours / 24
        return f"{days:.1f} days"


def create_train_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    by_year: bool = True
) -> tuple:
    """
    Create temporal train/test split avoiding data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    train_fraction : float
        Fraction of data for training
    by_year : bool
        If True, split by complete years

    Returns
    -------
    tuple
        (train_df, test_df, split_date)
    """
    if by_year:
        years = df.index.year.unique()
        n_train_years = int(len(years) * train_fraction)
        train_years = years[:n_train_years]
        test_years = years[n_train_years:]

        train_df = df[df.index.year.isin(train_years)]
        test_df = df[df.index.year.isin(test_years)]
        split_date = pd.Timestamp(f"{test_years[0]}-01-01", tz='UTC')
    else:
        n_train = int(len(df) * train_fraction)
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]
        split_date = test_df.index[0]

    logger.info(f"Train/test split at {split_date.strftime('%Y-%m-%d')}")
    logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    return train_df, test_df, split_date


def detect_outliers_iqr(
    values: np.ndarray,
    factor: float = 3.0
) -> np.ndarray:
    """
    Detect outliers using IQR method.

    Parameters
    ----------
    values : np.ndarray
        Values to check
    factor : float
        IQR multiplier for outlier threshold

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates outlier
    """
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return np.zeros_like(values, dtype=bool)

    q1 = np.percentile(valid, 25)
    q3 = np.percentile(valid, 75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    outliers = (values < lower) | (values > upper)
    return np.where(np.isnan(values), False, outliers)


def summarize_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing data in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    pd.DataFrame
        Summary with missing counts and percentages
    """
    summary = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_pct': 100 * df.isna().mean(),
        'total_count': len(df)
    })
    return summary


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
