"""
Threshold selection module for CFRI trigger optimization.

This module handles:
    - Grid search over CFRI thresholds
    - Train/test split validation
    - Threshold selection with constraints
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def create_predictions(
    df: pd.DataFrame,
    index_col: str,
    threshold: float
) -> pd.Series:
    """
    Create binary predictions from index values.

    Parameters
    ----------
    df : pd.DataFrame
        Data with index column
    index_col : str
        Column name of the index (CFRI, WL_only, etc.)
    threshold : float
        Trigger threshold

    Returns
    -------
    pd.Series
        Binary predictions (1 = predict flood)
    """
    predictions = (df[index_col] > threshold).astype(int)
    predictions[df[index_col].isna()] = np.nan
    return predictions


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    dict
        Dictionary of metrics
    """
    # Remove NaN
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
            'far': np.nan, 'accuracy': np.nan,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
        }

    # Confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0  # False Alarm Ratio
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def grid_search_threshold(
    df: pd.DataFrame,
    index_col: str,
    target_col: str = 'Y',
    threshold_range: Tuple[float, float, float] = (0.20, 0.95, 0.01)
) -> pd.DataFrame:
    """
    Grid search over thresholds to find optimal trigger.

    Parameters
    ----------
    df : pd.DataFrame
        Data with index and target columns
    index_col : str
        Index column to threshold
    target_col : str
        Target column (binary)
    threshold_range : tuple
        (start, end, step) for threshold grid

    Returns
    -------
    pd.DataFrame
        Results for each threshold
    """
    start, end, step = threshold_range
    thresholds = np.arange(start, end + step, step)

    results = []

    for T in thresholds:
        y_pred = create_predictions(df, index_col, T)
        y_true = df[target_col]

        metrics = compute_classification_metrics(
            y_true.values, y_pred.values
        )
        metrics['threshold'] = T

        results.append(metrics)

    return pd.DataFrame(results)


def select_threshold(
    df: pd.DataFrame,
    config: Dict,
    index_col: str = 'CFRI'
) -> Dict:
    """
    Select optimal threshold using train/test validation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and labels
    config : dict
        Thresholding configuration
    index_col : str
        Index column to optimize

    Returns
    -------
    dict
        Selection results including optimal threshold
    """
    thresh_config = config.get('thresholding', {})
    grid_config = thresh_config.get('grid', {})
    selection_config = thresh_config.get('selection', {})

    # Grid parameters
    grid_start = grid_config.get('start', 0.20)
    grid_end = grid_config.get('end', 0.95)
    grid_step = grid_config.get('step', 0.01)

    # Selection criteria
    primary_metric = selection_config.get('primary_metric', 'f1')
    max_far = selection_config.get('max_far', 0.30)
    tiebreaker = selection_config.get('tiebreaker', 'lead_time')

    # Train/test split
    train_frac = thresh_config.get('train_fraction', 0.70)
    years = df.index.year.unique()
    n_train_years = int(len(years) * train_frac)
    train_years = years[:n_train_years]
    test_years = years[n_train_years:]

    train_df = df[df.index.year.isin(train_years)]
    test_df = df[df.index.year.isin(test_years)]

    logger.info(f"Train years: {train_years[0]}-{train_years[-1]}")
    logger.info(f"Test years: {test_years[0]}-{test_years[-1]}")

    # Grid search on training data
    train_results = grid_search_threshold(
        train_df, index_col, 'Y',
        (grid_start, grid_end, grid_step)
    )

    # Filter by FAR constraint
    valid_thresholds = train_results[train_results['far'] <= max_far]

    if len(valid_thresholds) == 0:
        logger.warning(f"No thresholds satisfy FAR <= {max_far}. Relaxing constraint.")
        valid_thresholds = train_results

    # Select best by primary metric
    best_idx = valid_thresholds[primary_metric].idxmax()
    optimal_T = valid_thresholds.loc[best_idx, 'threshold']
    train_metrics = valid_thresholds.loc[best_idx].to_dict()

    # Evaluate on test data
    test_pred = create_predictions(test_df, index_col, optimal_T)
    test_metrics = compute_classification_metrics(
        test_df['Y'].values, test_pred.values
    )
    test_metrics['threshold'] = optimal_T

    # Compile results
    results = {
        'optimal_threshold': optimal_T,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_years': list(train_years),
        'test_years': list(test_years),
        'grid_results': train_results,
        'selection_criteria': {
            'primary_metric': primary_metric,
            'max_far': max_far
        }
    }

    logger.info(f"Optimal threshold: {optimal_T:.3f}")
    logger.info(f"  Train F1: {train_metrics['f1']:.3f}, FAR: {train_metrics['far']:.3f}")
    logger.info(f"  Test F1: {test_metrics['f1']:.3f}, FAR: {test_metrics['far']:.3f}")

    return results


def cross_validate_threshold(
    df: pd.DataFrame,
    config: Dict,
    index_col: str = 'CFRI',
    n_splits: int = 5
) -> Dict:
    """
    Cross-validate threshold selection using blocked time-series CV.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    config : dict
        Configuration
    index_col : str
        Index column
    n_splits : int
        Number of CV folds

    Returns
    -------
    dict
        Cross-validation results
    """
    grid_config = config.get('thresholding', {}).get('grid', {})
    grid_start = grid_config.get('start', 0.20)
    grid_end = grid_config.get('end', 0.95)
    grid_step = grid_config.get('step', 0.01)

    years = df.index.year.unique()
    n_years = len(years)
    fold_size = n_years // n_splits

    cv_results = []

    for fold in range(n_splits):
        # Test fold years
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n_years)
        test_years = years[test_start:test_end]

        # Train = all other years
        train_years = np.array([y for y in years if y not in test_years])

        train_df = df[df.index.year.isin(train_years)]
        test_df = df[df.index.year.isin(test_years)]

        # Grid search on train
        train_results = grid_search_threshold(
            train_df, index_col, 'Y',
            (grid_start, grid_end, grid_step)
        )

        # Find best threshold
        best_idx = train_results['f1'].idxmax()
        optimal_T = train_results.loc[best_idx, 'threshold']

        # Evaluate on test
        test_pred = create_predictions(test_df, index_col, optimal_T)
        test_metrics = compute_classification_metrics(
            test_df['Y'].values, test_pred.values
        )

        cv_results.append({
            'fold': fold,
            'threshold': optimal_T,
            'test_f1': test_metrics['f1'],
            'test_far': test_metrics['far'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        })

    cv_df = pd.DataFrame(cv_results)

    # Summary statistics
    summary = {
        'mean_threshold': cv_df['threshold'].mean(),
        'std_threshold': cv_df['threshold'].std(),
        'mean_f1': cv_df['test_f1'].mean(),
        'std_f1': cv_df['test_f1'].std(),
        'mean_far': cv_df['test_far'].mean(),
        'fold_results': cv_df
    }

    logger.info(f"CV threshold: {summary['mean_threshold']:.3f} ± {summary['std_threshold']:.3f}")
    logger.info(f"CV F1: {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}")

    return summary


def evaluate_all_models(
    df: pd.DataFrame,
    threshold_results: Dict,
    baselines: List[str] = None
) -> pd.DataFrame:
    """
    Evaluate CFRI and baseline models with their optimal thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Test data
    threshold_results : dict
        Results from threshold selection
    baselines : list
        Baseline model names

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    if baselines is None:
        baselines = ['WL_only', 'Q_only', 'Persistence']

    results = []

    # CFRI
    cfri_T = threshold_results['optimal_threshold']
    cfri_pred = create_predictions(df, 'CFRI', cfri_T)
    cfri_metrics = compute_classification_metrics(df['Y'].values, cfri_pred.values)
    cfri_metrics['model'] = 'CFRI'
    cfri_metrics['threshold'] = cfri_T
    results.append(cfri_metrics)

    # Baselines
    for baseline in baselines:
        if baseline not in df.columns:
            continue

        # Find optimal threshold for baseline
        baseline_results = grid_search_threshold(
            df, baseline, 'Y', (0.20, 0.95, 0.01)
        )
        best_idx = baseline_results['f1'].idxmax()
        baseline_T = baseline_results.loc[best_idx, 'threshold']

        baseline_pred = create_predictions(df, baseline, baseline_T)
        baseline_metrics = compute_classification_metrics(
            df['Y'].values, baseline_pred.values
        )
        baseline_metrics['model'] = baseline
        baseline_metrics['threshold'] = baseline_T
        results.append(baseline_metrics)

    return pd.DataFrame(results)


def compute_threshold_stability(
    results: pd.DataFrame,
    threshold: float,
    tolerance: float = 0.05
) -> Dict:
    """
    Analyze threshold stability across different metrics.

    Parameters
    ----------
    results : pd.DataFrame
        Grid search results
    threshold : float
        Selected threshold
    tolerance : float
        Acceptable deviation

    Returns
    -------
    dict
        Stability analysis
    """
    # Find thresholds within tolerance of selected
    nearby = results[
        (results['threshold'] >= threshold - tolerance) &
        (results['threshold'] <= threshold + tolerance)
    ]

    f1_range = nearby['f1'].max() - nearby['f1'].min()
    far_range = nearby['far'].max() - nearby['far'].min()

    stable = f1_range < 0.05 and far_range < 0.05

    return {
        'threshold': threshold,
        'tolerance': tolerance,
        'f1_range': f1_range,
        'far_range': far_range,
        'is_stable': stable,
        'n_nearby': len(nearby)
    }
