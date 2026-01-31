"""
Model evaluation module for CFRI performance assessment.

This module handles:
    - Lead time computation
    - Bootstrap confidence intervals
    - Comprehensive performance metrics
    - Model comparison analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .events import FloodEvent
from .thresholding import compute_classification_metrics, create_predictions


def compute_lead_times(
    df: pd.DataFrame,
    events: List[FloodEvent],
    index_col: str,
    threshold: float,
    max_lead_hours: int = 72
) -> List[Dict]:
    """
    Compute lead time for each flood event.

    Lead time = time from first trigger to flood onset.

    Parameters
    ----------
    df : pd.DataFrame
        Data with index column
    events : list
        List of FloodEvent objects
    index_col : str
        Index column to evaluate (CFRI, WL_only, etc.)
    threshold : float
        Trigger threshold
    max_lead_hours : int
        Maximum lead time to consider

    Returns
    -------
    list
        List of dicts with lead time info per event
    """
    lead_times = []

    for event in events:
        # Look for trigger in window before event
        window_start = event.start_time - pd.Timedelta(hours=max_lead_hours)
        window_end = event.start_time

        window_data = df.loc[window_start:window_end, index_col]

        if window_data.empty:
            lead_times.append({
                'event_id': event.event_id,
                'event_start': event.start_time,
                'triggered': False,
                'lead_time_hours': np.nan,
                'trigger_time': None
            })
            continue

        # Find first trigger time
        triggers = window_data > threshold
        if triggers.any():
            first_trigger = triggers.idxmax()
            lead_time = (event.start_time - first_trigger).total_seconds() / 3600

            lead_times.append({
                'event_id': event.event_id,
                'event_start': event.start_time,
                'triggered': True,
                'lead_time_hours': lead_time,
                'trigger_time': first_trigger
            })
        else:
            lead_times.append({
                'event_id': event.event_id,
                'event_start': event.start_time,
                'triggered': False,
                'lead_time_hours': 0,  # No lead time (missed or reactive)
                'trigger_time': None
            })

    return lead_times


def compute_lead_time_statistics(lead_times: List[Dict]) -> Dict:
    """
    Compute summary statistics for lead times.

    Parameters
    ----------
    lead_times : list
        List of lead time dicts from compute_lead_times

    Returns
    -------
    dict
        Summary statistics
    """
    if not lead_times:
        return {
            'n_events': 0,
            'n_triggered': 0,
            'trigger_rate': 0.0,
            'mean_lead_hours': np.nan,
            'median_lead_hours': np.nan,
            'std_lead_hours': np.nan,
            'min_lead_hours': np.nan,
            'max_lead_hours': np.nan,
            'pct_6h_plus': 0.0,
            'pct_12h_plus': 0.0,
            'pct_24h_plus': 0.0
        }

    df = pd.DataFrame(lead_times)
    triggered = df[df['triggered']]
    lead_values = triggered['lead_time_hours'].dropna()

    n_events = len(df)
    n_triggered = len(triggered)

    stats = {
        'n_events': n_events,
        'n_triggered': n_triggered,
        'trigger_rate': n_triggered / n_events if n_events > 0 else 0.0
    }

    if len(lead_values) > 0:
        stats.update({
            'mean_lead_hours': lead_values.mean(),
            'median_lead_hours': lead_values.median(),
            'std_lead_hours': lead_values.std(),
            'min_lead_hours': lead_values.min(),
            'max_lead_hours': lead_values.max(),
            'pct_6h_plus': 100 * (lead_values >= 6).sum() / len(lead_values),
            'pct_12h_plus': 100 * (lead_values >= 12).sum() / len(lead_values),
            'pct_24h_plus': 100 * (lead_values >= 24).sum() / len(lead_values)
        })
    else:
        stats.update({
            'mean_lead_hours': np.nan,
            'median_lead_hours': np.nan,
            'std_lead_hours': np.nan,
            'min_lead_hours': np.nan,
            'max_lead_hours': np.nan,
            'pct_6h_plus': 0.0,
            'pct_12h_plus': 0.0,
            'pct_24h_plus': 0.0
        })

    return stats


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict:
    """
    Compute bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Metrics with confidence intervals
    """
    # Remove NaN
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = len(y_true)
    if n < 10:
        logger.warning("Insufficient data for bootstrap CI")
        base_metrics = compute_classification_metrics(y_true, y_pred)
        return {k: {'value': v, 'ci_low': np.nan, 'ci_high': np.nan}
                for k, v in base_metrics.items()}

    np.random.seed(random_state)

    # Store bootstrap samples
    boot_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'far': [],
        'accuracy': []
    }

    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        metrics = compute_classification_metrics(y_true_boot, y_pred_boot)

        for key in boot_metrics:
            boot_metrics[key].append(metrics[key])

    # Compute confidence intervals
    alpha = 1 - confidence_level
    ci_low = alpha / 2
    ci_high = 1 - alpha / 2

    results = {}
    base_metrics = compute_classification_metrics(y_true, y_pred)

    for key in boot_metrics:
        values = np.array(boot_metrics[key])
        results[key] = {
            'value': base_metrics[key],
            'ci_low': np.nanpercentile(values, 100 * ci_low),
            'ci_high': np.nanpercentile(values, 100 * ci_high),
            'std': np.nanstd(values)
        }

    # Add confusion matrix values (no CI)
    for key in ['tp', 'fp', 'tn', 'fn']:
        results[key] = {'value': base_metrics[key]}

    return results


def evaluate_model_comprehensive(
    df: pd.DataFrame,
    events: List[FloodEvent],
    index_col: str,
    threshold: float,
    target_col: str = 'Y',
    n_bootstrap: int = 1000
) -> Dict:
    """
    Comprehensive model evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Data with index and target columns
    events : list
        List of FloodEvent objects
    index_col : str
        Index column to evaluate
    threshold : float
        Trigger threshold
    target_col : str
        Target column
    n_bootstrap : int
        Number of bootstrap samples for CI

    Returns
    -------
    dict
        Comprehensive evaluation results
    """
    # Create predictions
    y_pred = create_predictions(df, index_col, threshold)
    y_true = df[target_col]

    # Classification metrics with bootstrap CI
    boot_results = bootstrap_metrics(
        y_true.values, y_pred.values,
        n_bootstrap=n_bootstrap
    )

    # Lead time analysis
    lead_times = compute_lead_times(df, events, index_col, threshold)
    lead_stats = compute_lead_time_statistics(lead_times)

    # Compile results
    results = {
        'index': index_col,
        'threshold': threshold,
        'classification': boot_results,
        'lead_time': lead_stats,
        'n_samples': int((~df[target_col].isna()).sum()),
        'n_events': len(events),
        'prevalence': float(df[target_col].mean())
    }

    return results


def compare_models(
    df: pd.DataFrame,
    events: List[FloodEvent],
    model_configs: List[Dict],
    target_col: str = 'Y'
) -> pd.DataFrame:
    """
    Compare multiple models/indices.

    Parameters
    ----------
    df : pd.DataFrame
        Data with all index columns
    events : list
        List of FloodEvent objects
    model_configs : list
        List of dicts with 'name', 'index_col', 'threshold'
    target_col : str
        Target column

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    results = []

    for config in model_configs:
        name = config['name']
        index_col = config['index_col']
        threshold = config['threshold']

        if index_col not in df.columns:
            logger.warning(f"Index column {index_col} not found. Skipping {name}.")
            continue

        # Evaluate
        eval_results = evaluate_model_comprehensive(
            df, events, index_col, threshold, target_col,
            n_bootstrap=500  # Faster for comparison
        )

        # Extract key metrics
        row = {
            'Model': name,
            'Threshold': threshold,
            'F1': eval_results['classification']['f1']['value'],
            'F1_CI': f"({eval_results['classification']['f1']['ci_low']:.3f}, "
                     f"{eval_results['classification']['f1']['ci_high']:.3f})",
            'Precision': eval_results['classification']['precision']['value'],
            'Recall': eval_results['classification']['recall']['value'],
            'FAR': eval_results['classification']['far']['value'],
            'Mean_Lead_Hours': eval_results['lead_time']['mean_lead_hours'],
            'Median_Lead_Hours': eval_results['lead_time']['median_lead_hours'],
            'Trigger_Rate': eval_results['lead_time']['trigger_rate']
        }
        results.append(row)

    return pd.DataFrame(results)


def compute_improvement_statistics(
    baseline_metrics: Dict,
    cfri_metrics: Dict
) -> Dict:
    """
    Compute relative improvement of CFRI over baseline.

    Parameters
    ----------
    baseline_metrics : dict
        Baseline model metrics
    cfri_metrics : dict
        CFRI metrics

    Returns
    -------
    dict
        Improvement statistics
    """
    improvements = {}

    for metric in ['f1', 'precision', 'recall', 'accuracy']:
        baseline_val = baseline_metrics['classification'][metric]['value']
        cfri_val = cfri_metrics['classification'][metric]['value']

        if baseline_val > 0:
            rel_improvement = 100 * (cfri_val - baseline_val) / baseline_val
        else:
            rel_improvement = np.nan

        improvements[metric] = {
            'baseline': baseline_val,
            'cfri': cfri_val,
            'absolute_diff': cfri_val - baseline_val,
            'relative_pct': rel_improvement
        }

    # FAR should decrease
    baseline_far = baseline_metrics['classification']['far']['value']
    cfri_far = cfri_metrics['classification']['far']['value']

    if baseline_far > 0:
        far_reduction = 100 * (baseline_far - cfri_far) / baseline_far
    else:
        far_reduction = 0.0

    improvements['far'] = {
        'baseline': baseline_far,
        'cfri': cfri_far,
        'absolute_diff': cfri_far - baseline_far,
        'reduction_pct': far_reduction
    }

    # Lead time improvement
    baseline_lead = baseline_metrics['lead_time']['mean_lead_hours']
    cfri_lead = cfri_metrics['lead_time']['mean_lead_hours']

    improvements['lead_time'] = {
        'baseline_hours': baseline_lead,
        'cfri_hours': cfri_lead,
        'improvement_hours': cfri_lead - baseline_lead if not np.isnan(cfri_lead) else np.nan
    }

    return improvements


def compute_skill_scores(
    df: pd.DataFrame,
    index_col: str,
    threshold: float,
    target_col: str = 'Y'
) -> Dict:
    """
    Compute skill scores relative to climatology.

    Parameters
    ----------
    df : pd.DataFrame
        Data with index and target
    index_col : str
        Index column
    threshold : float
        Trigger threshold
    target_col : str
        Target column

    Returns
    -------
    dict
        Skill scores
    """
    y_pred = create_predictions(df, index_col, threshold)
    y_true = df[target_col].values
    y_pred = y_pred.values

    # Remove NaN
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {'hss': np.nan, 'pss': np.nan, 'bss': np.nan}

    # Confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    n = len(y_true)

    # Climatology (base rate)
    p = y_true.mean()  # Probability of positive class
    q = 1 - p

    # Expected hits by chance
    expected_correct = (tp + fp) * (tp + fn) / n + (tn + fn) * (tn + fp) / n

    # Heidke Skill Score (HSS)
    observed_correct = tp + tn
    if n != expected_correct:
        hss = (observed_correct - expected_correct) / (n - expected_correct)
    else:
        hss = 0.0

    # Peirce Skill Score (PSS) = Hit Rate - False Alarm Rate
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    pss = hit_rate - false_alarm_rate

    # Brier Skill Score (BSS) - simplified version
    # Using 0/1 predictions as probabilities
    brier_score = np.mean((y_pred - y_true) ** 2)
    brier_reference = np.mean((p - y_true) ** 2)  # Climatology forecast

    if brier_reference > 0:
        bss = 1 - brier_score / brier_reference
    else:
        bss = 0.0

    return {
        'hss': hss,
        'pss': pss,
        'bss': bss,
        'hit_rate': hit_rate,
        'false_alarm_rate': false_alarm_rate,
        'climatology': p
    }


def generate_performance_report(
    results: Dict,
    site_name: str
) -> str:
    """
    Generate human-readable performance report.

    Parameters
    ----------
    results : dict
        Evaluation results
    site_name : str
        Site name

    Returns
    -------
    str
        Formatted report
    """
    lines = [
        f"Performance Report: {site_name}",
        "=" * 60,
        "",
        f"Model: {results['index']}",
        f"Threshold: {results['threshold']:.3f}",
        f"Sample Size: {results['n_samples']:,}",
        f"Number of Events: {results['n_events']}",
        f"Flood Prevalence: {100*results['prevalence']:.1f}%",
        "",
        "Classification Performance:",
        "-" * 40
    ]

    metrics = results['classification']
    for metric in ['f1', 'precision', 'recall', 'far', 'accuracy']:
        if metric in metrics:
            m = metrics[metric]
            ci_str = ""
            if 'ci_low' in m and not np.isnan(m.get('ci_low', np.nan)):
                ci_str = f" (95% CI: {m['ci_low']:.3f} - {m['ci_high']:.3f})"
            lines.append(f"  {metric.upper():12s}: {m['value']:.3f}{ci_str}")

    # Confusion matrix
    lines.extend([
        "",
        "Confusion Matrix:",
        f"  TP: {metrics['tp']['value']:>6}  FP: {metrics['fp']['value']:>6}",
        f"  FN: {metrics['fn']['value']:>6}  TN: {metrics['tn']['value']:>6}"
    ])

    # Lead time
    lt = results['lead_time']
    lines.extend([
        "",
        "Lead Time Performance:",
        "-" * 40,
        f"  Events Triggered: {lt['n_triggered']} / {lt['n_events']} "
        f"({100*lt['trigger_rate']:.1f}%)",
        f"  Mean Lead Time: {lt['mean_lead_hours']:.1f} hours",
        f"  Median Lead Time: {lt['median_lead_hours']:.1f} hours",
        f"  Events with >6h Lead: {lt['pct_6h_plus']:.1f}%",
        f"  Events with >12h Lead: {lt['pct_12h_plus']:.1f}%",
        f"  Events with >24h Lead: {lt['pct_24h_plus']:.1f}%"
    ])

    return "\n".join(lines)


def create_comparison_table(
    comparison_df: pd.DataFrame,
    format: str = 'markdown'
) -> str:
    """
    Create formatted comparison table.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison results
    format : str
        Output format ('markdown' or 'latex')

    Returns
    -------
    str
        Formatted table
    """
    if format == 'latex':
        # LaTeX table
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Model Comparison}",
            "\\begin{tabular}{lcccccc}",
            "\\hline",
            "Model & F1 & Precision & Recall & FAR & Lead (h) \\\\",
            "\\hline"
        ]

        for _, row in comparison_df.iterrows():
            lines.append(
                f"{row['Model']} & {row['F1']:.3f} & {row['Precision']:.3f} & "
                f"{row['Recall']:.3f} & {row['FAR']:.3f} & "
                f"{row['Mean_Lead_Hours']:.1f} \\\\"
            )

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])

    else:  # Markdown
        lines = [
            "| Model | F1 | Precision | Recall | FAR | Mean Lead (h) |",
            "|-------|-----|-----------|--------|-----|---------------|"
        ]

        for _, row in comparison_df.iterrows():
            lead = row['Mean_Lead_Hours']
            lead_str = f"{lead:.1f}" if not np.isnan(lead) else "N/A"
            lines.append(
                f"| {row['Model']} | {row['F1']:.3f} | {row['Precision']:.3f} | "
                f"{row['Recall']:.3f} | {row['FAR']:.3f} | {lead_str} |"
            )

    return "\n".join(lines)
