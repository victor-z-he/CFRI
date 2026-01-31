#!/usr/bin/env python3
"""
Main pipeline script for CFRI analysis.

This script runs the complete analysis pipeline:
1. Load and preprocess data
2. Detect flood events
3. Estimate river-to-estuary lag
4. Build CFRI features
5. Select optimal threshold
6. Evaluate performance
7. Generate figures and reports

Usage:
    python scripts/run_pipeline.py --config configs/ncsef.yaml --site wilmington
    python scripts/run_pipeline.py --config configs/ncsef.yaml --site all
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from loguru import logger

from compound_flood import (
    setup_logging,
    load_config,
    load_site_data,
    load_or_create_synthetic_data,
    preprocess_site,
    detect_flood_state,
    create_predictive_labels,
    detect_flood_events,
    compute_event_statistics,
    events_to_dataframe,
    normalize_features,
    estimate_lag,
    compute_lagged_features,
    compute_overlap,
    compute_cfri,
    compute_baseline_indices,
    select_threshold,
    evaluate_all_models,
    evaluate_model_comprehensive,
    compare_models,
    create_figure_set,
    generate_full_report,
)


def run_site_analysis(
    site_name: str,
    site_config: dict,
    config: dict,
    base_path: Path,
    output_dir: Path,
    use_synthetic: bool = False
) -> dict:
    """
    Run complete analysis for a single site.

    Parameters
    ----------
    site_name : str
        Site identifier
    site_config : dict
        Site-specific configuration
    config : dict
        Global configuration
    base_path : Path
        Base path for data files
    output_dir : Path
        Output directory
    use_synthetic : bool
        If True, generate synthetic data if real data not found

    Returns
    -------
    dict
        Analysis results
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting analysis for {site_name}")
    logger.info(f"=" * 60)

    results = {'site': site_name}

    # Create site output directory
    site_output = output_dir / site_name
    site_output.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("Step 1: Loading data...")

    try:
        if use_synthetic:
            river_df, tide_df, load_meta = load_or_create_synthetic_data(
                site_config, base_path
            )
        else:
            river_df, tide_df, load_meta = load_site_data(
                site_config, base_path
            )
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.info("Run with --synthetic flag to generate synthetic data for testing")
        return results

    results['load_metadata'] = load_meta

    # =========================================================================
    # Step 2: Preprocess Data
    # =========================================================================
    logger.info("Step 2: Preprocessing data...")

    preprocess_config = config.get('preprocessing', {})
    merged_df, preprocess_stats = preprocess_site(
        river_df, tide_df,
        site_config,
        preprocess_config
    )

    results['preprocess_stats'] = preprocess_stats

    # =========================================================================
    # Step 3: Detect Flood State and Events
    # =========================================================================
    logger.info("Step 3: Detecting flood events...")

    flood_threshold = site_config['flood_thresholds']['minor']
    merged_df = detect_flood_state(merged_df, flood_threshold)
    merged_df.attrs['flood_threshold'] = flood_threshold

    # Create predictive labels
    lookahead = config.get('features', {}).get('lookahead_hours', 24)
    merged_df = create_predictive_labels(merged_df, lookahead_hours=lookahead)

    # Detect events
    event_config = config.get('events', {})
    events = detect_flood_events(merged_df, event_config)
    event_stats = compute_event_statistics(events)

    results['events'] = events
    results['event_stats'] = event_stats

    # Save event catalog
    event_df = events_to_dataframe(events)
    event_df.to_csv(site_output / 'flood_events.csv', index=False)
    logger.info(f"Saved {len(events)} flood events")

    # =========================================================================
    # Step 4: Train/Test Split
    # =========================================================================
    logger.info("Step 4: Creating train/test split...")

    thresh_config = config.get('thresholding', {})
    train_frac = thresh_config.get('train_fraction', 0.70)

    years = merged_df.index.year.unique()
    n_train_years = int(len(years) * train_frac)
    train_years = years[:n_train_years]
    test_years = years[n_train_years:]

    train_df = merged_df[merged_df.index.year.isin(train_years)].copy()
    test_df = merged_df[merged_df.index.year.isin(test_years)].copy()

    logger.info(f"Train: {train_years[0]}-{train_years[-1]} ({len(train_df)} records)")
    logger.info(f"Test: {test_years[0]}-{test_years[-1]} ({len(test_df)} records)")

    # =========================================================================
    # Step 5: Estimate Lag
    # =========================================================================
    logger.info("Step 5: Estimating river-to-estuary lag...")

    # Use only training data for lag estimation
    train_events = [e for e in events if e.start_time.year in train_years]
    lag_results = estimate_lag(train_df, train_events, config)

    optimal_lag = lag_results['optimal_lag']
    results['lag_results'] = lag_results

    # =========================================================================
    # Step 6: Build Features
    # =========================================================================
    logger.info("Step 6: Building CFRI features...")

    feat_config = config.get('features', {})
    cfri_config = feat_config.get('cfri', {})

    # Normalize using training data statistics
    norm_method = feat_config.get('normalization', {}).get('method', 'percentile')
    robust = feat_config.get('normalization', {}).get('robust_zscore', False)

    # Apply to full dataset using train stats
    merged_df = normalize_features(
        merged_df, train_df,
        method=norm_method,
        robust=robust
    )

    # Compute lagged features
    merged_df = compute_lagged_features(merged_df, optimal_lag)

    # Compute overlap
    overlap_method = cfri_config.get('overlap_method', 'multiplicative')
    overlap_thresh = cfri_config.get('overlap_thresholds', {})
    merged_df = compute_overlap(merged_df, method=overlap_method, thresholds=overlap_thresh)

    # Compute CFRI
    weights = cfri_config.get('default_weights', None)
    merged_df = compute_cfri(merged_df, weights=weights)

    # Compute baselines
    merged_df = compute_baseline_indices(merged_df)

    # =========================================================================
    # Step 7: Select Optimal Threshold
    # =========================================================================
    logger.info("Step 7: Selecting optimal CFRI threshold...")

    # Re-split after feature engineering
    train_df = merged_df[merged_df.index.year.isin(train_years)].copy()
    test_df = merged_df[merged_df.index.year.isin(test_years)].copy()

    threshold_results = select_threshold(train_df, config)
    optimal_threshold = threshold_results['optimal_threshold']

    results['threshold_results'] = threshold_results

    # =========================================================================
    # Step 8: Evaluate on Test Data
    # =========================================================================
    logger.info("Step 8: Evaluating on test data...")

    test_events = [e for e in events if e.start_time.year in test_years]

    # Comprehensive CFRI evaluation
    cfri_eval = evaluate_model_comprehensive(
        test_df, test_events, 'CFRI', optimal_threshold
    )

    results['evaluation'] = cfri_eval

    # Compare models
    model_configs = [
        {'name': 'CFRI', 'index_col': 'CFRI', 'threshold': optimal_threshold},
        {'name': 'WL_only', 'index_col': 'WL_only', 'threshold': 0.90},  # Will be optimized
        {'name': 'Q_only', 'index_col': 'Q_only', 'threshold': 0.90},
    ]

    comparison_df = compare_models(test_df, test_events, model_configs)
    comparison_df.to_csv(site_output / 'model_comparison.csv', index=False)

    results['comparison'] = comparison_df

    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())

    # =========================================================================
    # Step 9: Generate Figures
    # =========================================================================
    logger.info("Step 9: Generating figures...")

    figures_dir = site_output / 'figures'
    figure_paths = create_figure_set(
        merged_df, events,
        threshold_results, comparison_df,
        site_name, figures_dir
    )

    results['figures'] = figure_paths

    # =========================================================================
    # Step 10: Generate Reports
    # =========================================================================
    logger.info("Step 10: Generating NCSEF report...")

    reports_dir = site_output / 'reports'
    generate_full_report(
        site_name, config,
        preprocess_stats, lag_results,
        threshold_results, cfri_eval,
        comparison_df, reports_dir
    )

    # =========================================================================
    # Save Processed Data
    # =========================================================================
    logger.info("Saving processed data...")

    # Save full processed dataset
    merged_df.to_csv(site_output / 'processed_data.csv')

    # Save results summary
    summary = {
        'site': site_name,
        'n_records': len(merged_df),
        'n_events': len(events),
        'optimal_lag_hours': optimal_lag,
        'optimal_threshold': optimal_threshold,
        'train_years': f"{train_years[0]}-{train_years[-1]}",
        'test_years': f"{test_years[0]}-{test_years[-1]}",
        'test_f1': cfri_eval['classification']['f1']['value'],
        'test_precision': cfri_eval['classification']['precision']['value'],
        'test_recall': cfri_eval['classification']['recall']['value'],
        'test_far': cfri_eval['classification']['far']['value'],
        'mean_lead_hours': cfri_eval['lead_time']['mean_lead_hours'],
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(site_output / 'analysis_summary.csv', index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Analysis complete for {site_name}")
    logger.info(f"Results saved to: {site_output}")
    logger.info(f"{'=' * 60}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run CFRI analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config configs/ncsef.yaml --site wilmington
  python run_pipeline.py --config configs/ncsef.yaml --site all --synthetic
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--site', '-s',
        type=str,
        default='all',
        help='Site name to analyze (or "all" for all sites)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory'
    )

    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic data if real data not found'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level=log_level)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    base_path = config_path.parent.parent  # Assume configs/ is one level down

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine sites to analyze
    if args.site.lower() == 'all':
        sites = list(config['sites'].keys())
    else:
        if args.site not in config['sites']:
            logger.error(f"Site '{args.site}' not found in configuration")
            logger.info(f"Available sites: {list(config['sites'].keys())}")
            sys.exit(1)
        sites = [args.site]

    # Run analysis for each site
    all_results = {}

    for site_name in sites:
        site_config = config['sites'][site_name]

        try:
            results = run_site_analysis(
                site_name,
                site_config,
                config,
                base_path,
                output_dir,
                use_synthetic=args.synthetic
            )
            all_results[site_name] = results

        except Exception as e:
            logger.error(f"Error analyzing {site_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    for site_name, results in all_results.items():
        if 'evaluation' in results:
            eval_results = results['evaluation']
            logger.info(f"\n{site_name}:")
            logger.info(f"  F1 Score: {eval_results['classification']['f1']['value']:.3f}")
            logger.info(f"  Mean Lead Time: {eval_results['lead_time']['mean_lead_hours']:.1f} hours")


if __name__ == '__main__':
    main()
