"""
Compound Flood Risk Index (CFRI) Package
=========================================

A physics-informed early-warning index for predicting compound
river-coastal flooding in North Carolina estuaries.

Modules:
    io          - Data loading and validation
    preprocess  - Time alignment, datum conversion
    events      - Flood event detection
    features    - CFRI computation
    thresholding - Trigger selection
    evaluation  - Performance metrics
    plots       - Publication-quality figures
    report      - Auto-generate reports
    utils       - Logging, helpers
"""

__version__ = "1.0.0"
__author__ = "NCSEF Student Researcher"

# Utils
from .utils import (
    setup_logging,
    load_yaml_config,
    compute_percentile_rank,
    robust_zscore,
    create_train_test_split,
    detect_outliers_iqr,
    validate_datetime_index,
    summarize_missing_data
)

# I/O
from .io import (
    load_config,
    load_site_data,
    load_or_create_synthetic_data,
    save_dataframe,
    validate_loaded_data
)

# Preprocessing
from .preprocess import (
    preprocess_site,
    align_timeseries,
    handle_missing_data,
    handle_outliers,
    convert_to_mhhw,
    compute_water_level_anomaly,
    get_preprocessing_summary
)

# Events
from .events import (
    FloodEvent,
    detect_flood_state,
    create_predictive_labels,
    detect_flood_events,
    events_to_dataframe,
    compute_event_statistics,
    get_event_q_wl_lags,
    filter_events_by_period,
    get_event_windows
)

# Features
from .features import (
    normalize_features,
    estimate_lag_xcorr,
    estimate_lag_events,
    estimate_lag,
    compute_lagged_features,
    compute_overlap,
    compute_cfri,
    compute_baseline_indices,
    build_features_pipeline,
    fit_optimal_weights
)

# Thresholding
from .thresholding import (
    create_predictions,
    compute_classification_metrics,
    grid_search_threshold,
    select_threshold,
    cross_validate_threshold,
    evaluate_all_models,
    compute_threshold_stability
)

# Evaluation
from .evaluation import (
    compute_lead_times,
    compute_lead_time_statistics,
    bootstrap_metrics,
    evaluate_model_comprehensive,
    compare_models,
    compute_improvement_statistics,
    compute_skill_scores,
    generate_performance_report,
    create_comparison_table
)

# Plots
from .plots import (
    plot_event_case_study,
    plot_killer_figure,
    plot_killer_figure_simple,
    plot_timeseries_overview,
    plot_threshold_grid_search,
    plot_lag_analysis,
    plot_comparison_bars,
    create_figure_set
)

# Reports
from .report import (
    generate_abstract,
    generate_methods_section,
    generate_figure_captions,
    generate_judge_qa,
    generate_full_report,
    format_for_poster
)

__all__ = [
    # Utils
    "setup_logging",
    "load_yaml_config",
    "compute_percentile_rank",
    "robust_zscore",
    "create_train_test_split",
    "detect_outliers_iqr",
    "validate_datetime_index",
    "summarize_missing_data",
    # I/O
    "load_config",
    "load_site_data",
    "load_or_create_synthetic_data",
    "save_dataframe",
    "validate_loaded_data",
    # Preprocessing
    "preprocess_site",
    "align_timeseries",
    "handle_missing_data",
    "handle_outliers",
    "convert_to_mhhw",
    "compute_water_level_anomaly",
    "get_preprocessing_summary",
    # Events
    "FloodEvent",
    "detect_flood_state",
    "create_predictive_labels",
    "detect_flood_events",
    "events_to_dataframe",
    "compute_event_statistics",
    "get_event_q_wl_lags",
    "filter_events_by_period",
    "get_event_windows",
    # Features
    "normalize_features",
    "estimate_lag_xcorr",
    "estimate_lag_events",
    "estimate_lag",
    "compute_lagged_features",
    "compute_overlap",
    "compute_cfri",
    "compute_baseline_indices",
    "build_features_pipeline",
    "fit_optimal_weights",
    # Thresholding
    "create_predictions",
    "compute_classification_metrics",
    "grid_search_threshold",
    "select_threshold",
    "cross_validate_threshold",
    "evaluate_all_models",
    "compute_threshold_stability",
    # Evaluation
    "compute_lead_times",
    "compute_lead_time_statistics",
    "bootstrap_metrics",
    "evaluate_model_comprehensive",
    "compare_models",
    "compute_improvement_statistics",
    "compute_skill_scores",
    "generate_performance_report",
    "create_comparison_table",
    # Plots
    "plot_event_case_study",
    "plot_killer_figure",
    "plot_killer_figure_simple",
    "plot_timeseries_overview",
    "plot_threshold_grid_search",
    "plot_lag_analysis",
    "plot_comparison_bars",
    "create_figure_set",
    # Reports
    "generate_abstract",
    "generate_methods_section",
    "generate_figure_captions",
    "generate_judge_qa",
    "generate_full_report",
    "format_for_poster",
]
