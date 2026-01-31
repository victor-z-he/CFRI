#!/usr/bin/env python3
"""
CFRI-v2 Analysis and Killer Figure Generation

This script:
1. Loads the processed data
2. Computes CFRI-v2 (compound-focused formula)
3. Evaluates performance
4. Generates publication-quality killer figure

Usage:
    python scripts/run_cfri_v2_analysis.py --site wilmington
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from loguru import logger

from compound_flood.events import FloodEvent
from compound_flood.cfri_v2 import compute_cfri_v2, find_optimal_threshold_v2, compute_lead_times_v2


def load_data(site_dir: Path):
    """Load processed data and events."""
    # Load processed data
    df = pd.read_csv(site_dir / 'processed_data.csv', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Load events
    events_df = pd.read_csv(
        site_dir / 'flood_events.csv',
        parse_dates=['start_time', 'end_time', 'peak_time', 'q_peak_time']
    )

    events = []
    for _, row in events_df.iterrows():
        start_time = pd.Timestamp(row['start_time'])
        if start_time.tz is None:
            start_time = start_time.tz_localize('UTC')

        end_time = pd.Timestamp(row['end_time'])
        if end_time.tz is None:
            end_time = end_time.tz_localize('UTC')

        peak_time = pd.Timestamp(row['peak_time'])
        if peak_time.tz is None:
            peak_time = peak_time.tz_localize('UTC')

        q_peak_time = None
        if pd.notna(row.get('q_peak_time')):
            q_peak_time = pd.Timestamp(row['q_peak_time'])
            if q_peak_time.tz is None:
                q_peak_time = q_peak_time.tz_localize('UTC')

        event = FloodEvent(
            event_id=row['event_id'],
            start_time=start_time,
            end_time=end_time,
            duration_hours=row['duration_hours'],
            peak_time=peak_time,
            peak_wl=row['peak_wl'],
            peak_exceedance=row['peak_exceedance'],
            pre_event_wl=row['pre_event_wl'],
            peak_q=row.get('peak_q'),
            q_peak_time=q_peak_time
        )
        events.append(event)

    return df, events


def create_killer_figure(
    df: pd.DataFrame,
    events: list,
    cfri_threshold: float,
    wl_threshold: float,
    metrics: dict,
    lead_time_results: dict,
    site_name: str,
    save_path: Path
):
    """
    Create publication-quality killer figure for CFRI-v2.

    4-panel figure:
    A) Compound effect heatmap
    B) Early warning example (Hurricane Matthew)
    C) Lead time distribution
    D) Performance comparison (CFRI-v2 vs baselines)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # ================================================================
    # Panel A: Compound Effect Heatmap
    # ================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Create bins for Q and WL percentiles
    q_bins = [0, 25, 50, 75, 90, 100]
    wl_bins = [0, 25, 50, 75, 90, 100]

    df_temp = df.copy()
    df_temp['Q_pct_bin'] = pd.cut(
        df_temp['Q_lag'].rank(pct=True) * 100,
        bins=q_bins,
        labels=['0-25', '25-50', '50-75', '75-90', '90-100']
    )
    df_temp['WL_pct_bin'] = pd.cut(
        df_temp['WL_norm'].rank(pct=True) * 100,
        bins=wl_bins,
        labels=['0-25', '25-50', '50-75', '75-90', '90-100']
    )

    # Compute flood probability for each bin
    flood_prob = df_temp.groupby(['Q_pct_bin', 'WL_pct_bin'], observed=False)['Y'].mean().unstack() * 100

    im = ax1.imshow(flood_prob.values, cmap='YlOrRd', aspect='auto', origin='lower',
                    vmin=0, vmax=flood_prob.values.max())
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['0-25', '25-50', '50-75', '75-90', '90-100'], fontsize=10)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(['0-25', '25-50', '50-75', '75-90', '90-100'], fontsize=10)
    ax1.set_xlabel('Water Level Percentile', fontsize=11)
    ax1.set_ylabel('River Discharge Percentile (lagged)', fontsize=11)
    ax1.set_title('(A) Compound Flood Effect\nFlood probability increases when BOTH drivers are elevated',
                  fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(5):
        for j in range(5):
            val = flood_prob.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > 3 else 'black'
                ax1.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Flood Probability (%)', fontsize=10)

    # Highlight compound region
    rect = plt.Rectangle((3.5, 3.5), 1.5, 1.5, fill=False,
                         edgecolor='blue', linewidth=3, linestyle='--')
    ax1.add_patch(rect)
    ax1.text(4.25, 5.2, 'COMPOUND\nZONE', ha='center', fontsize=9,
             color='blue', fontweight='bold')

    # ================================================================
    # Panel B: Early Warning Example (Hurricane Matthew)
    # ================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Hurricane Matthew: Oct 2016
    win_start = pd.Timestamp('2016-10-06 00:00:00', tz='UTC')
    win_end = pd.Timestamp('2016-10-10 00:00:00', tz='UTC')
    plot_df = df.loc[win_start:win_end].copy()

    # Plot water level
    ax2.fill_between(plot_df.index, 0, plot_df['WL_MHHW'].clip(lower=0),
                     alpha=0.4, color='#6495ED', label='Water Level')
    ax2.axhline(y=wl_threshold, color='red', linestyle='--', linewidth=2,
                label='Flood Threshold')

    # CFRI-v2 on secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(plot_df.index, plot_df['CFRI_v2'], color='purple', linewidth=2.5,
              label='CFRI-v2')
    ax2b.axhline(y=cfri_threshold, color='orange', linestyle='--', linewidth=2,
                 label='CFRI Threshold')
    ax2b.set_ylabel('CFRI-v2', color='purple', fontsize=11)
    ax2b.set_ylim(0, 1.1)
    ax2b.tick_params(axis='y', labelcolor='purple')

    # Shade compound conditions
    compound_mask = plot_df['Both_elevated'] == 1
    ax2.fill_between(plot_df.index, -0.2, 1.5,
                     where=compound_mask, alpha=0.15, color='red',
                     label='Compound Conditions')

    # Find and annotate lead time
    # Find first flood in this window
    flood_times = plot_df[plot_df['WL_MHHW'] > wl_threshold].index
    if len(flood_times) > 0:
        first_flood = flood_times[0]

        # Find when CFRI first exceeded threshold before this
        pre_flood = plot_df.loc[:first_flood, 'CFRI_v2']
        triggers = pre_flood > cfri_threshold
        if triggers.any():
            first_trigger = triggers.idxmax()
            lead_hours = (first_flood - first_trigger).total_seconds() / 3600

            if lead_hours > 0:
                # Draw vertical lines
                ax2.axvline(x=first_trigger, color='orange', linewidth=2, alpha=0.8)
                ax2.axvline(x=first_flood, color='red', linewidth=2, alpha=0.8)

                # Draw arrow
                arrow_y = wl_threshold * 0.7
                ax2.annotate('', xy=(first_flood, arrow_y), xytext=(first_trigger, arrow_y),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))

                mid_time = first_trigger + (first_flood - first_trigger) / 2
                ax2.text(mid_time, arrow_y + 0.08, f'{lead_hours:.0f}h\nlead time',
                        ha='center', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Water Level (m above MHHW)', color='#4169E1', fontsize=11)
    ax2.set_xlabel('Date (October 2016)', fontsize=11)
    ax2.set_ylim(-0.1, 1.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.set_title('(B) Early Warning Example: Hurricane Matthew\nCFRI-v2 triggers before flood onset',
                  fontsize=12, fontweight='bold')

    # Legend
    handles = [
        mpatches.Patch(color='#6495ED', alpha=0.4, label='Water Level'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Flood Threshold'),
        Line2D([0], [0], color='purple', linewidth=2.5, label='CFRI-v2'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='CFRI Threshold'),
        mpatches.Patch(color='red', alpha=0.15, label='Compound Conditions'),
    ]
    ax2.legend(handles=handles, loc='upper left', fontsize=8, ncol=2)

    # ================================================================
    # Panel C: Lead Time Distribution
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    lead_times = lead_time_results['lead_times']
    if lead_times:
        bins = np.arange(0, max(lead_times) + 6, 6)
        ax3.hist(lead_times, bins=bins, color='#E74C3C', edgecolor='black',
                alpha=0.8, label='Lead times')

        mean_lead = np.mean(lead_times)
        median_lead = np.median(lead_times)

        ax3.axvline(mean_lead, color='black', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_lead:.1f}h')
        ax3.axvline(median_lead, color='gray', linestyle=':', linewidth=2,
                   label=f'Median: {median_lead:.1f}h')

    ax3.set_xlabel('Lead Time (hours before flood)', fontsize=11)
    ax3.set_ylabel('Number of Events', fontsize=11)
    ax3.legend(loc='upper right', fontsize=10)

    n_detected = lead_time_results['statistics']['n_detected']
    n_total = lead_time_results['statistics']['n_total']
    ax3.set_title(f'(C) CFRI-v2 Early Warning Lead Times\n{n_detected}/{n_total} events detected '
                  f'({n_detected/n_total*100:.0f}%)',
                  fontsize=12, fontweight='bold')

    # ================================================================
    # Panel D: Performance Metrics
    # ================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Compare CFRI-v2 with baselines
    # Compute baseline performance
    def compute_baseline_metrics(df, col, target='Y'):
        """Compute best F1 for a baseline."""
        valid = ~(df[col].isna() | df[target].isna())
        values = df.loc[valid, col].values
        y_true = df.loc[valid, target].values

        best_f1 = 0
        best_metrics = {}

        for T in np.linspace(0.3, 0.99, 30):
            y_pred = (values > T).astype(int)
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {'f1': f1, 'precision': precision, 'recall': recall}

        return best_metrics

    # Get metrics for each model
    models = {
        'CFRI-v2\n(Compound)': {'f1': metrics['f1'], 'precision': metrics['precision'],
                                'recall': metrics['recall']},
        'CFRI-v1\n(Original)': compute_baseline_metrics(df, 'CFRI'),
        'WL only': compute_baseline_metrics(df, 'WL_norm'),
        'Q only': compute_baseline_metrics(df, 'Q_lag'),
    }

    x = np.arange(len(models))
    width = 0.25

    f1_vals = [m['f1'] for m in models.values()]
    prec_vals = [m['precision'] for m in models.values()]
    rec_vals = [m['recall'] for m in models.values()]

    bars1 = ax4.bar(x - width, f1_vals, width, label='F1 Score', color='#2ecc71', edgecolor='black')
    bars2 = ax4.bar(x, prec_vals, width, label='Precision', color='#3498db', edgecolor='black')
    bars3 = ax4.bar(x + width, rec_vals, width, label='Recall', color='#9b59b6', edgecolor='black')

    ax4.set_xticks(x)
    ax4.set_xticklabels(models.keys(), fontsize=10)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylim(0, 1.0)
    ax4.set_title('(D) Model Performance Comparison\nCFRI-v2 outperforms single-variable approaches',
                  fontsize=12, fontweight='bold')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.02:
                ax4.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    # Highlight CFRI-v2 improvement
    improvement = (f1_vals[0] / f1_vals[1] - 1) * 100 if f1_vals[1] > 0 else 0
    ax4.annotate(f'+{improvement:.0f}%', xy=(0, f1_vals[0] + 0.05),
                ha='center', fontsize=11, fontweight='bold', color='green')

    # Overall title
    fig.suptitle(f'Compound Flood Risk Index v2 (CFRI-v2): {site_name}\n'
                 f'Physics-Informed Early Warning for Compound River-Coastal Flooding',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved killer figure: {save_path}")

    return fig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run CFRI-v2 analysis')
    parser.add_argument('--site', '-s', type=str, default='wilmington',
                       help='Site name')
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                       help='Output directory')
    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Paths
    site_dir = Path(args.output_dir) / args.site
    if not site_dir.exists():
        logger.error(f"Site directory not found: {site_dir}")
        sys.exit(1)

    # Load data
    logger.info(f"Loading data from {site_dir}")
    df, events = load_data(site_dir)

    # Get flood threshold
    wl_threshold = 0.56  # Wilmington minor flood threshold

    # Create target variable (flood in next 24 hours)
    lookahead = 24
    df['Y'] = df['WL_MHHW'].rolling(window=lookahead, min_periods=1).max().shift(-lookahead) > wl_threshold
    df['Y'] = df['Y'].astype(float)

    # Train/test split (70/30 temporal)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df)} hours, Test: {len(test_df)} hours")

    # Compute CFRI-v2
    logger.info("Computing CFRI-v2...")
    df = compute_cfri_v2(df, train_df=train_df)

    # Find optimal threshold on training data
    logger.info("Finding optimal threshold...")
    threshold_results = find_optimal_threshold_v2(
        train_df.join(df[['CFRI_v2', 'Q_elevated', 'WL_elevated', 'Both_elevated', 'Compound_term']]),
        far_limit=0.60
    )
    cfri_threshold = threshold_results['threshold']

    # Evaluate on full data
    logger.info("Evaluating performance...")
    full_results = find_optimal_threshold_v2(df, threshold_range=(cfri_threshold-0.01, cfri_threshold+0.01), n_steps=3)

    # Compute lead times
    logger.info("Computing lead times...")
    lead_time_results = compute_lead_times_v2(df, events, cfri_threshold)

    # Print summary
    print("\n" + "=" * 60)
    print("CFRI-v2 ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nOptimal Threshold: {cfri_threshold:.3f}")
    print(f"\nPerformance Metrics:")
    print(f"  F1 Score:  {full_results['f1']:.3f}")
    print(f"  Precision: {full_results['precision']:.3f}")
    print(f"  Recall:    {full_results['recall']:.3f}")
    print(f"  FAR:       {full_results['far']:.3f}")
    print(f"\nLead Time Statistics:")
    stats = lead_time_results['statistics']
    print(f"  Events detected: {stats['n_detected']}/{stats['n_total']} ({stats['detection_rate']*100:.1f}%)")
    if stats['mean_lead_hours']:
        print(f"  Mean lead time:   {stats['mean_lead_hours']:.1f} hours")
        print(f"  Median lead time: {stats['median_lead_hours']:.1f} hours")

    # Create killer figure
    logger.info("Creating killer figure...")
    fig_path = site_dir / 'cfri_v2_killer_figure.png'
    create_killer_figure(
        df=df,
        events=events,
        cfri_threshold=cfri_threshold,
        wl_threshold=wl_threshold,
        metrics=full_results,
        lead_time_results=lead_time_results,
        site_name=args.site.replace('_', ' ').title() + ', NC',
        save_path=fig_path
    )

    # Save results
    results_df = pd.DataFrame([{
        'site': args.site,
        'cfri_version': 'v2',
        'threshold': cfri_threshold,
        'f1': full_results['f1'],
        'precision': full_results['precision'],
        'recall': full_results['recall'],
        'far': full_results['far'],
        'n_events_detected': stats['n_detected'],
        'n_events_total': stats['n_total'],
        'mean_lead_hours': stats['mean_lead_hours'],
        'median_lead_hours': stats['median_lead_hours'],
    }])
    results_df.to_csv(site_dir / 'cfri_v2_results.csv', index=False)

    # Save processed data with CFRI-v2
    df.to_csv(site_dir / 'processed_data_v2.csv')

    print(f"\nOutputs saved to: {site_dir}")
    print(f"  - cfri_v2_killer_figure.png")
    print(f"  - cfri_v2_results.csv")
    print(f"  - processed_data_v2.csv")


if __name__ == '__main__':
    main()
