"""
Visualization module for CFRI analysis.

This module handles:
    - Publication-quality figures
    - Event case study plots
    - Performance comparison charts
    - The "killer figure" for NCSEF
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from loguru import logger

from .events import FloodEvent


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})


def plot_event_case_study(
    df: pd.DataFrame,
    event: FloodEvent,
    threshold: float,
    cfri_threshold: float,
    pre_hours: int = 48,
    post_hours: int = 24,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create case study plot for a single flood event.

    Shows Q, WL, CFRI with threshold crossings highlighted.

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q, WL_MHHW, CFRI columns
    event : FloodEvent
        Flood event to plot
    threshold : float
        Flood threshold (WL)
    cfri_threshold : float
        CFRI trigger threshold
    pre_hours : int
        Hours before event to show
    post_hours : int
        Hours after event to show
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    # Extract event window
    win_start = event.start_time - pd.Timedelta(hours=pre_hours)
    win_end = event.end_time + pd.Timedelta(hours=post_hours)
    event_data = df.loc[win_start:win_end].copy()

    if len(event_data) == 0:
        logger.warning(f"No data in event window for event {event.event_id}")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Panel 1: River Discharge
    ax1 = axes[0]
    ax1.plot(event_data.index, event_data['Q'], 'b-', linewidth=1.5, label='Discharge (Q)')
    ax1.axvspan(event.start_time, event.end_time, alpha=0.2, color='red', label='Flood Period')
    ax1.set_ylabel('Discharge (m³/s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Event {event.event_id}: {event.start_time.strftime("%Y-%m-%d %H:%M")}')

    # Panel 2: Water Level
    ax2 = axes[1]
    ax2.plot(event_data.index, event_data['WL_MHHW'], 'g-', linewidth=1.5, label='Water Level')
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Flood Threshold ({threshold:.2f}m)')
    ax2.axvspan(event.start_time, event.end_time, alpha=0.2, color='red')
    ax2.set_ylabel('WL above MHHW (m)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    # Panel 3: CFRI with Threshold
    ax3 = axes[2]
    ax3.plot(event_data.index, event_data['CFRI'], 'purple', linewidth=1.5, label='CFRI')
    ax3.axhline(y=cfri_threshold, color='orange', linestyle='--', linewidth=1,
                label=f'CFRI Threshold ({cfri_threshold:.2f})')
    ax3.axvspan(event.start_time, event.end_time, alpha=0.2, color='red')

    # Find and mark trigger time
    pre_event = event_data.loc[:event.start_time, 'CFRI']
    triggers = pre_event > cfri_threshold
    if triggers.any():
        first_trigger = triggers.idxmax()
        lead_time = (event.start_time - first_trigger).total_seconds() / 3600
        ax3.axvline(x=first_trigger, color='orange', linestyle='-', linewidth=2, alpha=0.7)
        ax3.annotate(f'Lead: {lead_time:.1f}h',
                     xy=(first_trigger, cfri_threshold),
                     xytext=(first_trigger, cfri_threshold + 0.15),
                     fontsize=9, ha='center',
                     arrowprops=dict(arrowstyle='->', color='orange'))

    ax3.set_ylabel('CFRI', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_xlabel('Date/Time (UTC)')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 1.1)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")

    return fig


def plot_killer_figure(
    df: pd.DataFrame,
    events: List[FloodEvent],
    cfri_threshold: float,
    wl_threshold: float,
    comparison_results: pd.DataFrame,
    site_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create the "killer figure" for NCSEF presentation.

    Multi-panel figure showing:
    - Top left: Event case study with lead time
    - Top right: Performance comparison bar chart
    - Bottom left: Lead time histogram
    - Bottom right: ROC-like curve or threshold sensitivity

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    events : list
        List of FloodEvent objects
    cfri_threshold : float
        Optimal CFRI threshold
    wl_threshold : float
        Flood threshold
    comparison_results : pd.DataFrame
        Model comparison results
    site_name : str
        Site name for title
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # === Panel A: Example Event with Lead Time ===
    ax_event = fig.add_subplot(gs[0, 0])

    # Select a representative event (one with good lead time)
    best_event = None
    best_lead = 0

    for event in events:
        win_start = event.start_time - pd.Timedelta(hours=48)
        pre_event = df.loc[win_start:event.start_time, 'CFRI']
        triggers = pre_event > cfri_threshold
        if triggers.any():
            first_trigger = triggers.idxmax()
            lead = (event.start_time - first_trigger).total_seconds() / 3600
            if lead > best_lead and lead < 72:
                best_lead = lead
                best_event = event

    if best_event is None and len(events) > 0:
        best_event = events[0]

    if best_event:
        win_start = best_event.start_time - pd.Timedelta(hours=48)
        win_end = best_event.end_time + pd.Timedelta(hours=12)
        event_data = df.loc[win_start:win_end]

        # Plot WL and CFRI
        ax_event.fill_between(event_data.index, 0, event_data['WL_MHHW'].clip(lower=0),
                              alpha=0.3, color='blue', label='WL > MHHW')
        ax_event.axhline(y=wl_threshold, color='red', linestyle='--', linewidth=1,
                         label=f'Flood Thresh.')

        ax2 = ax_event.twinx()
        ax2.plot(event_data.index, event_data['CFRI'], 'purple', linewidth=2, label='CFRI')
        ax2.axhline(y=cfri_threshold, color='orange', linestyle='--', linewidth=1)
        ax2.set_ylabel('CFRI', color='purple')
        ax2.set_ylim(0, 1.1)

        # Mark trigger and lead time
        pre_event = event_data.loc[:best_event.start_time, 'CFRI']
        triggers = pre_event > cfri_threshold
        if triggers.any():
            first_trigger = triggers.idxmax()
            lead_time = (best_event.start_time - first_trigger).total_seconds() / 3600
            ax_event.axvline(x=first_trigger, color='orange', linewidth=2, alpha=0.8)
            ax_event.axvline(x=best_event.start_time, color='red', linewidth=2, alpha=0.8)

            # Add arrow showing lead time
            mid_time = first_trigger + (best_event.start_time - first_trigger) / 2
            ax_event.annotate('', xy=(best_event.start_time, wl_threshold * 0.8),
                              xytext=(first_trigger, wl_threshold * 0.8),
                              arrowprops=dict(arrowstyle='<->', color='black', lw=2))
            ax_event.text(mid_time, wl_threshold * 0.9, f'{lead_time:.0f}h lead',
                          ha='center', fontsize=10, fontweight='bold')

        ax_event.set_ylabel('Water Level (m above MHHW)', color='blue')
        ax_event.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_event.set_xlabel('Date')
        ax_event.set_title('(A) Early Warning: CFRI Trigger Before Flood Onset',
                           fontsize=11, fontweight='bold')

        # Custom legend
        handles = [
            mpatches.Patch(color='blue', alpha=0.3, label='Water Level'),
            Line2D([0], [0], color='red', linestyle='--', label='Flood Threshold'),
            Line2D([0], [0], color='purple', linewidth=2, label='CFRI'),
            Line2D([0], [0], color='orange', linestyle='--', label='CFRI Threshold')
        ]
        ax_event.legend(handles=handles, loc='upper left', fontsize=8)

    # === Panel B: Performance Comparison ===
    ax_perf = fig.add_subplot(gs[0, 1])

    if len(comparison_results) > 0:
        models = comparison_results['Model'].values
        x = np.arange(len(models))
        width = 0.25

        f1_vals = comparison_results['F1'].values
        prec_vals = comparison_results['Precision'].values
        rec_vals = comparison_results['Recall'].values

        bars1 = ax_perf.bar(x - width, f1_vals, width, label='F1 Score', color='#2ecc71')
        bars2 = ax_perf.bar(x, prec_vals, width, label='Precision', color='#3498db')
        bars3 = ax_perf.bar(x + width, rec_vals, width, label='Recall', color='#9b59b6')

        ax_perf.set_ylabel('Score')
        ax_perf.set_xticks(x)
        ax_perf.set_xticklabels(models, rotation=15, ha='right')
        ax_perf.legend(loc='upper right')
        ax_perf.set_ylim(0, 1.1)
        ax_perf.set_title('(B) Model Performance Comparison', fontsize=11, fontweight='bold')

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax_perf.annotate(f'{height:.2f}',
                                     xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 3), textcoords="offset points",
                                     ha='center', va='bottom', fontsize=7)

    # === Panel C: Lead Time Distribution ===
    ax_lead = fig.add_subplot(gs[1, 0])

    lead_times = []
    for event in events:
        win_start = event.start_time - pd.Timedelta(hours=72)
        pre_event = df.loc[win_start:event.start_time, 'CFRI']
        triggers = pre_event > cfri_threshold
        if triggers.any():
            first_trigger = triggers.idxmax()
            lead = (event.start_time - first_trigger).total_seconds() / 3600
            if lead > 0 and lead < 72:
                lead_times.append(lead)

    if lead_times:
        bins = np.arange(0, max(lead_times) + 6, 6)
        ax_lead.hist(lead_times, bins=bins, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax_lead.axvline(np.mean(lead_times), color='black', linestyle='--',
                        linewidth=2, label=f'Mean: {np.mean(lead_times):.1f}h')
        ax_lead.axvline(np.median(lead_times), color='gray', linestyle=':',
                        linewidth=2, label=f'Median: {np.median(lead_times):.1f}h')
        ax_lead.set_xlabel('Lead Time (hours)')
        ax_lead.set_ylabel('Number of Events')
        ax_lead.legend()
        ax_lead.set_title('(C) CFRI Early Warning Lead Time Distribution',
                          fontsize=11, fontweight='bold')

    # === Panel D: Threshold Sensitivity / FAR-Recall Tradeoff ===
    ax_roc = fig.add_subplot(gs[1, 1])

    # Compute FAR and Recall for different thresholds
    thresholds = np.arange(0.2, 0.95, 0.02)
    fars = []
    recalls = []

    for T in thresholds:
        pred = (df['CFRI'] > T).astype(int)
        true = df['Y'].dropna()
        pred = pred.loc[true.index]

        valid = ~pred.isna() & ~true.isna()
        y_true = true[valid].values
        y_pred = pred[valid].values

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0

        recalls.append(recall)
        fars.append(far)

    ax_roc.plot(fars, recalls, 'b-', linewidth=2)

    # Mark optimal threshold
    opt_pred = (df['CFRI'] > cfri_threshold).astype(int)
    opt_true = df['Y'].dropna()
    opt_pred = opt_pred.loc[opt_true.index]
    valid = ~opt_pred.isna() & ~opt_true.isna()
    y_true = opt_true[valid].values
    y_pred = opt_pred[valid].values

    opt_tp = ((y_true == 1) & (y_pred == 1)).sum()
    opt_fp = ((y_true == 0) & (y_pred == 1)).sum()
    opt_fn = ((y_true == 1) & (y_pred == 0)).sum()
    opt_recall = opt_tp / (opt_tp + opt_fn) if (opt_tp + opt_fn) > 0 else 0
    opt_far = opt_fp / (opt_tp + opt_fp) if (opt_tp + opt_fp) > 0 else 0

    ax_roc.scatter([opt_far], [opt_recall], color='red', s=100, zorder=5,
                   label=f'Optimal T={cfri_threshold:.2f}')
    ax_roc.annotate(f'T={cfri_threshold:.2f}', xy=(opt_far, opt_recall),
                    xytext=(opt_far + 0.05, opt_recall - 0.05), fontsize=9)

    ax_roc.set_xlabel('False Alarm Ratio (FAR)')
    ax_roc.set_ylabel('Recall (Hit Rate)')
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.legend()
    ax_roc.set_title('(D) CFRI Threshold Sensitivity', fontsize=11, fontweight='bold')
    ax_roc.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Compound Flood Risk Index (CFRI) Performance: {site_name}',
                 fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved killer figure: {save_path}")

    return fig


def plot_killer_figure_simple(
    df: pd.DataFrame,
    events: List[FloodEvent],
    cfri_threshold: float,
    wl_threshold: float,
    site_name: str,
    event_name: str = None,
    event_date_range: Tuple[str, str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a simplified 2-panel "killer figure" for presentations.

    Panel layout:
    - (A) Early Warning Example: specific event showing WL, CFRI, and lead time
    - (B) Lead Time Distribution: histogram of all event lead times

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with CFRI, WL_MHHW columns
    events : list
        List of FloodEvent objects
    cfri_threshold : float
        Optimal CFRI threshold
    wl_threshold : float
        Flood threshold (water level)
    site_name : str
        Site name for title (e.g., "Wilmington, NC")
    event_name : str, optional
        Name of the example event (e.g., "Hurricane Floyd")
    event_date_range : tuple, optional
        (start_date, end_date) strings for the example event window
        e.g., ("1999-09-14", "1999-09-18")
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Panel A: Early Warning Example ===
    ax_event = axes[0]

    # Find the best event to display, or use specified date range
    if event_date_range:
        win_start = pd.Timestamp(event_date_range[0], tz='UTC')
        win_end = pd.Timestamp(event_date_range[1], tz='UTC')
        event_data = df.loc[win_start:win_end].copy()
        # Find the corresponding event
        selected_event = None
        for event in events:
            if win_start <= event.start_time <= win_end:
                selected_event = event
                break
    else:
        # Auto-select best event (good lead time, representative)
        best_event = None
        best_lead = 0

        for event in events:
            pre_start = event.start_time - pd.Timedelta(hours=72)
            pre_event = df.loc[pre_start:event.start_time, 'CFRI']
            triggers = pre_event > cfri_threshold
            if triggers.any():
                first_trigger = triggers.idxmax()
                lead = (event.start_time - first_trigger).total_seconds() / 3600
                if lead > best_lead and lead < 72:
                    best_lead = lead
                    best_event = event

        if best_event is None and len(events) > 0:
            best_event = events[0]

        selected_event = best_event
        if selected_event:
            win_start = selected_event.start_time - pd.Timedelta(hours=48)
            win_end = selected_event.end_time + pd.Timedelta(hours=12)
            event_data = df.loc[win_start:win_end].copy()
        else:
            event_data = pd.DataFrame()

    if len(event_data) > 0:
        # Plot water level as filled area
        ax_event.fill_between(
            event_data.index,
            0,
            event_data['WL_MHHW'].clip(lower=0),
            alpha=0.4,
            color='#6495ED',
            label='Water Level'
        )

        # Flood threshold line
        ax_event.axhline(
            y=wl_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Flood Threshold'
        )

        # CFRI on secondary axis
        ax2 = ax_event.twinx()
        ax2.plot(
            event_data.index,
            event_data['CFRI'],
            color='purple',
            linewidth=2.5,
            label='CFRI'
        )
        ax2.axhline(
            y=cfri_threshold,
            color='orange',
            linestyle='--',
            linewidth=2,
            label='CFRI Threshold'
        )
        ax2.set_ylabel('CFRI', color='purple', fontsize=12)
        ax2.set_ylim(0, 1.15)
        ax2.tick_params(axis='y', labelcolor='purple')

        # Calculate and annotate lead time
        if selected_event:
            pre_start = selected_event.start_time - pd.Timedelta(hours=72)
            pre_event = df.loc[pre_start:selected_event.start_time, 'CFRI']
            triggers = pre_event > cfri_threshold
            if triggers.any():
                first_trigger = triggers.idxmax()
                lead_time = (selected_event.start_time - first_trigger).total_seconds() / 3600

                # Draw vertical lines at trigger and flood onset
                ax_event.axvline(x=first_trigger, color='orange', linewidth=2, alpha=0.8)
                ax_event.axvline(x=selected_event.start_time, color='red', linewidth=2, alpha=0.8)

                # Add double-headed arrow showing lead time
                arrow_y = wl_threshold * 0.85
                ax_event.annotate(
                    '',
                    xy=(selected_event.start_time, arrow_y),
                    xytext=(first_trigger, arrow_y),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2)
                )

                # Lead time text
                mid_time = first_trigger + (selected_event.start_time - first_trigger) / 2
                ax_event.text(
                    mid_time,
                    arrow_y + wl_threshold * 0.12,
                    f'{lead_time:.0f}h lead',
                    ha='center',
                    fontsize=11,
                    fontweight='bold'
                )

        # Labels and formatting
        ax_event.set_ylabel('Water Level (m above MHHW)', color='#4169E1', fontsize=12)
        ax_event.tick_params(axis='y', labelcolor='#4169E1')
        ax_event.set_xlabel(f'Date ({event_data.index[0].strftime("%b %Y")})', fontsize=11)
        ax_event.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # Title
        title_text = '(A) Early Warning Example'
        if event_name:
            title_text += f': {event_name}'
        ax_event.set_title(title_text, fontsize=12, fontweight='bold')

        # Custom legend
        handles = [
            mpatches.Patch(color='#6495ED', alpha=0.4, label='Water Level'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Flood Threshold'),
            Line2D([0], [0], color='purple', linewidth=2.5, label='CFRI'),
            Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='CFRI Threshold')
        ]
        ax_event.legend(handles=handles, loc='upper left', fontsize=9)

    # === Panel B: Lead Time Distribution ===
    ax_lead = axes[1]

    lead_times = []
    for event in events:
        pre_start = event.start_time - pd.Timedelta(hours=72)
        pre_event = df.loc[pre_start:event.start_time, 'CFRI']
        triggers = pre_event > cfri_threshold
        if triggers.any():
            first_trigger = triggers.idxmax()
            lead = (event.start_time - first_trigger).total_seconds() / 3600
            if lead > 0 and lead < 72:
                lead_times.append(lead)

    if lead_times:
        # Create histogram
        bins = np.arange(0, max(lead_times) + 10, 10)
        ax_lead.hist(
            lead_times,
            bins=bins,
            color='#E74C3C',
            edgecolor='black',
            alpha=0.8
        )

        # Mean line
        mean_lead = np.mean(lead_times)
        ax_lead.axvline(
            mean_lead,
            color='black',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_lead:.1f}h'
        )

        ax_lead.set_xlabel('Lead Time (hours)', fontsize=11)
        ax_lead.set_ylabel('Number of Events', fontsize=11)
        ax_lead.legend(loc='upper left', fontsize=10)
        ax_lead.set_title(
            f'(B) Lead Time Distribution ({len(lead_times)} events)',
            fontsize=12,
            fontweight='bold'
        )

    # Overall title
    fig.suptitle(
        f'Compound Flood Risk Index (CFRI) Performance: {site_name}',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved simple killer figure: {save_path}")

    return fig


def plot_timeseries_overview(
    df: pd.DataFrame,
    events: List[FloodEvent],
    year: int,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot annual time series overview.

    Parameters
    ----------
    df : pd.DataFrame
        Data with Q, WL_MHHW, CFRI
    events : list
        Flood events
    year : int
        Year to plot
    save_path : Path, optional
        Save path

    Returns
    -------
    matplotlib.Figure
    """
    # Filter to year
    year_start = pd.Timestamp(f'{year}-01-01', tz='UTC')
    year_end = pd.Timestamp(f'{year}-12-31 23:59:59', tz='UTC')
    year_data = df.loc[year_start:year_end]

    if len(year_data) == 0:
        logger.warning(f"No data for year {year}")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Discharge
    axes[0].plot(year_data.index, year_data['Q'], 'b-', linewidth=0.5)
    axes[0].set_ylabel('Q (m³/s)')
    axes[0].set_title(f'Annual Overview: {year}')

    # Panel 2: Water Level
    axes[1].plot(year_data.index, year_data['WL_MHHW'], 'g-', linewidth=0.5)
    axes[1].set_ylabel('WL (m MHHW)')

    # Panel 3: CFRI with events
    axes[2].plot(year_data.index, year_data['CFRI'], 'purple', linewidth=0.5)
    axes[2].set_ylabel('CFRI')
    axes[2].set_xlabel('Date')

    # Mark flood events
    year_events = [e for e in events if e.start_time.year == year]
    for event in year_events:
        for ax in axes:
            ax.axvspan(event.start_time, event.end_time, alpha=0.3, color='red')

    # Format
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")

    return fig


def plot_threshold_grid_search(
    grid_results: pd.DataFrame,
    optimal_threshold: float,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot grid search results.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Results from grid_search_threshold
    optimal_threshold : float
        Selected optimal threshold
    save_path : Path, optional
        Save path

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(grid_results['threshold'], grid_results['f1'],
            'b-', linewidth=2, label='F1 Score')
    ax.plot(grid_results['threshold'], grid_results['precision'],
            'g--', linewidth=1.5, label='Precision')
    ax.plot(grid_results['threshold'], grid_results['recall'],
            'r--', linewidth=1.5, label='Recall')
    ax.plot(grid_results['threshold'], grid_results['far'],
            'orange', linestyle=':', linewidth=1.5, label='FAR')

    # Mark optimal
    ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2,
               label=f'Optimal T={optimal_threshold:.2f}')

    ax.set_xlabel('CFRI Threshold')
    ax.set_ylabel('Score')
    ax.set_xlim(grid_results['threshold'].min(), grid_results['threshold'].max())
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax.set_title('Threshold Grid Search Results')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")

    return fig


def plot_lag_analysis(
    correlations: np.ndarray,
    lags: np.ndarray,
    optimal_lag: int,
    event_lags: List[float] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot lag analysis results.

    Parameters
    ----------
    correlations : np.ndarray
        Cross-correlation values
    lags : np.ndarray
        Lag values in hours
    optimal_lag : int
        Selected optimal lag
    event_lags : list, optional
        Event-based lag estimates
    save_path : Path, optional
        Save path

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Cross-correlation
    ax1 = axes[0]
    ax1.plot(lags, correlations, 'b-', linewidth=1.5)
    ax1.axvline(x=optimal_lag, color='red', linestyle='--',
                label=f'Optimal: {optimal_lag}h')
    ax1.set_xlabel('Lag (hours, Q leads WL)')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Cross-Correlation Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Event-based lags
    ax2 = axes[1]
    if event_lags and len(event_lags) > 0:
        ax2.hist(event_lags, bins=20, color='blue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.median(event_lags), color='red', linestyle='--',
                    label=f'Median: {np.median(event_lags):.1f}h')
        ax2.set_xlabel('Q-to-WL Lag (hours)')
        ax2.set_ylabel('Number of Events')
        ax2.set_title('Event-Based Lag Distribution')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No event lags available',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Event-Based Lag Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")

    return fig


def plot_comparison_bars(
    comparison_df: pd.DataFrame,
    metric: str = 'F1',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create bar chart comparing models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison results
    metric : str
        Metric to plot
    save_path : Path, optional
        Save path

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = comparison_df['Model'].values
    values = comparison_df[metric].values
    colors = ['#e74c3c' if m == 'CFRI' else '#3498db' for m in models]

    bars = ax.bar(models, values, color=colors, edgecolor='black')

    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Score by Model')
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")

    return fig


def create_figure_set(
    df: pd.DataFrame,
    events: List[FloodEvent],
    threshold_results: Dict,
    comparison_df: pd.DataFrame,
    site_name: str,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create all figures for a site.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    events : list
        Flood events
    threshold_results : dict
        Threshold selection results
    comparison_df : pd.DataFrame
        Model comparison
    site_name : str
        Site name
    output_dir : Path
        Output directory

    Returns
    -------
    dict
        Paths to created figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Killer figure
    killer_path = output_dir / f'{site_name}_killer_figure.png'
    plot_killer_figure(
        df, events,
        cfri_threshold=threshold_results['optimal_threshold'],
        wl_threshold=df.attrs.get('flood_threshold', 0),
        comparison_results=comparison_df,
        site_name=site_name,
        save_path=killer_path
    )
    paths['killer_figure'] = killer_path

    # Grid search
    grid_path = output_dir / f'{site_name}_grid_search.png'
    plot_threshold_grid_search(
        threshold_results['grid_results'],
        threshold_results['optimal_threshold'],
        save_path=grid_path
    )
    paths['grid_search'] = grid_path

    # Comparison bars
    for metric in ['F1', 'Precision', 'Recall']:
        bar_path = output_dir / f'{site_name}_comparison_{metric.lower()}.png'
        plot_comparison_bars(comparison_df, metric=metric, save_path=bar_path)
        paths[f'comparison_{metric.lower()}'] = bar_path

    # Event case studies (top 3 events by lead time)
    for i, event in enumerate(events[:3]):
        event_path = output_dir / f'{site_name}_event_{event.event_id}.png'
        plot_event_case_study(
            df, event,
            threshold=df.attrs.get('flood_threshold', 0),
            cfri_threshold=threshold_results['optimal_threshold'],
            save_path=event_path
        )
        paths[f'event_{event.event_id}'] = event_path

    logger.info(f"Created {len(paths)} figures in {output_dir}")

    return paths
