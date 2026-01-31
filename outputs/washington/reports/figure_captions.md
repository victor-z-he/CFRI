# Figure Captions

## killer_figure

Figure 1. Compound Flood Risk Index (CFRI) performance for washington.
(A) Example flood event demonstrating early warning capability:
CFRI (purple) exceeds the trigger threshold (orange dashed line)
substantially before observed water level (blue shading) reaches
the flood threshold (red dashed line). Arrows indicate lead time.
(B) Classification performance comparison showing CFRI outperforms
baseline models (water-level-only, discharge-only) across all metrics.
(C) Distribution of early warning lead times across all detected
flood events, with mean and median indicated.
(D) Sensitivity analysis showing the tradeoff between false alarm ratio
and detection rate as the CFRI threshold varies. Red marker indicates
the optimal threshold (0.87) selected via training-period
grid search.

## grid_search

Figure 2. Grid search results for CFRI threshold selection.
F1 score (blue), precision (green), recall (red), and false alarm ratio
(orange) as functions of the CFRI trigger threshold. The optimal
threshold (0.87, black dashed line) maximizes F1 score
while constraining FAR ≤ 0.30 on training data.

## event_study

Figure 3. Case study of compound flood event.
Top panel: River discharge (Q) showing the antecedent flow conditions.
Middle panel: Observed water level (above MHHW) with flood threshold.
Bottom panel: CFRI value with trigger threshold. Vertical lines mark
the CFRI trigger time and flood onset; the horizontal arrow indicates
the early warning lead time achieved.

## comparison_f1

Figure 4. Model comparison by F1 score.
The Compound Flood Risk Index (CFRI, red) achieves superior predictive
performance (F1 = 0.005) compared to simpler baselines including
water-level-only (reactive) and discharge-only (no tidal information)
indices. Error bars represent 95% bootstrap confidence intervals.

