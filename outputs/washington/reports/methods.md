## Data Sources and Preprocessing

Hourly river discharge (Q) data were obtained from USGS streamflow gauges,
and hourly water level (WL) data from NOAA tide gauges. The analysis period
spans 30 years of continuous observations.

All water level data were converted to a common datum (Mean Higher High Water, MHHW)
to ensure consistent flood threshold definition. Missing data gaps less than
6 hours
were filled using linear interpolation; larger gaps were left as missing.

## Compound Flood Risk Index (CFRI)

The CFRI combines normalized river discharge, normalized water level,
and an interaction term:

$$CFRI(t) = \frac{w_1 \cdot Q^*_{lag}(t) + w_2 \cdot WL^*(t) + w_3 \cdot Overlap(t)}{w_1 + w_2 + w_3}$$

where:
- $Q^*_{lag}(t)$ = percentile-normalized river discharge shifted by 0 hours
- $WL^*(t)$ = percentile-normalized water level above MHHW
- $Overlap(t) = Q^*_{lag}(t) \times WL^*(t)$ (multiplicative interaction)
- Weights: $w_1 = 0.333$, $w_2 = 0.333$, $w_3 = 0.333$

## River-to-Estuary Lag Estimation

The optimal lag (0 hours) was determined using a hybrid approach:
1. Cross-correlation analysis between Q and WL time series
2. Event-based analysis of Q-peak to WL-peak timing during floods

## Threshold Selection and Validation

To prevent data leakage, we employed temporal train/test splitting:
- Training period: 2001-2011 (11 years)
- Test period: 2012-2016 (5 years)

The optimal threshold (0.870) was selected via grid search
on training data, maximizing F1 score subject to a false alarm ratio
constraint (FAR ≤ 0.30). Performance was then evaluated on the
held-out test period.

## Predictive Label Definition

For each hour t, the target label Y(t) = 1 if observed water level
exceeds the flood threshold at any point in the subsequent 24 hours:

$$Y(t) = \mathbb{1}[\max_{\tau \in [t, t+24h]} WL(\tau) > threshold]$$

This formulation enables lead time evaluation: how far in advance
does CFRI trigger before actual flooding occurs?