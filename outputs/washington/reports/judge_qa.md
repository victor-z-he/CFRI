# Anticipated Judge Questions & Answers

## Q1: Why did you use a train/test split instead of cross-validation?

**Answer:**

Temporal train/test splitting is essential for time series prediction problems.
Cross-validation with random splits would cause data leakage - the model could
"learn" from future data that wouldn't be available in real-world forecasting.
We used a 70/30 temporal split where all training data precedes all test data,
mimicking how the model would actually be deployed for operational forecasting.

---

## Q2: How did you determine the lag between river discharge and coastal flooding?

**Answer:**

We used two complementary methods:
1. Cross-correlation analysis of the entire time series identifies the lag
   that maximizes correlation between Q and WL.
2. Event-based analysis measures the time between peak discharge and peak
   water level during individual flood events.

The hybrid approach uses event-based estimates when sufficient events exist
(≥10), as these directly capture flood dynamics rather than average conditions.

---

## Q3: What makes your index 'physics-informed' rather than purely statistical?

**Answer:**

The CFRI structure is motivated by physical understanding of compound flooding:
1. River discharge provides water volume that must drain through the estuary
2. High coastal water levels impede drainage (backwater effect)
3. The overlap term captures the nonlinear interaction - flooding risk is
   highest when both drivers are elevated simultaneously
4. The lag accounts for physical travel time from river to estuary

This differs from "black box" machine learning that discovers patterns without
physical constraints.

---

## Q4: Why did you normalize using percentiles instead of z-scores?

**Answer:**

Percentile normalization has several advantages:
1. Bounded output [0,1] regardless of extreme values
2. Robust to outliers (one extreme storm doesn't distort the scale)
3. Directly interpretable (0.9 means "higher than 90% of historical values")
4. Works well for non-Gaussian distributions common in hydrology

Critical point: we compute percentiles using only training data to avoid
data leakage into the test period.

---

## Q5: How would you deploy this for operational flood warning?

**Answer:**

For real-time deployment:
1. Ingest live discharge from USGS and water level from NOAA APIs
2. Apply the pre-fitted normalization parameters (from historical training)
3. Compute CFRI using the calibrated lag and weights
4. Trigger alert when CFRI exceeds the optimized threshold

The ~24 hour average lead time provides actionable warning for emergency
managers to pre-position resources, issue evacuation orders, and alert
low-lying communities.

---

## Q6: What are the limitations of your approach?

**Answer:**

Key limitations:
1. Requires consistent historical data - gaps reduce training quality
2. Assumes stationarity - climate change may shift relationships over time
3. Single-threshold approach may miss multi-stage events
4. Does not account for rainfall or wind (future extension)
5. Calibrated for specific site - would need re-tuning for other estuaries

These are areas for future research and model improvement.

---

## Q7: Why is the overlap term important?

**Answer:**

The overlap term captures compound flood physics: total flood risk is not
simply additive. When river discharge is high AND coastal water level is
elevated simultaneously, the backwater effect prevents normal drainage,
causing water levels to rise disproportionately.

Mathematically, Overlap = Q* × WL* gives high values only when BOTH
drivers are elevated. This explains ~30% of CFRI predictive power beyond
the individual terms.

---

