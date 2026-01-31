# Compound Flood Risk Index (CFRI)
## A Physics-Informed Early Warning System for Compound River-Coastal Flooding

**Author:** NCSEF Student Researcher
**Version:** 2.0
**Date:** January 2025
**Study Site:** Wilmington, NC (Cape Fear River Estuary)

---

## Abstract

Compound flooding occurs when elevated river discharge coincides with high coastal water levels, creating flood impacts greater than either driver alone. This study develops the Compound Flood Risk Index (CFRI), a physics-informed index that captures the nonlinear amplification of flood risk during compound events. Applied to 30 years of data (1995-2024) at Wilmington, NC, the CFRI achieves **100% detection of compound flood events** with a **mean lead time of 10.5 days**. The methodology uses daily mean CFRI to eliminate tidal oscillations, providing clear, actionable early warnings without arbitrary lookback windows.

---

## 1. Introduction

### 1.1 Problem Statement

Coastal communities face increasing flood risk from compound events where:
- **River flooding**: High discharge from upstream rainfall/snowmelt
- **Coastal flooding**: Elevated water levels from storm surge, tides, or sea level rise

When both drivers occur simultaneously, flood risk is **nonlinearly amplified** - the combined effect exceeds the sum of individual effects.

### 1.2 Objectives

1. Develop a physics-informed index that captures compound flood risk
2. Provide early warning with sufficient lead time for emergency response
3. Achieve high detection rate for compound flood events
4. Eliminate tidal oscillation artifacts for clear warning signals

---

## 2. Data

### 2.1 Input Variables

| Variable | Source | Station | Units | Temporal Resolution |
|----------|--------|---------|-------|---------------------|
| River Discharge (Q) | USGS | Cape Fear River | m³/s | Hourly |
| Water Level (WL) | NOAA | Wilmington, NC (8658120) | m above MHHW | Hourly |

### 2.2 Study Period

- **Full record:** 1995-2024 (30 years)
- **Total hours:** ~263,000
- **Compound flood events:** 34

### 2.3 Preprocessing

1. **Time alignment:** Synchronize Q and WL to common hourly timestamps
2. **Quality control:** Remove outliers, interpolate small gaps
3. **Datum conversion:** Convert WL to meters above Mean Higher High Water (MHHW)
4. **Normalization:** Transform Q and WL to [0,1] using cumulative distribution function (CDF)

---

## 3. Methodology

### 3.1 River-to-Coast Lag Estimation

River discharge measured at the upstream USGS gauge takes time to reach the coastal area. The optimal lag is determined empirically via **cross-correlation analysis**:

```
lag_optimal = argmax[ corr(Q(t - lag), WL(t)) ]
```

**Result:** Optimal lag = **22 hours** for the Cape Fear River system

This means Q_lag(t) represents discharge that will arrive at the coast around time t.

### 3.2 CFRI Formula

The CFRI combines three components:

```
CFRI(t) = w₁·Q_lag(t) + w₂·WL_norm(t) + w₃·[Q_lag(t) × WL_norm(t) × I_compound(t)]
```

**Where:**
- `Q_lag(t)` = Normalized river discharge, lagged by 22 hours
- `WL_norm(t)` = Normalized water level at current time
- `I_compound(t)` = Compound indicator function
- `w₁ = 0.15` (river weight)
- `w₂ = 0.15` (coastal weight)
- `w₃ = 0.70` (compound interaction weight)

**Compound Indicator:**
```
I_compound(t) = 1  if Q_lag(t) > Q_median AND WL_norm(t) > WL_median
              = 0  otherwise
```

### 3.3 Physical Interpretation

The CFRI formula captures **nonlinear risk amplification**:

| Condition | CFRI Components Active | Risk Level |
|-----------|----------------------|------------|
| Q low, WL low | Individual terms only (30%) | Low |
| Q high, WL low | Q term + partial compound | Moderate |
| Q low, WL high | WL term + partial compound | Moderate |
| **Q high, WL high** | **All terms (100%)** | **High** |

The 70% weight on the compound term ensures CFRI is strongly elevated only when BOTH drivers exceed their median values simultaneously.

### 3.4 Daily Mean CFRI (Operational Approach)

**Problem:** Hourly CFRI oscillates with tides, crossing thresholds multiple times per day.

**Solution:** Use daily mean to eliminate tidal oscillations:

```
CFRI_daily(d) = mean[ CFRI(t) for all t in day d ]
```

**Advantages:**
- One clear value per day
- Single threshold crossing = unambiguous warning onset
- No arbitrary lookback window required

### 3.5 Warning System

**Warning issued when:** CFRI_daily > 0.45

**Lead time calculation:**
1. Identify flood onset (first hour when WL > flood threshold)
2. Find first day when CFRI_daily crossed above 0.45
3. Lead time = (flood onset date) - (warning date)

---

## 4. Compound Flood Event Definition

An event is classified as a **compound flood** if during the event:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Water Level | > 0.56 m above MHHW | NWS minor flood stage |
| River Discharge | > 75th percentile | Elevated above typical conditions |

This distinguishes:
- **Compound floods (34 events):** Both drivers elevated
- **Coastal-only floods (19 events):** High WL with normal Q

---

## 5. Results

### 5.1 Performance Metrics

| Metric | Value |
|--------|-------|
| Total flood events | 53 |
| Compound flood events | 34 |
| **Detection rate** | **100% (34/34)** |
| Mean lead time | 10.5 days |
| Median lead time | 13.5 days |
| Lead time range | 0-14 days |

### 5.2 Case Study: Hurricane Matthew (2016)

Hurricane Matthew caused severe compound flooding at Wilmington on October 8-9, 2016:

- **Flood onset:** October 8, 2016 17:00 UTC
- **Warning issued:** October 1, 2016 (CFRI_daily first exceeded 0.45)
- **Lead time:** 7 days
- **Peak water level:** 1.07 m above MHHW
- **River discharge:** >99th percentile

---

## 6. Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| w₁ | 0.15 | River discharge weight |
| w₂ | 0.15 | Water level weight |
| w₃ | 0.70 | Compound interaction weight |
| Q lag | 22 hours | River-to-coast travel time |
| Warning threshold | 0.45 | Daily mean CFRI threshold |
| Flood threshold | 0.56 m | Water level above MHHW |
| Q elevated | 75th percentile | River discharge threshold for compound classification |

---

## 7. Code Structure

### 7.1 Core Modules (`src/compound_flood/`)

| Module | Purpose |
|--------|---------|
| `io.py` | Data loading and validation |
| `preprocess.py` | Time alignment, normalization |
| `features.py` | Lag estimation, CFRI computation |
| `events.py` | Flood event detection |
| `evaluation.py` | Performance metrics, lead times |
| `cfri_v2.py` | Compound-focused CFRI formula |
| `plots.py` | Publication-quality figures |

### 7.2 Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Main analysis workflow |
| `run_cfri_v2_analysis.py` | CFRI v2 analysis with Daily Mean approach |
| `download_data.py` | Fetch data from NOAA/USGS APIs |

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Single study site:** Validated only at Wilmington, NC
2. **Historical analysis:** Uses observed data, not forecasts
3. **Binary compound indicator:** Does not capture gradual transitions

### 8.2 Future Directions

1. **Multi-site validation:** Apply to other estuarine systems
2. **Forecast integration:** Couple with NWS river and storm surge forecasts
3. **Probabilistic framework:** Provide confidence intervals on warnings
4. **Climate projections:** Assess future compound flood risk under sea level rise

---

## 9. References

### Data Sources
- USGS National Water Information System (NWIS): https://waterdata.usgs.gov
- NOAA Tides and Currents: https://tidesandcurrents.noaa.gov

### Study Site
- Location: Wilmington, NC (34.2257°N, 77.9447°W)
- NOAA Station: 8658120
- USGS Gauge: Cape Fear River

---

## 10. Appendix: Mathematical Derivations

### A.1 CDF Normalization

For variable X with empirical CDF F_X:
```
X_norm = F_X(X) = P(X ≤ x)
```

This transforms any distribution to uniform [0,1] while preserving rank order.

### A.2 Cross-Correlation for Lag Estimation

The cross-correlation at lag τ:
```
ρ(τ) = Σ[(Q(t-τ) - μ_Q)(WL(t) - μ_WL)] / (σ_Q × σ_WL × N)
```

Optimal lag maximizes ρ(τ), representing the time shift where Q best predicts WL.

### A.3 Compound Amplification Factor

The ratio of observed to expected flood probability:
```
Amplification = P(flood | Q>90th, WL>90th) / [P(flood|Q>90th) + P(flood|WL>90th)]
```

For Wilmington: Amplification ≈ 1.6x (flood risk 60% higher than additive expectation)
