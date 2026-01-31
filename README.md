# Compound Flood Risk Index (CFRI)

## A Physics-Informed Early Warning System for Compound River-Coastal Flooding

**A NCSEF Project by Victor He, Cary Academy**

---

## Overview

The **Compound Flood Risk Index (CFRI)** is a physics-informed early-warning metric that predicts compound flooding events by capturing the nonlinear interaction between river discharge and coastal water levels.

**Key Results (Wilmington, NC, 1995-2024):**
- **100% detection rate** for compound flood events (34/34)
- **10.5 days mean lead time**
- Clear, actionable warnings without tidal oscillation artifacts

## CFRI Formula

```
CFRI = 0.15·Q_lag + 0.15·WL_norm + 0.70·(Q_lag × WL_norm × I_compound)
```

Where:
- `Q_lag` = Normalized river discharge (lagged 22 hours for river-to-coast travel time)
- `WL_norm` = Normalized water level
- `I_compound` = 1 when BOTH Q and WL exceed their median values

The 70% weight on the compound interaction term ensures CFRI is strongly elevated only during true compound conditions.

## Quick Start

```bash
# Install dependencies
conda env create -f environment.yml
conda activate compound-flood

# Run the analysis
python scripts/run_cfri_v2_analysis.py --config configs/ncsef.yaml
```

## Project Structure

```
CFRI/
├── docs/
│   └── CFRI_scientific_documentation.md   # Full methodology
├── src/compound_flood/
│   ├── io.py              # Data loading
│   ├── preprocess.py      # Time alignment, normalization
│   ├── features.py        # CFRI computation, lag estimation
│   ├── events.py          # Flood event detection
│   ├── evaluation.py      # Performance metrics
│   ├── cfri_v2.py         # Compound-focused CFRI formula
│   └── plots.py           # Publication figures
├── scripts/
│   ├── run_pipeline.py           # Core analysis workflow
│   ├── run_cfri_v2_analysis.py   # CFRI v2 with Daily Mean approach
│   └── download_data.py          # Fetch NOAA/USGS data
├── configs/
│   └── ncsef.yaml         # Configuration file
├── outputs/
│   └── wilmington/        # Results and figures
└── data/                  # Input data (not in repo)
```

## Key Innovation

**Problem:** Standard flood thresholds trigger only when water is already flooding - no advance warning.

**Solution:** CFRI provides early warning by:
1. Using **lagged river discharge** to anticipate downstream effects
2. Modeling **compound amplification** when both drivers are elevated
3. Using **daily mean CFRI** to eliminate tidal oscillation artifacts

## Documentation

See [`docs/CFRI_scientific_documentation.md`](docs/CFRI_scientific_documentation.md) for:
- Complete methodology and equations
- Data sources and preprocessing
- Performance results and case studies
- Parameter descriptions

## Study Sites

| Site | River Gauge | Tide Gauge |
|------|-------------|------------|
| Wilmington, NC | Cape Fear River (USGS) | NOAA 8658120 |

