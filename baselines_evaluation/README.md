# Baseline Evaluation for RTS Curator Recommendation System

This directory contains a **completely separate** evaluation framework for comparing baseline methods against the main CTS (Contextual Thompson Sampling) approach.

## Overview

This evaluation framework implements offline historical replay to compare:
- **6 baseline methods**: Random, Static Scalarization, Audience-Only, Curator Logistic Regression, Thompson Sampling (global), LinUCB
- **3 ablation studies**: CTS variants isolating specific components
- **1 main method**: Full CTS approach

See `docs/implementation_evaluation.md` for the complete evaluation specification.

## Structure

```
baselines_evaluation/
├── src/                    # Evaluation-specific code
│   ├── policies/          # All baseline policy implementations
│   ├── metrics/           # Evaluation metrics (Hit@K, NDCG@K, value signals)
│   ├── replay/            # Offline replay engine
│   └── visualization/     # Plotting utilities
│
├── notebooks/             # Interactive evaluation notebooks (00-07)
│   ├── 00_setup_validation.ipynb
│   ├── 01_baseline_policies.ipynb
│   ├── 02_replay_engine.ipynb
│   ├── 03_curator_logistic_baseline.ipynb
│   ├── 04_bandit_baselines.ipynb
│   ├── 05_ablation_studies.ipynb
│   ├── 06_full_evaluation.ipynb
│   └── 07_results_analysis.ipynb
│
├── results/               # Generated results (gitignored)
│   ├── metrics/          # Hit@K, NDCG@K CSVs
│   ├── value_signals/    # Aggregated value signals
│   ├── logs/             # Per-policy decision logs
│   └── trained_models/   # Trained baseline models
│
├── figures/               # Generated plots
│   ├── radar_plots/      # Multi-objective trade-off visualizations
│   ├── heatmaps/         # Context-stratified performance
│   └── comparisons/      # Bar charts and comparisons
│
└── config/                # Evaluation configuration
    └── evaluation_config.yaml
```

## Key Principles

- **Completely separate** from main codebase (`src/cts_recommender/`)
- **Self-contained** evaluation directory with its own structure
- **Imports from main codebase** but doesn't modify it
- **Notebook-driven** for interactive analysis
- **Modular and reusable** evaluation code

## Usage

### Getting Started

1. **Install dependencies** (if not already done):
   ```bash
   uv sync --group notebooks
   ```

2. **Run notebooks in order**:
   - Start with `00_setup_validation.ipynb` to verify imports and data loading
   - Progress through `01-07` notebooks following the implementation phases

3. **View results**:
   - Metrics saved to `results/metrics/`
   - Plots generated in `figures/`

### Running Full Evaluation

The complete evaluation pipeline is in `06_full_evaluation.ipynb`:
- Replays all 9 policies on historical data
- Computes ranking metrics (Hit@K, NDCG@K)
- Aggregates value signals (global + per-context)
- Generates summary tables

## Implementation Phases

See `docs/BASELINE_EVALUATION_PLAN.md` for the detailed implementation plan:

- [x] **Phase 0:** Setup & Directory Structure
- [ ] **Phase 1:** Policy Interface & Simple Baselines
- [ ] **Phase 2:** Evaluation Metrics & Replay Engine
- [ ] **Phase 3:** Supervised Baseline (Logistic Regression)
- [ ] **Phase 4:** Bandit Baselines
- [ ] **Phase 5:** Ablation Studies
- [ ] **Phase 6:** Full Evaluation Run
- [ ] **Phase 7:** Visualization & Analysis
- [ ] **Phase 8:** Documentation

## Importing from Main Codebase

All notebooks use this pattern:

```python
import sys
from pathlib import Path

# Add both baselines_evaluation/src and main src to path
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "baselines_evaluation" / "src"))

# Import from main codebase
from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.environments.reward import RewardCalculator

# Import from evaluation code
from policies.base import BasePolicy
from metrics.ranking import hit_at_k, ndcg_at_k
```

## Notes

- **No changes to main codebase** - Only imports
- **Self-contained** - Can be deleted/archived after evaluation
- **Reproducible** - All config in `config/evaluation_config.yaml`
- **Clean separation** - Clear distinction between production and evaluation code
