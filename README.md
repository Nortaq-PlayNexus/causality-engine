# CAUSALITY ENGINE

Computational research environment for distinguishing causal mechanisms from correlations.

## Features
- Configurable ecosystems, population dynamics, resource systems, physical models, artificial economies with known ground-truth causal graphs.
- Structural causal models (SCM), interventions (do-operator), counterfactuals.
- Hidden/latent variables, time-series analysis, alternative hypotheses, automated experiment selection.
- Causal graph recovery (PC skeleton, hill-climbing) with SHD/FDR/TPR metrics.
- Negative controls, reproducible experiments (seeded).
- Interactive PySide6 desktop app: graph editing, simulation controls, intervention tools, results dashboards.
- Benchmark: graph recovery, FDR, intervention prediction, noise robustness, generalization.

## Core principle
Correlation does not imply causation. Causation requires temporal precedence, covariation, elimination of confounders, interventional evidence, and counterfactual support.

## Run
```
python main.py
```

## Tests
```
pytest tests/ -q
```

## License
MIT
