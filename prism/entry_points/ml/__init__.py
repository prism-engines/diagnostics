"""
ML Entry Points
===============

Machine learning entry points for PRISM diagnostics.

Usage:
    python -m prism.entry_points.ml.features --target RUL
    python -m prism.entry_points.ml.train --model xgboost
    python -m prism.entry_points.ml.predict
    python -m prism.entry_points.ml.ablation
    python -m prism.entry_points.ml.baseline
    python -m prism.entry_points.ml.benchmark

Modules:
    features    - Generate ML-ready feature tables from PRISM outputs
    train       - Train ML models (XGBoost, CatBoost, LightGBM, etc.)
    predict     - Run predictions on new data
    ablation    - Feature ablation studies
    baseline    - Baseline XGBoost model without PRISM features
    benchmark   - Compare PRISM vs baseline performance
"""
