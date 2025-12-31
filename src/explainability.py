import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from typing import Any

def plot_builtin_importance(model: Any, feature_names: pd.Index, save_path: str, top_n: int = 10):
    """Saves and displays Built-in Gini importance."""
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.head(top_n).values, y=feat_imp.head(top_n).index, palette='viridis')
    plt.title(f'Top {top_n} Built-in Feature Importances')
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.show()
    return feat_imp

def perform_shap_analysis(model: Any, X_test: pd.DataFrame, report_dir: str):
    """Saves and displays Global SHAP summary plots."""
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_sample)

    # Resolve SHAP values for Class 1 (Fraud)
    if isinstance(shap_raw, list):
        sv_plot = np.array(shap_raw[1])
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        sv_plot = shap_raw[:, :, 1]
    else:
        sv_plot = np.array(shap_raw)

    # 1. Save Bar Plot
    plt.figure()
    shap.summary_plot(sv_plot, X_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP Bar Plot)")
    plt.tight_layout()
    bar_path = os.path.join(report_dir, "shap_feature_importance_bar.png")
    plt.savefig(bar_path)
    print(f"Saved: {bar_path}")
    plt.show()
    
    # 2. Save Summary/Density Plot
    plt.figure()
    shap.summary_plot(sv_plot, X_sample, show=False)
    plt.tight_layout()
    summary_path = os.path.join(report_dir, "shap_summary_density.png")
    plt.savefig(summary_path)
    print(f"Saved: {summary_path}")
    plt.show()

def plot_individual_explanation(model: Any, X_row: pd.DataFrame, title: str, save_path: str):
    """Saves and displays Waterfall plot for a specific prediction."""
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_row)
    base_raw: Any = explainer.expected_value

    # Extract SHAP values
    if isinstance(shap_raw, list):
        sv = np.array(shap_raw[1][0])
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        sv = shap_raw[0, :, 1]
    else:
        sv = np.array(shap_raw[0])

    # Extract Base Value
    if isinstance(base_raw, (list, np.ndarray)):
        bv = float(base_raw[1]) if len(base_raw) > 1 else float(base_raw[0])
    else:
        bv = float(base_raw)

    # Create Explanation
    exp = shap.Explanation(
        values=sv.astype(float).flatten(), 
        base_values=bv, 
        data=X_row.values.astype(float).flatten(), 
        feature_names=X_row.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    # Note: Waterfall plot is complex; we use show=False to capture it for saving
    shap.plots.waterfall(exp, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.show()