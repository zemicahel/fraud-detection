import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    auc, roc_auc_score, f1_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Prints comprehensive performance metrics and plots Confusion Matrix."""
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # Metrics
    auc_roc = roc_auc_score(y_test, y_probs)
    auc_pr = average_precision_score(y_test, y_probs)
    f1 = f1_score(y_test, y_pred)
    
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR (Precision-Recall): {auc_pr:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title(f'Confusion Matrix: {model_name}')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    ax[1].plot(recall, precision, label=f'AUC-PR: {auc_pr:.2f}')
    ax[1].set_title(f'Precision-Recall Curve: {model_name}')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend()
    
    plt.show()
    
    return {"AUC-ROC": auc_roc, "AUC-PR": auc_pr, "F1": f1}

def perform_cross_validation(model, X, y):
    """Performs Stratified K-Fold CV and returns mean scores."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['f1', 'roc_auc', 'average_precision']
    
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    
    print("\n--- 5-Fold Cross Validation ---")
    print(f"Mean AUC-PR: {cv_results['test_average_precision'].mean():.4f} (+/- {cv_results['test_average_precision'].std():.4f})")
    print(f"Mean F1-Score: {cv_results['test_f1'].mean():.4f}")
    return cv_results