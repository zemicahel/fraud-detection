# Fraud Detection for E-commerce and Banking

## 📌 Project Overview
This project focuses on building a robust fraud detection system for **Adey Innovations Inc.** By integrating geolocation data and analyzing transaction patterns, we identify high-risk activities to minimize financial losses in e-commerce and banking environments.

---

## 📂 Project Structure
```text
fraud-detection/
├── .github/
│   └── workflows/           # CI/CD (Automation for unit tests)
├── data/                    # Data storage (Ignored by Git)
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and feature-engineered datasets
├── notebooks/               # Interactive analysis
│   ├── eda-fraud-data.ipynb # Task 1a: Visual Analysis
│   ├── feature-engineering.ipynb # Task 1b: Transformation
│   └── README.md            # Notebook descriptions
├── src/                     # Modular Source Code
│   ├── __init__.py
│   ├── data_preprocessing.py # Cleaning & Geolocation Integration
│   ├── feature_engineering.py# Transformation & SMOTE
│   ├── model_training.py    # Training & Evaluation logic
│   └── visualization.py     # EDA plotting logic
├── tests/                   # Automated tests
├── models/                  # Saved ML model artifacts (.pkl)
├── scripts/                 # Standalone production scripts
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
🚀 Task 1: Data Analysis and Preprocessing

Objective: Clean data and engineer features to prepare for machine learning.

Data Cleaning: Handled missing values and removed duplicates to ensure data integrity.

Geolocation Integration: Mapped IP addresses to countries using pd.merge_asof for high-performance range lookups.

Feature Engineering:

Temporal: Extracted hour_of_day and day_of_week.

Velocity: Created time_since_signup and frequency counts for device_id and ip_address.

Imbalance Handling: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training set to address the highly imbalanced nature of fraud data.

🚀 Task 2: Model Building and Training

Objective: Develop and evaluate models to accurately detect fraudulent transactions.

1. Model Selection

Baseline: Logistic Regression – Established a performance floor using a linear, interpretable model.

Ensemble: Random Forest – Utilized to capture non-linear relationships and complex fraud patterns.

2. Hyperparameter Tuning

Used GridSearchCV to optimize the Random Forest (tuning n_estimators and max_depth).

Optimized specifically for AUC-PR (Area Under Precision-Recall Curve) rather than simple accuracy.

3. Evaluation & Cross-Validation

Metrics: Focused on AUC-PR, F1-Score, and Confusion Matrices.

Cross-Validation: Implemented 5-Fold Stratified Cross-Validation.

Stability: Reported mean and standard deviation for all folds to ensure model robustness.

4. Results & Justification

The Random Forest model outperformed the baseline across all minority-class metrics. It was selected for the final pipeline due to its superior ability to handle high-dimensional feature interactions and its stability during cross-validation.

🛠️ Installation & Setup

Clone the repository:


git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

Install dependencies:

pip install -r requirements.txt

Data Setup:

Place Fraud_Data.csv and IpAddress_to_Country.csv in data/raw/.

Run the Pipeline:


# This script runs preprocessing and model training
python scripts/run_modeling.py
✅ Best Practices Implemented

Leakage Prevention: Scaling and SMOTE were performed strictly within the training folds.

Modularity: Code is organized into a src/ directory for production-level reusability.

Performance: Used optimized merging techniques for geolocation data integration.

📅 Future Roadmap

Task 3: Model explainability using SHAP and LIME.

Task 4: Deployment via Flask and Docker.

Task 5: Real-time dashboarding with Streamlit.

Author: Zemichael Abraham
Status: Task 2 Completed (Modeling & Evaluation)

