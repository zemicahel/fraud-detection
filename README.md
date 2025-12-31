# Fraud Detection for E-commerce and Banking

## 📌 Project Overview
This project focuses on building a robust fraud detection system for **Adey Innovations Inc.** By integrating geolocation data and analyzing transaction patterns, we identify high-risk activities to minimize financial losses in e-commerce and banking environments.

---

## 📂 Project Structure
```text

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


Author: Zemichael Abraham
Status: Task 2 Completed (Modeling & Evaluation)





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
│   ├── modeling.ipynb # Task 2: modeling
│   ├── shap-explainability.ipynb # Task 3: shap$explainability
│   └── README.md            # Notebook descriptions
├── src/                     # Modular Source Code
│   ├── __init__.py
│   ├── data_preprocessing.py # Cleaning & Geolocation Integration
│   ├── feature_engineering.py# Transformation & SMOTE
│   ├── model_training.py    # Training & Evaluation logic
│   ├── explainability.py    # SHAP & Feature Importance logic
│   └── visualization.py     # EDA plotting logic
├── tests/                   # Automated tests
├── models/                  # Saved ML model artifacts (.pkl)
├── scripts/                 # Standalone production scripts
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation


🚀 Task 1: Data Analysis and Preprocessing

Objective: Clean data and engineer features to prepare for machine learning.

Data Cleaning: Handled missing values and removed duplicates to ensure data integrity.

Geolocation Integration: Mapped IP addresses to countries using high-performance range lookups.

Feature Engineering: Extracted temporal patterns (hour_of_day) and transaction velocity (time_since_signup, device_count).

Imbalance Handling: Applied SMOTE to the training set to address the sparse nature of fraudulent transactions.

🚀 Task 2: Model Building and Training

Objective: Develop and evaluate models to accurately detect fraudulent transactions.

Model Selection: Established a baseline with Logistic Regression and achieved superior performance with a tuned Random Forest ensemble.

Evaluation: Focused on AUC-PR and F1-Score to prioritize the detection of the minority fraud class.

Cross-Validation: Implemented 5-Fold Stratified Cross-Validation to ensure model stability and robustness.

🚀 Task 3: Model Explainability

Objective: Interpret model decisions using SHAP to provide actionable business recommendations.

Feature Importance:

Identified time_since_signup, device_count, and purchase_value as the top drivers of fraud.

Compared built-in Gini importance with SHAP global importance for a comprehensive view.

SHAP Analysis:

Global: Generated Summary and Bar plots to visualize feature impact across the entire dataset.

Local: Generated Waterfall plots for specific cases: True Positives (Correct catches), False Positives (False alarms), and False Negatives (Missed fraud).

Business Recommendations:

Velocity Cool-down: Implement a 5-minute cooling-off period or mandatory MFA for transactions occurring immediately after signup.

Device Fingerprinting: Flag accounts where a single device_id is linked to more than 2 unique users within a 24-hour window.

Proxy/VPN Alert: Trigger additional verification for transactions originating from IPs with high user-association counts (ip_count).

🛠️ Installation & Setup

Clone the repository:

code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

Install dependencies:

code
Bash
download
content_copy
expand_less
pip install -r requirements.txt

Data Setup:

Place Fraud_Data.csv and IpAddress_to_Country.csv in data/raw/.

Run the Pipeline:

code
Bash
download
content_copy
expand_less
# Runs preprocessing, training, and explainability analysis
python scripts/run_modeling.py
✅ Best Practices Implemented

Interpretability: Moving beyond "black-box" models by using SHAP to explain individual transaction risks.

Leakage Prevention: Scaling, SMOTE, and feature selection were performed strictly within training folds.

Modularity: Clean separation of data logic, modeling logic, and explainability logic in the src/ directory.

📅 Future Roadmap

Task 4: Deployment and API development via Flask or FastAPI.

Task 5: Real-time monitoring dashboard with Streamlit.

Author: Zemichael Abraham
Status: Task 3 Completed (Explainability & Insights)

code
Code
download
content_copy
expand_less