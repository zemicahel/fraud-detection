


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

git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

Install dependencies:


pip install -r requirements.txt

Data Setup:

Place Fraud_Data.csv and IpAddress_to_Country.csv in data/raw/.

Run the Pipeline:


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

