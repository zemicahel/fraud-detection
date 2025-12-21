# Fraud Detection for E-commerce and Banking

## 📌 Project Overview
This project focuses on building a robust fraud detection system for **Adey Innovations Inc.** and an e-commerce platform. By integrating geolocation data and analyzing transaction patterns, we identify high-risk activities to minimize financial losses.

---

## 📂 Project Structure
To maintain professional standards, this repository follows a modular structure:

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
│   └── visualization.py     # EDA plotting logic
├── tests/                   # Automated tests
├── models/                  # Saved ML model artifacts (.pkl)
├── scripts/                 # Standalone production scripts
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
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

Data Placement:

Place Fraud_Data.csv and IpAddress_to_Country.csv into the data/raw/ directory.

🚀 Task 1: Data Analysis and Preprocessing
1a. Data Cleaning & EDA

Modules: src/data_preprocessing.py, src/visualization.py

Cleaning: Handled missing values and duplicates. Removed nulls in critical fields (device_id, ip_address) to ensure identity integrity.

Geolocation Integration: Performed range-based lookup using pd.merge_asof to map transaction IP addresses to their corresponding countries.

Visualizations Included:

Class Distribution: Quantified the imbalance (Normal vs. Fraud).

Univariate Analysis: Distributions of age, purchase_value, source, and browser.

Bivariate Analysis: Relationships between features and the target fraud class.

Geographic Analysis: Top 10 countries by fraud count.

1b. Feature Engineering & Transformation

Module: src/feature_engineering.py

New Features:

hour_of_day & day_of_week: Captures temporal patterns.

time_since_signup: Detects automated "instant" fraud.

device_count & ip_count: Measures transaction frequency/velocity.

Transformation:

Encoding: One-Hot Encoding for categorical variables.

Scaling: StandardScaler applied to all numerical features.

Class Imbalance Handling:

Applied SMOTE strictly to the training set.

Justification: SMOTE creates synthetic samples rather than duplicates, improving the model's ability to learn the minority class boundary without losing data.

✅ Best Practices Implemented

Separation of Concerns: EDA code is separated from engineering logic.

Data Leakage Prevention: Stratified splitting was performed before any oversampling or scaling.

Modularity: All core functions are defined in src/ for reusability.

PEP 8 Compliance: Code follows professional Python naming and documentation standards.

🚀 Future Tasks

Task 2: Model training and evaluation.

Task 3: Model explainability using SHAP.

Task 4: Deployment and Dashboarding.

Author: Zemicahel Abraham
Status: Task 1 Completed

