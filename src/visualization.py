import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_class_distribution(df: pd.DataFrame):
    """Quantify the imbalance with Count and Pie charts."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x='class', data=df, ax=ax[0], palette='viridis')
    ax[0].set_title('Transaction Count (0: Normal, 1: Fraud)')
    
    df['class'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1], colors=['#66b3ff','#ff9999'], startangle=90)
    ax[1].set_title('Percentage of Transactions')
    plt.show()

def plot_univariate_analysis(df: pd.DataFrame):
    """Distributions of key variables."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Purchase Value Distribution
    sns.histplot(df['purchase_value'], bins=50, kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Distribution of Purchase Value')
    
    # Age Distribution
    sns.histplot(df['age'], bins=30, kde=True, ax=axes[0, 1], color='green')
    axes[0, 1].set_title('Distribution of User Age')
    
    # Source Distribution
    sns.countplot(x='source', data=df, ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('Distribution of Marketing Source')
    
    # Browser Distribution
    sns.countplot(x='browser', data=df, ax=axes[1, 1], palette='Set3')
    axes[1, 1].set_title('Distribution of Browser')
    plt.tight_layout()
    plt.show()

def plot_bivariate_analysis(df: pd.DataFrame):
    """Relationships between features and the target class."""
    # 1. Age vs Fraud
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='class', y='age', data=df, palette='Set2')
    plt.title('Age vs Fraudulent Status')
    plt.show()
    
    # 2. Purchase Value vs Fraud
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='class', y='purchase_value', data=df, palette='Set3')
    plt.title('Purchase Value vs Fraudulent Status')
    plt.show()
    
    # 3. Time Since Signup vs Fraud
    if 'time_since_signup' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.violinplot(x='class', y='time_since_signup', data=df)
        plt.title('Time Since Signup (Velocity) vs Fraud')
        plt.show()

def plot_geographic_fraud(df: pd.DataFrame):
    """Analyze fraud patterns by country."""
    plt.figure(figsize=(12, 6))
    fraud_data = df[df['class'] == 1]
    fraud_data['country'].value_counts().head(10).plot(kind='bar', color='salmon')
    plt.title('Top 10 Countries with Highest Fraud Incidents')
    plt.ylabel('Fraud Count')
    plt.show()