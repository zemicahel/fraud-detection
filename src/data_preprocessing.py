import pandas as pd
import numpy as np
import sklearn.utils.validation

# Compatibility Patch for SMOTE/Sklearn
if not hasattr(sklearn.utils.validation, '_is_pandas_df'):
    def _is_pandas_df(X):
        return hasattr(X, "columns") and hasattr(X, "index")
    sklearn.utils.validation._is_pandas_df = _is_pandas_df

def load_data(fraud_path, ip_path):
    return pd.read_csv(fraud_path), pd.read_csv(ip_path)

def clean_data(df):
    """Handles missing values, duplicates, and type correction."""
    df = df.dropna()
    df = df.drop_duplicates()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def merge_with_ip(fraud_df, ip_df):
    """Integrates geolocation data using optimized merge_asof."""
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(float)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(float)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(float)

    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')

    merged = pd.merge_asof(
        fraud_df, ip_df, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address'
    )

    merged['country'] = np.where(
        (merged['ip_address'] >= merged['lower_bound_ip_address']) & 
        (merged['ip_address'] <= merged['upper_bound_ip_address']),
        merged['country'], 'Unknown'
    )
    return merged.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)