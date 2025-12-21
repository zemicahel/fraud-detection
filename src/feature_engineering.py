from typing import NamedTuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class PreparedData(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time-based and velocity features.
    """
    df = df.copy()

    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds()

    # Transaction frequency / velocity
    df['device_count'] = df.groupby('device_id')['device_id'].transform('count')
    df['ip_count'] = df.groupby('ip_address')['ip_address'].transform('count')

    return df


def prepare_data(df: pd.DataFrame) -> PreparedData:
    """
    Encodes categorical variables, splits the dataset,
    scales numerical features, and balances the training data using SMOTE.
    """
    features = [
        'purchase_value', 'age', 'hour_of_day', 'day_of_week',
        'time_since_signup', 'device_count', 'ip_count',
        'source', 'browser', 'sex'
    ]

    X = df[features].copy()
    y = df['class'].copy()

    # Explicit numeric casting (prevents pandas FutureWarning)
    num_cols = [
        'purchase_value', 'age', 'hour_of_day', 'day_of_week',
        'time_since_signup', 'device_count', 'ip_count'
    ]
    X[num_cols] = X[num_cols].astype('float64')

    # One-Hot Encoding
    X = pd.get_dummies(
        X,
        columns=['source', 'browser', 'sex'],
        drop_first=True
    )

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling (fit only on training data)
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Handle class imbalance (training data only)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return PreparedData(
        X_train=X_train_res,
        X_test=X_test,
        y_train=y_train_res,
        y_test=y_test
    )
