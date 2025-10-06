import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class AmountTimeScaler(BaseEstimator, TransformerMixin):
    """Standardize only Amount (+ optionally Time); leave V1..V28 unchanged."""
    def __init__(self, scale_time=True):
        self.scale_time = scale_time
        self.scaler_amount = StandardScaler()
        self.scaler_time = StandardScaler() if scale_time else None

    def fit(self, X, y=None):
        self.scaler_amount.fit(X[["Amount"]])
        if self.scale_time:
            self.scaler_time.fit(X[["Time"]])
        return self

    def transform(self, X):
        X = X.copy()
        X["Amount"] = self.scaler_amount.transform(X[["Amount"]])
        if self.scale_time:
            X["Time"] = self.scaler_time.transform(X[["Time"]])
        return X

def split_X_y(df, label_col="Class"):
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col])
    return X, y
