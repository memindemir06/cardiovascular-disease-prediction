import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def select_features(X, y, k=10):
    """
    Select the top k features using chi-squared test.
    
    Parameters:
    X (DataFrame): The input data
    y (Series): The target variable
    k (int): The number of top features to select

    Returns:
    DataFrame: The input data with only the top k features
    """
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    cols = selector.get_support(indices=True)
    return X.iloc[:, cols]