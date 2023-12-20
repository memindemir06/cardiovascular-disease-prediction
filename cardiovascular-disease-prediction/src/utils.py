import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from a csv file.
    """
    return pd.read_csv(file_path)

def save_data(df, file_path):
    """
    Save a DataFrame to a csv file.
    """
    df.to_csv(file_path, index=False)

def plot_correlation_matrix(df):
    """
    Plot a correlation matrix for a DataFrame.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.show()