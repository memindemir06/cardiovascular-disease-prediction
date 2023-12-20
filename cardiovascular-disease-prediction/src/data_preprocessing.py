import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load dataset from a CSV file.
    
    :param filepath: str, path to the CSV file
    :return: DataFrame, the loaded data
    """
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Clean the data.
    
    :param df: DataFrame, the data to be cleaned
    :return: DataFrame, the cleaned data
    """
    # Implement data cleaning steps if necessary
    return df

def preprocess_data(df):
    """
    Preprocess the data.
    
    :param df: DataFrame, the data to be preprocessed
    :return: DataFrame, the preprocessed data
    """
    # Implement data preprocessing steps like encoding categorical variables, imputation, etc.
    return df

def scale_data(df):
    """
    Scale the data.
    
    :param df: DataFrame, the data to be scaled
    :return: DataFrame, the scaled data
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled