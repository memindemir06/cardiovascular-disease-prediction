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
    df.drop("id", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def preprocess_data(df):
    """
    Preprocess the data.
    
    :param df: DataFrame, the data to be preprocessed
    :return: DataFrame, the preprocessed data
    """
    # Calculate the Body Mass Index (BMI) for each row in the DataFrame
    df["bmi"] = df["weight"] / (df["height"]/100)**2
    # Drop the height and weight columns
    df.drop(["height", "weight"], axis=1, inplace=True)
    # Convert age from days to years
    df["age"] = df["age"] / 365
    # Convert age to integer
    df["age"] = df["age"].astype(int)
    
    # Remove outliers
    out_filter = ((df["ap_hi"]>250) | (df["ap_lo"]>200))
    df = df[~out_filter]
    out_filter2 = ((df["ap_hi"] < 0) | (df["ap_lo"] < 0))
    df = df[~out_filter2]
    out_filter3 = (df["ap_hi"] < df["ap_lo"])
    df = df[~out_filter3]


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