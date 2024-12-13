import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean the dataset by performing operations like handling missing values and basic feature engineering.
    Args:
        df (pd.DataFrame): The input DataFrame to clean.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        print("Error: DataFrame is empty.")
        return None

    # Example cleaning steps
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.fillna(method='bfill', inplace=True)  # Backward fill missing values


def main():
    try:
        raw_data = pd.read_csv(
            'data/processed/raw_data.csv',
            sep=';',  # Adjust delimiter if needed
            on_bad_lines='skip',
            encoding='utf-8'
        )
        print(f"Raw data loaded successfully. Shape: {raw_data.shape}")
        print(raw_data.head())
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        return
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return



    # Clean the data
    cleaned_data = clean_data(raw_data)
    if cleaned_data is None:
        print("Error: clean_data returned None.")
        return

    try:
        cleaned_data.to_csv('data/processed/cleaned_data.csv', index=False)
        print("Cleaned data saved to 'data/processed/cleaned_data.csv'")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

if __name__ == "__main__":
    main()
