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
    # Handling missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.fillna(method='bfill', inplace=True)  # Backward fill missing values (optional)
    
    # Feature engineering
    if 'price' in df.columns:
        df['price_log'] = np.log1p(df['price'].replace(0, np.nan))  # Avoid log(0) errors
    
    return df

def main():
    # Load the raw data
    try:
        raw_data = pd.read_csv(
            'data/processed/raw_data.csv', 
            sep=';', 
            on_bad_lines='skip',  # Skips problematic lines
            encoding='utf-8'      # Ensures consistent encoding
        )
        print(f"Raw data loaded successfully. Shape: {raw_data.shape}")
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        return
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Validate the presence of critical columns
    if 'price' not in raw_data.columns:
        print("Error: The required column 'price' is missing from the dataset.")
        return
    
    # Clean the data
    try:
        cleaned_data = clean_data(raw_data)
        print(f"Data cleaned. Shape: {cleaned_data.shape}")
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return
    
    # Save cleaned data
    try:
        cleaned_data.to_csv('data/processed/cleaned_data.csv', index=False)
        print("Cleaned data saved to 'data/processed/cleaned_data.csv'")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

if __name__ == "__main__":
    main()
