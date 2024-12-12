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
    # Example cleaning steps:
    
    # Handling missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    
    # Feature engineering (if needed)
    # For instance, let's assume we want to add a new column:
    if 'price' in df.columns:
        df['price_log'] = np.log1p(df['price'])  # Log transformation for price
    
    return df

def main():
    # Load the raw data (assuming it's already loaded and saved as raw_data.csv)
    try:
        raw_data = pd.read_csv('data/processed/raw_data.csv', sep=';')
        print(f"Raw data loaded successfully. Shape: {raw_data.shape}")
        
        # Clean the data
        cleaned_data = clean_data(raw_data)
        print(f"Data cleaned. Shape: {cleaned_data.shape}")
        
        # Save cleaned data
        cleaned_data.to_csv('data/processed/cleaned_data.csv', index=False)
        print("Cleaned data saved to 'data/processed/cleaned_data.csv'")
    except Exception as e:
        print(f"Error cleaning data: {e}")

if __name__ == "__main__":
    main()
