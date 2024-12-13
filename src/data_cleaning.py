import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """
    Clean the dataset by handling missing values, combining relevant columns,
    and encoding labels.
    Args:
        df (pd.DataFrame): The input DataFrame to clean.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        print("Error: DataFrame is empty.")
        return None
    
    # Fill missing values for 'title' and 'text' columns with empty strings
    df['title'].fillna('', inplace=True)
    df['text'].fillna('', inplace=True)

    # Combine title and text into one column 'content'
    df['content'] = df['title'] + " " + df['text']

    # Encode the labels (assuming 'label' column exists)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    return df

def save_data(df, file_path):
    """Save the cleaned DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Cleaned data saved to {file_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

def main():
    # Load raw data
    try:
        test_data = pd.read_csv('data/processed/raw_data.csv', sep=',', encoding='utf-8', on_bad_lines='skip')
        print(f"Raw data loaded successfully. Shapes:{test_data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Clean the data
    test_data = clean_data(test_data)

    # Save cleaned data
    save_data(test_data, 'data/processed/cleaned_data.csv')

if __name__ == "__main__":
    main()
