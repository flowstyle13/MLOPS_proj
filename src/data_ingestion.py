import pandas as pd
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file to be loaded.
    
    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    data = pd.read_csv(file_path, sep=';')  # Adjust separator if needed
    return data

def main():
    # Specify the file path
    input_file = 'data/raw_news_bbc.csv'
    
    # Load the dataset
    try:
        data = load_data(input_file)
        print(f"Data loaded successfully. Shape: {data.shape}")
        
        # Save the data to a processed directory (for example)
        data.to_csv('data/processed/raw_data.csv', index=False)
        print("Data saved to 'data/processed/raw_data.csv'")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    main()
