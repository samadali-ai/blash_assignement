# load_data.py
import pandas as pd

def load_csv(file_path):
    """
    Load the CSV file into a pandas DataFrame.
    :param file_path: Path to the CSV file.
    :return: DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    df = load_csv('/home/abdulsamad/blaash_assignement/Procore_Subcontractor_Invoice_20_Duplicates.csv')
    if df is not None:
        print(df.head())
