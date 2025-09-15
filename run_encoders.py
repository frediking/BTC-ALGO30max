import pandas as pd
from deep_learning_btc import RobustCategoryEncoder
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import KFold

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
csv_path = 'Bitcoin_Historical_Data_Enhanced.csv'
output_path = 'Bitcoin_Historical_Data_Enhanced.csv'
#'encoded_bitcoin_data.csv'

def verify_data_quality(df):
    """Verify data quality before encoding with focus on training suitability."""
    logging.info("Performing data quality checks...")
    
    # 1. Check for missing values in Profit/Loss and Close (needed for encoding)
    missing_pl = df['Profit/Loss'].isnull().sum()
    missing_close = df['Close'].isnull().sum()
    if missing_pl > 0 or missing_close > 0:
        logging.warning(f"Missing values found: Profit/Loss: {missing_pl}, Close: {missing_close}")
    
    # 2. Verify expected values and class balance
    valid_values = {'PROFIT', 'LOSS'}
    non_null_values = df['Profit/Loss'].dropna().unique()
    invalid_values = set(non_null_values) - valid_values
    if invalid_values:
        logging.error(f"Found unexpected values in Profit/Loss: {invalid_values}")
        raise ValueError("Invalid values found in Profit/Loss column")
    
    # Check class balance
    class_counts = df['Profit/Loss'].value_counts()
    class_ratio = class_counts.min() / class_counts.max()
    if class_ratio < 0.1:  # Severe class imbalance
        logging.warning(f"Severe class imbalance detected. Ratio: {class_ratio:.2f}")
        logging.warning(f"Class distribution: {class_counts.to_dict()}")
    
    # 3. Check for sequential integrity (no gaps in time series)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        date_diffs = df['Date'].diff().value_counts()
        if len(date_diffs) > 1:
            logging.warning("Irregular time intervals detected in data")
    
    logging.info("Data quality checks completed")

def main():
    try:
        # 1. Check if input file exists
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Input file not found: {csv_path}")

        # 2. Load and analyze the data
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

        # 3. Verify data quality
        verify_data_quality(df)

        # 4. Create backup of original data
        df_original = df.copy()

        # 5. Add fold column for target mean encoding
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df['fold'] = -1
        for fold_number, (_, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold_number

        # 6. Encode Profit/Loss column
        logging.info("Starting encoding process...")
        
        # Binary encoding
        df['Profit/Loss_binary'] = df['Profit/Loss'].map({'LOSS': 0, 'PROFIT': 1})
        
        # Target mean encoding
        try:
            encoder = RobustCategoryEncoder(df, n_splits=5, smoothing=10)
            df['Profit/Loss_target_mean'] = encoder.target_mean_encode(
                df, 'Profit/Loss', 'Close'
            )
        except Exception as e:
            logging.error(f"Error during target mean encoding: {str(e)}")
            # Fallback to just binary encoding if target mean fails
            df = df.drop(columns=['Profit/Loss_target_mean'], errors='ignore')
            logging.info("Proceeding with binary encoding only")

        # 7. Drop unnecessary columns
        df = df.drop(columns=['Profit/Loss','Dividends','Stock Splits','Daily Returns','%Return','Adj_Close','fold'])
        logging.info("Original Profit/Loss and fold columns dropped")

        # 8. Verify no data loss
        if len(df) != len(df_original):
            raise ValueError("Data loss detected during encoding")

        # 9. Save encoded dataset
        df.to_csv(output_path, index=False)
        logging.info(f"Encoded data saved to {output_path}")

        # 10. Print summary statistics
        print("\nEncoding Summary:")
        print("-" * 50)
        print(f"Original columns: {len(df_original.columns)}")
        print(f"Final columns: {len(df.columns)}")
        print("\nSample of encoded values:")
        encoded_cols = [col for col in df.columns if 'Profit/Loss' in col]
        print(df[encoded_cols].head())
        
        # 11. Verify file was saved
        if not Path(output_path).exists():
            raise FileNotFoundError("Failed to save encoded data")

    except Exception as e:
        logging.error(f"Error during encoding process: {str(e)}")
        # Restore original data if something went wrong
        if 'df_original' in locals():
            df_original.to_csv(output_path, index=False)
            logging.info("Restored original data due to error")
        raise

if __name__ == "__main__":
    main()