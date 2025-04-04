import pandas as pd
import os

# Path to May data file
base_dir = os.path.abspath(os.path.dirname(__file__))
may_data_path = os.path.join(base_dir, '..', '..', 'compressore_1', 'compressore1_predictions_onlymay.xlsx')

print(f"Loading May data from: {may_data_path}")

try:
    # Read the data
    may_data = pd.read_excel(may_data_path)
    
    # Display the first few rows to understand structure
    print("\nFirst 5 rows of the Excel file:")
    print(may_data.head())
    
    # Check columns
    print("\nColumns in the file:")
    print(may_data.columns.tolist())
    
    # Check data types
    print("\nData types:")
    print(may_data.dtypes)
    
    # If datetime column exists but in wrong format:
    if 'DateTime' in may_data.columns:
        print("\nConverting DateTime to proper format...")
        
        # Try to convert to datetime
        may_data['DateTime'] = pd.to_datetime(may_data['DateTime'], errors='coerce')
        
        # Check for NaT values which indicate conversion failure
        if may_data['DateTime'].isna().any():
            print(f"Warning: {may_data['DateTime'].isna().sum()} rows had invalid date formats")
        
        # Set as index
        may_data.set_index('DateTime', inplace=True)
        
        # Try the filtering that was failing
        print("\nAttempting to filter May data...")
        may_2024 = may_data.copy()
        
        # Update the year to 2024 if needed
        if may_2024.index.year[0] != 2024:
            print(f"Original year: {may_2024.index.year[0]}, updating to 2024")
            may_2024.index = may_2024.index.map(lambda x: x.replace(year=2024))
        
        # Now filter for May
        may_filter = (may_2024.index >= '2024-05-01') & (may_2024.index <= '2024-05-31 23:59:59')
        may_filtered = may_2024[may_filter]
        
        print(f"Successfully filtered {len(may_filtered)} records for May 2024")
        
        # Save the corrected file
        output_path = os.path.join(base_dir, 'may_data_2024_corrected.xlsx')
        may_filtered.reset_index().to_excel(output_path, index=False)
        print(f"\nSaved corrected May data to: {output_path}")
    else:
        print("\nNo DateTime column found in the file")
        
except Exception as e:
    print(f"Error processing the file: {e}")