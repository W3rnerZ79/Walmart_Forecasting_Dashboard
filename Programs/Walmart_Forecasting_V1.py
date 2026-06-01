import pandas as pd
import numpy as np

def load_data(file_path):

    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
def weighted_moving_average(data, window_size):
    weights = np.array([0.15, 0.35, 0.5])
    
    # 1. Sort the entire dataframe by Date first so the timeline is sequential
    data = data.sort_values(by=['Store', 'Date']).reset_index(drop=True)
    
    # 2. THE FIX: Use groupby directly with a lambda function inside .transform()
    # This calculates the WMA per store and returns ONLY the new values, 
    # leaving your original 'Store' and 'Date' columns completely untouched.
    data['Weighted_Moving_Average'] = (
        data.groupby('Store')['Weekly_Sales']
        .rolling(window=window_size)
        .apply(lambda x: np.dot(x, weights), raw=True)
        .reset_index(level=0, drop=True)
        .groupby(data['Store'])
        .shift(1) # Drops the temporary groupby index layer
    )
    
    return data

def main():
    # Load the dataset
    file_path = r'C:\Users\zackc\Develop\supply_chain_project\Walmart.csv'
    data = load_data(file_path)
    if data is not None:
        # Calculate the weighted moving average with a window size of 3
        df_final = weighted_moving_average(data, window_size=3)
        df_final['Forecast_Error'] = df_final['Weekly_Sales'] - df_final['Weighted_Moving_Average']
        df_final['Absolute_Error'] = df_final['Forecast_Error'].abs()

        df_final = df_final.sort_values(by=['Store', 'Date'])
        output_path = r'C:\Users\zackc\Develop\supply_chain_project\Walmart_Forecasted.csv'
        df_final.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()