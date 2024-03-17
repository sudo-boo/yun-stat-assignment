import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

# Function to read data from JSON files
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['result']

# Function to interpolate data to ensure equal length
def interpolate_data(data, target_length):
    dates = [entry['date'] for entry in data]
    straddle_prices = [float(entry['close']) for entry in data]

    # Interpolate data to match the target length
    f = interp1d(np.arange(len(straddle_prices)), straddle_prices, kind='linear')
    interpolated_prices = f(np.linspace(0, len(straddle_prices) - 1, target_length))

    return interpolated_prices

# Function to calculate correlation between two indices
def calculate_correlation(index1_data, index2_data):
    max_length = max(len(index1_data), len(index2_data))

    index1_straddle_prices = interpolate_data(index1_data, max_length)
    index2_straddle_prices = interpolate_data(index2_data, max_length)

    correlation, _ = pearsonr(index1_straddle_prices, index2_straddle_prices)
    return correlation

# Main function
def main():
    # Read data from JSON files
    banknifty_data = read_json('banknifty_data_1.json')
    finnifty_data = read_json('finnifty_data_1.json')

    # Calculate correlation between BANKNIFTY and FINNIFTY
    correlation = calculate_correlation(banknifty_data, finnifty_data)
    print("Correlation between BANKNIFTY and FINNIFTY:", correlation)

    # Plot straddle prices
    plt.figure(figsize=(10, 6))
    plt.plot(interpolate_data(banknifty_data, len(finnifty_data)), label='BANKNIFTY')
    plt.plot(interpolate_data(finnifty_data, len(banknifty_data)), label='FINNIFTY')
    plt.title('Interpolated Straddle Prices Comparison')
    plt.xlabel('Time')
    plt.ylabel('Straddle Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('correlation.png')

if __name__ == "__main__":
    main()
