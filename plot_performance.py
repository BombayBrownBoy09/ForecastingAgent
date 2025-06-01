import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_store_forecast(store_id):
    """
    Plot actual vs. forecasted sales for a specific store (tcin) over the validation period.
    Saves a PNG file and displays the plot.
    """
    # 1) Read the validation report generated earlier
    df = pd.read_csv('results/rossmann_validation_report.csv', parse_dates=['date'])

    # 2) Filter for the chosen store (tcin) and sort by date
    store_df = df[df['tcin'] == store_id].sort_values('date')

    # 3) Create a line plot: Actual vs Final Forecast
    plt.figure(figsize=(10, 6))
    plt.plot(store_df['date'], store_df['units_sold'],
             label='Actual', marker='o', linestyle='-')
    plt.plot(store_df['date'], store_df['yhat_final'],
             label='Forecast', marker='x', linestyle='--')

    plt.title(f'Store {store_id}: Actual vs. Forecast')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 4) Save to PNG
    plt.savefig(f'results/store_{store_id}_forecast.png')
    plt.show()


def plot_error_distribution():
    """
    Plot the distribution of forecast errors (actual - final forecast)
    across all stores in the validation set. Saves a PNG, displays it, 
    and prints the directory contents to confirm where the file is saved.
    """
    # 1) Load validation report and compute error
    df = pd.read_csv('results/rossmann_validation_report.csv')
    df['error'] = df['units_sold'] - df['yhat_final']

    # 2) Create the histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df['error'], bins=50, edgecolor='black')
    plt.title('Error Distribution (Actual − Forecast)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

    # 3) Save the figure
    filename = 'results/error_distribution.png'
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    # === CONFIGURE THIS ===
    sample_store_id = 1

    # Generate and show the "Actual vs Forecast" plot for store=1
    plot_store_forecast(sample_store_id)

    # Generate and show the overall "Error Distribution" histogram
    plot_error_distribution()
