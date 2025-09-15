import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def analyze_weekly_returns():
    """Present weekly Bitcoin return predictions in a user-friendly format"""
    
    # Load daily predictions
    predictions = pd.read_csv('rnn_predictions.csv')
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    
    # Calculate weekly cumulative returns
    predictions['Week'] = predictions['Date'].dt.isocalendar().week
    weekly_returns = predictions.groupby('Week').agg({
        'Date': 'first',  # Get week start date
        'Predicted_Return': lambda x: (1 + x/100).prod() - 1  # Compound daily returns
    }).reset_index()
    
    # Create visual summary
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(weekly_returns)), 
            weekly_returns['Predicted_Return'] * 100,
            color=['green' if x > 0 else 'red' for x in weekly_returns['Predicted_Return']])
    plt.title('Predicted Weekly Bitcoin Returns')
    plt.xlabel('Weeks')
    plt.ylabel('Expected Return (%)')
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    plt.savefig('weekly_bitcoin_returns.png')
    
    # Create user-friendly summary
    print("\n=== Weekly Bitcoin Return Forecast ===\n")
    for _, week in weekly_returns.iterrows():
        direction = "📈 Increase" if week['Predicted_Return'] > 0 else "📉 Decrease"
        print(f"Week of {week['Date'].strftime('%B %d')}:")
        print(f"Expected Change: {direction}")
        print(f"Predicted Return: {week['Predicted_Return']*100:.1f}%")
        print("-" * 40)
    
    # Show risk disclaimer
    print("\n⚠️ Important Note:")
    print("These are predictions based on historical patterns.")
    print("Actual returns may vary significantly.")
    print("Always invest responsibly and consider consulting a financial advisor.")

if __name__ == "__main__":
    analyze_weekly_returns()