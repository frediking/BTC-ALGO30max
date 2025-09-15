## Detailed Report on Rebuilding the CNN-LSTM Model with Boruta Feature Selection

To reconstruct the CNN-LSTM model with Boruta feature selection for Bitcoin price direction prediction, as detailed in the 2024 Financial Innovation study by Omole and Enke, we need to meticulously replicate the data collection, preprocessing, feature selection, model architecture, training, and evaluation processes. 
The study reported an accuracy of 82.44% and a remarkable 6654% annual return using a long-and-short trading strategy. Below is a comprehensive guide to rebuilding a similar model, aiming for an accuracy of 80.44-82.44%, including all necessary details and a sample implementation code.

1. **Data Collection**
The study utilized a dataset spanning February 6, 2013, to February 18, 2023 (3665 days), comprising Bitcoin closing prices and 87 on-chain metrics sourced from Glassnode. These metrics capture various aspects of Bitcoin’s blockchain activity, such as transaction volumes, network activity, and investor behavior. The Boruta feature selection method reduced these to 26 key features, listed belo


FEATURE NAMES:  
90-day coin days destroyed
Adjusted SOPR
Average coin dormancy
Average spent output lifespan
Coin days destroyed
Coin years destroyed
Cumulative value days destroyed
Difficulty ribbon compression
Entity-adjusted dormancy flow
All exchange net position change (BTC)
HODL waves (1W-1M)
HODL Waves (24H)
HODL waves (2Y-3Y)
HODL waves (3M-6M)
Inflation rate
Issuance (BTC)
Median Spent output lifespan
MVRV Z-score
Net realized profit/loss (USD)
All exchanges net transfer volume (BTC)
Net unrealized profit/loss
Percent of UTXOs in profit
Percent of supply in profit
Realized loss (USD)
Realized profit/loss ratio
Realized profit (USD)

Action: These Metrics are available, write code in that way. 
        Give brief but detailed explanation about how they are used


2. **Data Preprocessing**

Missing Data Handling: The study addressed missing data using listwise deletion for data missing completely at random (MCAR) and regression imputation for data missing not at random (MNAR). Apply similar techniques to ensure data integrity.
Target Variable: The target is a binary classification variable:

1: If the next day’s closing price is higher than the current day’s (<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>y</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>></mo><msub><mi>y</mi><mi>t</mi></msub></mrow><annotation encoding="application/x-tex"> y_{t+1} > y_t </annotation></semantics></math>).
0: If the next day’s closing price is less than or equal to the current day’s (<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>y</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>≤</mo><msub><mi>y</mi><mi>t</mi></msub></mrow><annotation encoding="application/x-tex"> y_{t+1} \leq y_t </annotation></semantics></math>).


**Data Split**:
Split the dataset into 80% training and 20% testing, maintaining chronological order (no shuffling) to preserve temporal dependencies.
Input Structure: Structure the input as a 3D array (samples, time steps, features), where time steps represent a window of past days. The study tested window sizes of 3, 5, 7, 14, and 30 days, but the exact size for the 82.44% accuracy was not specified. A window size of 14 days is a reasonable starting point, as it balances short- and medium-term trends.

3. **Feature Selection with Boruta**
The Boruta algorithm, a wrapper method based on random forest, was used to select 26 features from the initial 87. Boruta identifies features by comparing their importance to that of shadow features (randomized copies). The selected features (listed above) were deemed most relevant for predicting price direction.
Action: If you have access to the full set of 87 metrics, apply the Boruta algorithm to select features. Alternatively, use the 26 features directly, as they were validated in the study. Python’s BorutaPy library can be used if feature selection is needed.

4. **Model Architecture**
The CNN-LSTM model combines convolutional layers to extract spatial features and LSTM layers to capture temporal dependencies. The architecture is as follows:

**CNN Component**:

1D Convolutional layer: 64 filters, kernel size 3, ReLU activation.
Batch Normalization for training stability.
Average Pooling layer: pool size 1.
Dropout: 0.5 to prevent overfitting.


**LSTM Component**:

First LSTM layer: 128 units, TanH activation, followed by Batch Normalization and Dropout (0.5).
Second LSTM layer: 80 units, TanH activation, followed by Batch Normalization and Dropout (0.5).


Output Layer:

Dense layer: 1 unit, sigmoid activation for binary classification.



5. **Model Training**

Data Split: Use 80% of the data (approximately 2932 days) for training and 20% (733 days) for testing.
Hyperparameters: The study used random search for optimization, exploring ranges for:

Learning rate
Batch size
Number of epochs
Dropout coefficient (fixed at 0.5 in the architecture)


Suggested Starting Values:

Learning rate: 0.001
Batch size: 32
Epochs: 50


Optimization: Use an optimizer like Adam, common for deep learning models. Perform hyperparameter tuning to optimize performance, as the exact values for the 82.44% accuracy were not specified.

6. **Evaluation Metrics**
The model’s performance was evaluated using:

Accuracy: 82.44%
Recall: 0.8078
Precision: 0.8309
F1-Score: 0.8192
AUC-ROC: 0.8242
MCC: 0.6489

Action: Evaluate your model using these metrics to ensure comparability. Aim for an accuracy of 80.44-82.44%, adjusting window size and hyperparameters as needed.

7. **Backtesting and Trading Strategy**
The study backtested three trading strategies over the last 729 days with a starting capital of $1000, 30% tax, and 0.5% transaction cost:

Long-and-Short Buy-and-Sell: Buy if the model predicts a price increase, sell if a decrease, taking both long and short positions. This yielded a 6653.7497% annual return, with a Sharpe ratio of 1.8583 and maximum drawdown (MDD) of -0.0704.
Long-Only Buy-and-Sell: 437.2461% annual return.
Short-Only Buy-and-Sell: 1084.16% annual return.

Note: Replicating the 6654% return may be challenging due to market changes and transaction costs. Focus on achieving high prediction accuracy first.

ADD ALL DEPENDENCIES NEEDED