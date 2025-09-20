S&P 500 Daily Price Movement Prediction using Machine Learning

This project explores the challenge of predicting the daily price movement of the S&P 500 index. It follows an iterative data science workflow, starting with a simple model and progressively enhancing it with more sophisticated features and a more robust evaluation framework. The goal was to develop a model with a measurable predictive edge over a random guess.

Project Journey & Methodology
The notebook tells a story of iterative model improvement, highlighting key lessons in machine learning for time-series data.

1. Initial Model & The Need for Backtesting
I began by training a RandomForestClassifier on basic S&P 500 data (Open, High, Low, Close, Volume). A simple train-test split initially suggested an optimistic precision score of over 60%.

However, this method is often misleading for financial data. To get a more realistic assessment, I implemented a backtesting framework. This approach simulates how the model would have performed historically by training it on expanding windows of past data and testing it on subsequent periods. This reality check brought the precision down to a more honest ~52.4%, which became the baseline for improvement.

2. Feature Engineering for a Smarter Model
To give the model more context beyond simple price data, I engineered several new features designed to capture market momentum and trend. These included:

Close Price Ratios: The ratio of the current day's closing price to rolling averages over various time horizons (3, 6, 9, 300, and 1000 days).

Trend Indicators: The number of days the market went up in the preceding periods.

I also adjusted the model to be more conservative, requiring a 60% probability threshold before predicting an "up" day. These enhancements successfully pushed the backtested precision score to ~54.3%.

3. Advancing to a Neural Network (LSTM)
Recognizing that stock prices are inherently sequential, the final step was to implement a Long Short-Term Memory (LSTM) neural network. LSTMs are specifically designed to learn from patterns in time-series data.

After scaling the features and reshaping the data into 60-day sequences, the LSTM was trained and evaluated. It consistently outperformed the Random Forest, achieving a precision score in the 55-58% range, demonstrating the value of using a sequence-aware architecture for this type of forecasting task.

Technology Stack
Python 3.x

Pandas & NumPy for data manipulation

yfinance for downloading historical stock data

Scikit-learn for the RandomForestClassifier and evaluation metrics

TensorFlow & Keras for building and training the LSTM model

Matplotlib & Seaborn for data visualization

How to Use This Repository
Clone the repository to your local machine.

Install the required libraries:

pip install yfinance pandas scikit-learn tensorflow matplotlib seaborn

Open and run the Stock_Market_Predictor.ipynb notebook in a Jupyter environment.

Key Takeaways
Backtesting is crucial for an honest evaluation of financial forecasting models.

Thoughtful feature engineering can significantly improve model performance by providing essential context.

Sequence-aware models like LSTMs are highly effective for time-series data and can offer a superior predictive edge compared to traditional models.
