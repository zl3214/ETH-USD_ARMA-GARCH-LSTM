import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from arch import arch_model
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("/Users/lizan/Desktop/ETH-USD (1).csv")

    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    def plot_stock_price(df):

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Adj Close'], label='Close')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Stock Price')
        ax.legend()
        plt.show()

    eth_price_diff1 = data['Adj Close'].diff().dropna()
    eth_price_diff2 = eth_price_diff1.diff().dropna()


    plot_stock_price(data)

    # ACF, PACF
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(data['Adj Close'], ax=axes[0], title='ACF of Original ETH-USD Price', lags=np.arange(1, 41))
    plot_pacf(data['Adj Close'], ax=axes[1], title='PACF of Original ETH-USD Price', lags=np.arange(1, 41))
    plt.tight_layout()
    plt.show()


    # 差分1
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(eth_price_diff1, ax=axes[0], title='ACF of 1st Order Differenced ETH-USD Price', lags=np.arange(1, 41))
    plot_pacf(eth_price_diff1, ax=axes[1], title='PACF of 1st Order Differenced ETH-USD Price', lags=np.arange(1, 41))
    plt.tight_layout()
    plt.show()

    # 差分2
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(eth_price_diff2, ax=axes[0], title='ACF of 2nd Order Differenced ETH-USD Price', lags=np.arange(1, 41))
    plot_pacf(eth_price_diff2, ax=axes[1], title='PACF of 2nd Order Differenced ETH-USD Price', lags=np.arange(1, 41))
    plt.tight_layout()
    plt.show()

    train_data, test_data = data['Adj Close'][:-30], data['Adj Close'][-30:]

    best_order = (4, 2, 1)
    best_mdl = ARIMA(train_data, order=best_order).fit()

    residuals = best_mdl.resid
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_acf(residuals, ax=ax[0])
    plot_pacf(residuals, ax=ax[1])
    plt.show()

    predictions = []
    current_data = train_data.copy()

    '''循环以达到one-step-ahead-prediction'''
    for i in range(len(test_data)):
        arima_mdl = ARIMA(current_data, order=best_order).fit()

        residuals = arima_mdl.resid
        garch_mdl = arch_model(residuals, vol='Garch', p=1, q=1).fit()

        arma_forecast = arima_mdl.forecast(steps=1)
        garch_forecast = garch_mdl.forecast(horizon=1).variance

        prediction = arma_forecast.values[0]
        predictions.append(prediction)

        # 增加新的真实数据到currentdata
        current_data = current_data._append(test_data.iloc[[i]])

    predicted_data = pd.Series(predictions, index=test_data.index, name="Predicted")


    plt.figure(figsize=(10, 5))
    # plt.plot(train_data, label='Training Set')
    plt.plot(test_data, label='Test Set')
    plt.plot(predicted_data, label='Predicted', linestyle='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
