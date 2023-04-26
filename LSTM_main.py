# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

from lstm_optimizer import LSTMOptimizer


def main():

    data = pd.read_csv("/Users/lizan/Desktop/ETH-USD (1).csv")

    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    train_data, test_data = data['Adj Close'][:-30], data['Adj Close'][-30:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))


    lookback = 5 #3，5，7
    X_train, y_train = [], []# output sequences for LSTM model

    for i in range(lookback, len(scaled_train_data)):
        X_train.append(scaled_train_data[i-lookback:i, 0])
        y_train.append(scaled_train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #Lstm
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1) #batch_size:16,32,64 GPU 对 2 的幂次的 batch 可以发挥更好性能，因此设置成 16、32、64、128

    inputs = data['Adj Close'].values[-len(test_data)-lookback:].reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(lookback, len(inputs)):
        X_test.append(inputs[i-lookback:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_data = model.predict(X_test)
    predicted_data = scaler.inverse_transform(predicted_data)

    # Error
    mse = mean_squared_error(test_data.values, predicted_data)
    print(f"Mean Squared Error: {mse}")


    plt.figure(figsize=(10, 5))
    plt.plot(data.index[-30:], test_data.values, label="Actual")
    plt.plot(data.index[-30:], predicted_data, label="Predicted", linestyle="--")
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


    '''GridSearch'''
    # # Optimize
    # lstm_optimizer = LSTMOptimizer(X_train, y_train)
    # best_params = lstm_optimizer.optimize()
    # print(best_params)
    #
    # # new optimal hyperparameters
    # model = lstm_optimizer.create_model(units=best_params['units'], optimizer=best_params['optimizer'])
    # model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=2)
    #
    # predicted_data = model.predict(X_test)
    # predicted_data = scaler.inverse_transform(predicted_data)
    #
    # # Error
    # mse = mean_squared_error(test_data.values, predicted_data)
    # print(f"Mean Squared Error: {mse}")
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(data.index[-30:], test_data.values, label="Actual")
    # plt.plot(data.index[-30:], predicted_data, label="Predicted", linestyle="--")
    # plt.legend()
    # plt.show()
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

