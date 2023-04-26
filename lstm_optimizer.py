import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

class LSTMOptimizer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def create_model(self, units=50, optimizer='adam'):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(LSTM(units=units))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def optimize(self):
        model = KerasRegressor(build_fn=self.create_model, verbose=1)

        param_grid = {
            'units': [30, 50, 100],
            'optimizer': ['adam', 'rmsprop','sgd'],
            'batch_size': [1, 16, 32, 64],
            'epochs': [50, 100]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid_search.fit(self.X_train, self.y_train)

        return grid_result.best_params_
