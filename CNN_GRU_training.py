from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import math
import time
import pandas as pd
import numpy as np
# import os
# os.environ["PATH"] += ";D:/software of office/Graphviz/bin/"

# creating dataframe
def create_dataframe(data_dir, y_index):
    data = pd.read_csv(data_dir)
    # data = data.drop(['W', 'Y'], axis=1, inplace=False)
    data['Time'] = pd.to_datetime(data['Time'])
    data = data.sort_values("Time")
    # data = df.sort_index(ascending=True, axis=0)

    new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'y'])
    new_data['Date'] = data['Time'].values
    new_data['y'] = data[y_index].values
    new_data.set_index('Date', drop=True, inplace=True)
    return new_data


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 利用列表来计算
def mean_absolute_error_list(y_true, y_pred):
    errors = []
    for i, value in enumerate(y_true):
        if value is None or value == 0.0:
            continue
        else:
            errors.append(np.abs(value - y_pred[i]) / value)
    return np.mean(errors)*100, errors*100


data_dir = 'file/2019.8.9—9.19.csv'
new_data = create_dataframe(data_dir, 'I')
# new_data = create_dataframe(data_dir, 'Y')
# new_data = create_dataframe(data_dir, 'W')

# creating train and test sets
dataset = new_data.values
print(len(dataset))
dataset = dataset[30000:]
# print(dataset.shape)
# data_max = max(dataset[30000:])

# converting dataset into x_train and y_train
# 特征值归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
train = scaled_data[:25000, :]
test = scaled_data[25000:, :]

look_back = 20
x_train, y_train = create_dataset(train, look_back=look_back)
x_test, y_test = create_dataset(test, look_back=look_back)
# print(x_train.shape)
# print(y_train.shape)

# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# y_train = np.reshape(y_train, (y_train.shape[0], 1 ))
print(x_train.shape)

# create and fit the LSTM network
# 投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]
# start = time.time()
# model = Sequential()
# model.add(Conv1D(32, 5, padding='same', activation='relu', input_shape=(look_back, 1)))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(32, 5, padding='same', activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(32, 2, padding='same', activation='relu'))
# # model.add(Flatten())
# model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2, activation='tanh',  return_sequences=True, name='GRU_1'))
# model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2, activation='tanh',  return_sequences=True, name='GRU_2'))
# model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2, activation='tanh',  return_sequences=False, name='GRU_3'))
# model.add(Dense(1, activation='relu'))
# model.compile(loss='mse', optimizer='RMSProp', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=2)
# end = time.time()
# print('time:' + str(end - start))
# joblib.dump(model, 'save/cnn_gru_current64-64-64.model')
# joblib.dump(model, 'save/cnn_gru_usefulwork.model')
# joblib.dump(model, 'save/cnn_gru_uselesswork.model')

model = joblib.load('save/cnn_gru_current64-64-64.model')
# model = joblib.load('save/cnn_gru_usefulwork.model')
# model = joblib.load('save/cnn_gru_uselesswork.model')
model.summary()
weight = model.get_layer('GRU_1').get_weights()
print(type(weight))


gru1_layer_model = Model(inputs=model.input, outputs=model.get_layer('GRU_1').output)
gru1_predict = gru1_layer_model.predict(x_test)
print(gru1_predict.shape)
plt.scatter(range(gru1_predict.shape[0]), [y, ])

gru2_layer_model = Model(inputs=model.input, outputs=model.get_layer('GRU_2').output)
gru2_predict = gru2_layer_model.predict(x_test)
print(gru2_predict.shape)

gru3_layer_model = Model(inputs=model.input, outputs=model.get_layer('GRU_3').output)
gru3_predict = gru3_layer_model.predict(x_test)
print(gru3_predict.shape)


# plot_model(model, 'save/model.png', show_shapes=True)
train_predict = model.predict(x_train)  # 为训练的拟合值
print(train_predict)
test_predict = model.predict(x_test)  # 为预测值
print([y_train])
train_predict = scaler.inverse_transform(train_predict)
print(train_predict)
train_y = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([y_test])

# 以下为将预测的数据和真实的数据传入到csv中
data_y = np.array(test_y)
data_y = data_y.reshape(data_y.shape[-1], )

data_predict = np.array(test_predict)
data_predict = data_predict.reshape(data_predict.shape[0], )
data_frame = pd.DataFrame({'time': new_data.index[55000+look_back:],
                           'predict': data_predict,
                           'y': data_y,
                           'error': (data_predict-data_y)/data_y})
data_frame.to_csv('save/predict_cnn_gru64-64-64.csv')
# data_frame.to_csv('save/predict_lstm.csv')


trainScore = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
print('Test Score: %.2f RMSE' % testScore)

trainScore, _ = mean_absolute_error_list(train_y[0], train_predict[:, 0])
print('Train Score: %.2f MAPE' % trainScore)
testScore, _ = mean_absolute_error_list(test_y[0], test_predict[:, 0])
print('Test Score: %.2f  MAPE' % testScore)

plt.plot(new_data.index[30000:], dataset, label='orig')
plt.plot(new_data.index[30000+look_back:55000], train_y.reshape(train_y.shape[1], 1), label='fit')
plt.plot(new_data.index[55000+look_back:], test_predict, label='predict')
plt.xlabel('Date')
plt.ylabel('Value')
# plt.title('CNN_GRU for current value forecast')
plt.title('CNN_GRU for useful-work forecast')
# plt.title('CNN_GRU for useless-work forecast')
plt.legend()
plt.grid()
plt.show()
