from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import math
import time
import pandas as pd
import numpy as np


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

# 以下为各种模型的切换选择
model_name = ''
csv_name = ''
plot_name = ''
model = None
choice = 2

if choice == 0:
    tscv = TimeSeriesSplit(n_splits=5)
    model = RidgeCV(cv=tscv)
    model_name = 'ridgecv_current.model'
    csv_name = 'predict_ridge.csv'
    plot_name = 'Ridge'
elif choice == 1:
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model_name = 'svr_current.model'
    csv_name = 'predict_svr.csv'
    plot_name = 'SVR'
elif choice == 2:
    model = RandomForestRegressor(n_estimators=5, max_depth=3)
    model_name = 'rf2_current.model'
    csv_name = 'predict_rf.csv'
    plot_name = 'RF'

start = time.time()
model.fit(x_train, y_train)
end = time.time()
print('time:' + str(end - start))
# joblib.dump(model, 'save/' + model_name)

train_predict = model.predict(x_train)  # 为训练的拟合值
# print(train_predict)
test_predict = model.predict(x_test)  # 为预测值

train_predict = np.array(train_predict)
train_predict = train_predict.reshape(train_predict.shape[-1], 1)

test_predict = np.array(test_predict)
test_predict = test_predict.reshape(test_predict.shape[-1], 1)

train_predict = scaler.inverse_transform(train_predict)
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
data_frame.to_csv('save/' + csv_name)

# 计算均方误差
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
plt.title(plot_name + ' for current value forecast')
# plt.title('CNN_GRU for useful-work forecast')
# plt.title('CNN_GRU for useless-work forecast')
plt.legend()
plt.grid()
plt.show()
