# -*- coding : utf-8-*-
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import pandas as pd

# model_CNN_GRU = joblib.load('save/cnn_gru_current.model')
# model_RF = joblib.load('save/rf2_current.model')
# model_SVR = joblib.load('save/svr_current.model')

# rf_dir = r'save/predict_rf.csv'
# svr_dir = r'save/predict_svr.csv'
# cnn_gru_dir = r'save/predict_cnn_gru.csv'

# dict_dir = {'RF_Prediction': r'save/predict_rf.csv',
#             'SVR_Prediction': r'save/predict_svr.csv',
#             'LSTM_Prediction': r'save/predict_lstm.csv',
#             'CNN_GRU_Prediction': r'save/predict_cnn_gru64-64-64.csv'}
dict_dir = {'RF预测值': r'save/predict_rf.csv',
            'SVR预测值': r'save/predict_svr.csv',
            'GRU预测值': r'save/predict_gru_current20-64-64-64.csv',
            'LSTM预测值': r'save/predict_lstm.csv',
            'CNN-GRU预测值': r'save/predict_cnn_gru64-64-64.csv'}

real = None
date =None
plt.figure(0, figsize=(16, 8))
plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文标签

for dir in dict_dir:
    data = pd.read_csv(dict_dir[dir])
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S')
    # print(data['time'])
    # data.set_index("Time", drop=True, inplace=True)
    data = data.sort_values('time')
    real = data['y']
    date = data['time']
    plt.plot(data.index, data['predict'], label=dir)

# plt.plot(date.index, real, label='Real_data')
plt.plot(date.index, real, label='真实值')
plt.tick_params(labelsize=23)
plt.ylim((10, 80))
plt.xlabel('时间/min', size=23)
plt.ylabel('负荷/A', size=23)
plt.legend(loc='left',  prop={'size': 20}, frameon=False)
plt.savefig(r'D:\SJTU\机器学习\模型图\7-CNN_GRU\ALL_Data4.png',
            format='png',
            bbox_inches='tight',
            transparent=True,
            )  # bbox_inches='tight' 图片边界空白紧致, 背景透明


# 画出在300到750之间的图像
plt.figure(1)
plt.rcParams['axes.spines.top'] = True  # 显示顶部轴，必须放在plot之前
plt.rcParams['axes.spines.right'] = True  # 显示右部轴
for dir in dict_dir:
    data = pd.read_csv(dict_dir[dir])
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S')
    # print(data['time'])
    # data.set_index("Time", drop=True, inplace=True)
    data = data.sort_values('time')
    data = data[300:750]
    real = data['y'][300:750]
    date = data['time'][300:750]
    plt.plot(data.index, data['predict'], label=dir)

plt.plot(date.index, real, label='Real_data')
plt.tick_params(labelsize=20)
# plt.xlabel('Time/min', size=10)
# plt.ylabel('Value/A', size=10)
# plt.legend(loc='upper right', prop={'size': 15})
plt.savefig(r'D:\SJTU\机器学习\模型图\7-CNN_GRU\Some_Data4.png',
            format='png',
            bbox_inches='tight',
            transparent=True,
            )  # bbox_inches='tight' 图片边界空白紧致, 背景透明

# 画出在1700到2250之间的图像
plt.figure(2)
plt.rcParams['axes.spines.top'] = True  # 显示顶部轴，必须放在plot之前
plt.rcParams['axes.spines.right'] = True  # 显示右部轴
for dir in dict_dir:
    data = pd.read_csv(dict_dir[dir])
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S')
    # print(data['time'])
    # data.set_index("Time", drop=True, inplace=True)
    data = data.sort_values('time')
    data = data[1700:2250]
    real = data['y'][1700:2250]
    date = data['time'][1700:2250]
    plt.plot(data.index, data['predict'], label=dir)

plt.plot(date.index, real, label='Real_data')
plt.tick_params(labelsize=20)
# plt.xlabel('Time/min', size=10)
# plt.ylabel('Value/A', size=10)
# plt.legend(loc='upper right', prop={'size': 15})
plt.savefig(r'D:\SJTU\机器学习\模型图\7-CNN_GRU\Some_Data5.png',
            format='png',
            bbox_inches='tight',
            transparent=True,
            )  # bbox_inches='tight' 图片边界空白紧致, 背景透明

plt.show()
