# -*- coding : utf-8-*-
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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


def cal_precision(error_dataframe):
    sum = 0
    count = 0
    error_list = list(error_dataframe.values)
    for error in error_list:
        if error < float('inf'):
            count += 1
            sum += abs(error)
    return (1 - sum / count) * 100


def plot_predice_pic(dict_dir):
    real = None
    date =None

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
    plt.savefig(r'picture/ALL_Data4.png',
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
    plt.savefig(r'picture/Some_Data4.png',
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
    plt.savefig(r'picture/Some_Data5.png',
                format='png',
                bbox_inches='tight',
                transparent=True,
                )  # bbox_inches='tight' 图片边界空白紧致, 背景透明

    plt.show()


def plot_error_pic(dict_dir):
    plt.figure(figsize=(16, 9))
    plt.tick_params(labelsize=23)
    precision_list = []
    for dir in dict_dir:
        data = pd.read_csv(dict_dir[dir])
        data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S')
        data = data.sort_values('time')
        date = data['time']
        precision_list.append(cal_precision(data['error']))
        plt.plot(data.index, [abs(error)*100 for error in list(data['error'].values)], label=dir)

    plt.ylim((0, 30))
    plt.xlabel('时间/min', size=23)
    plt.ylabel('绝对百分比误差', size=23)
    plt.legend(loc='upper left', prop={'size': 20}, frameon=False, ncol=3)
    plt.savefig(r'picture/error.png',
                format='png',
                bbox_inches='tight',
                transparent=True,
                )  # bbox_inches='tight' 图片边界空白紧致, 背景透明

    # 画出全时间的预测精度结果
    plt.figure(figsize=(16, 9))
    plt.tick_params(labelsize=23)
    plt.ylabel('平均预测精度', size=23)
    x = np.arange(len(dict_dir.keys()))
    plt.bar(x, height=precision_list, width=0.5, tick_label=list(dict_dir.keys()))
    for a, b in zip(x, precision_list):
        plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=20)
    plt.savefig(r'picture/precision.png',
                format='png',
                bbox_inches='tight',
                transparent=True,
                )  # bbox_inches='tight' 图片边界空白紧致, 背景透明


def plot_part_error_pic(dict_dir):
    # 画出在300到750之间的预测误差图像
    precision_list1 = []
    plt.figure(figsize=(16, 9))
    for dir in dict_dir:
        data = pd.read_csv(dict_dir[dir])
        data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S')
        data = data.sort_values('time')
        data = data[300:750]
        precision_list1.append(cal_precision(data['error']))
        plt.plot(data.index, [abs(error)*100 for error in list(data['error'].values)], label=dir)

    plt.tick_params(labelsize=20)
    plt.ylim((0, 30))
    plt.xlabel('时间/min', size=23)
    plt.ylabel('300~750时段绝对百分比误差', size=23)
    plt.legend(loc='upper left', prop={'size': 20}, frameon=False, ncol=3)
    plt.savefig(r'picture/some_data_error1.png',
                format='png',
                bbox_inches='tight',
                transparent=True,
                )  # bbox_inches='tight' 图片边界空白紧致, 背景透明

    # 画出300到750之间的预测精度结果
    plt.figure(figsize=(16, 9))
    plt.tick_params(labelsize=23)
    plt.ylabel('300~750时段平均预测精度', size=23)
    x = np.arange(len(dict_dir.keys()))
    plt.bar(x, height=precision_list1, width=0.5, tick_label=list(dict_dir.keys()))
    for a, b in zip(x, precision_list1):
        plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=20)
    plt.savefig(r'picture/some_data_precision1.png',
                format='png',
                bbox_inches='tight',
                transparent=True,
                )  # bbox_inches='tight' 图片边界空白紧致, 背景透明

    # 画出在1700到2250之间的预测误差图像
    precision_list2 = []
    plt.figure(figsize=(16, 9))
    plt.ylim((0, 30))
    for dir in dict_dir:
        data = pd.read_csv(dict_dir[dir])
        data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S')
        data = data.sort_values('time')
        data = data[1700:2250]
        precision_list2.append(cal_precision(data['error']))
        plt.plot(data.index, [abs(error)*100 for error in list(data['error'].values)], label=dir)
        # plt.plot(data.index, data['predict'], label=dir)

    plt.tick_params(labelsize=20)
    plt.ylim((0, 30))
    plt.xlabel('时间/min', size=23)
    plt.ylabel('1700~2250时段绝对百分比误差', size=23)
    plt.legend(loc='upper left', prop={'size': 20}, frameon=False, ncol=3)
    plt.savefig(r'picture/some_data_error2.png',
                format='png',
                bbox_inches='tight',
                transparent=True,
                )  # bbox_inches='tight' 图片边界空白紧致, 背景透明

    # 画出1700到2250之间的预测精度结果
    plt.figure(figsize=(16, 9))
    plt.tick_params(labelsize=23)
    plt.ylabel('1700~2250时段平均预测精度', size=23)
    x = np.arange(len(dict_dir.keys()))
    plt.bar(x, height=precision_list2, width=0.5, tick_label=list(dict_dir.keys()))
    for a, b in zip(x, precision_list2):
        plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=20)
    plt.savefig(r'picture/some_data_precision2.png',
                format='png',
                bbox_inches='tight',
                transparent=True,
                )  # bbox_inches='tight' 图片边界空白紧致, 背景透明


if __name__ == '__main__':
    dict_dir1 = {'RF预测值': r'save/predict_rf.csv',
                'SVR预测值': r'save/predict_svr.csv',
                'GRU预测值': r'save/predict_cnn_gru64-64-64.csv',
                'LSTM预测值': r'save/predict_lstm.csv',
                'CNN-GRU预测值': r'save/predict_gru_current20-64-64-64.csv'}
    dict_dir2 = {'RF预测值': r'save/predict_rf.csv',
                'SVR预测值': r'save/predict_svr.csv',
                'GRU预测值': r'save/predict_cnn_gru64-64-64.csv',
                'LSTM预测值': r'save/predict_gru_current20-64-64-64.csv',
                'CNN-GRU预测值': r'save/predict_lstm.csv'}
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
    plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
    plt.tick_params(labelsize=23)
    plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
    plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白
    # plot_predice_pic(dict_dir)
    # plot_error_pic(dict_dir1)
    plot_part_error_pic(dict_dir2)