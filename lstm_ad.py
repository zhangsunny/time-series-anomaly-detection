# -*- coding: utf-8 -*-
"""
测试一维时间点回归
检测目标是判断时间序列片段是否异常
"""
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
from model import LSTM_TS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 读入csv文件并适当处理
def read_data(path):
    df = pd.read_csv(path, index_col=['timestamp'])
    scaler = MinMaxScaler()
    df['scaled_value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    return df


# 将一个含有多个异常点的时间序列划分为多个较短的时间序列片段
def gen_segments(x, y, T):
    ts = []
    label = []
    for i in range(x.shape[0] - T + 1):
        last = i + T
        ts.append(x[i:last])
        tmp = y[i:last]
        if 1 in tmp:
            label.append(1)
        else:
            label.append(0)
    return np.array(ts), np.array(label)


def split_train_test(x, y, ratio=0.2):
    x_normal, y_normal = x[y == 0], y[y == 0]
    x_anomaly, y_anomaly = x[y == 1], y[y == 1]
    nx_train, nx_test, ny_train, ny_test = train_test_split(x_normal, y_normal, test_size=ratio)
    ax_train, ax_test, ay_train, ay_test = train_test_split(x_anomaly, y_anomaly, test_size=ratio)
    x_train = np.concatenate([nx_train, ax_train])
    x_test = np.concatenate([nx_test, ax_test])
    y_train = np.concatenate([ny_train, ay_train])
    y_test = np.concatenate([ny_test, ay_test])
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, x_test, y_train, y_test


# 滑动窗口处理生成输入和预测目标
def slide_window(x, T):
    in_x, out_x = [], []
    for i in range(x.shape[0]):
        tmp = x[i]
        tmp_in, tmp_out = [], []
        for t in range(tmp.shape[0] - T):
            last = t + T
            tmp_in.append(tmp[t:last])
            tmp_out.append(tmp[last])
        in_x.append(tmp_in)
        out_x.append(tmp_out)
    return np.array(in_x), np.array(out_x)


if __name__ == '__main__':

    seg_T = 100
    split_ratio = 0.4
    win_T = 10
    path = './input/A1Benchmark/real_1.csv'
    save_path = './models/lstm_model.model'
    df = read_data(path)
    seg_ts, seg_label = gen_segments(df['scaled_value'].values, df['is_anomaly'].values, seg_T)
    train_ts, test_ts, train_label, test_label = split_train_test(seg_ts, seg_label, split_ratio)
    # print(train_ts.shape, test_ts.shape, train_label.shape, test_label.shape)
    train_data, train_target = slide_window(train_ts, win_T)
    test_data, test_target = slide_window(test_ts, win_T)
    # print(train_data.shape, train_target.shape, test_data.shape, test_target.shape)

    # 使用input_size=1, time_step=win_T进行预测
    X = train_data.reshape(-1, train_data.shape[-1])
    y = train_target.reshape(-1, 1)
    # print(X.shape, y.shape)
    X = torch.from_numpy(X).float().unsqueeze(2)
    y = torch.from_numpy(y).float()

    input_size = 1
    time_step = win_T
    batch_size = 1000
    hidden_size = 64
    lr = 0.01
    EPOCH = 100
    TEST = True
    lstm_model = LSTM_TS(input_size, batch_size, hidden_size, time_step).to(DEVICE)

    if not TEST:
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
        loss_func = torch.nn.MSELoss()
        tic = time.time()
        for epoch in range(EPOCH):
            i = 0
            loss_sum = 0
            while i < X.shape[0]:
                batch_end = i + batch_size
                if batch_end >= X.shape[0]:
                    batch_end = X.shape[0]
                var_x = X[i:batch_end].to(DEVICE)
                var_y = y[i:batch_end].to(DEVICE)
                pred = lstm_model(var_x)
                loss = loss_func(pred, var_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i = batch_end
                loss_sum += loss.item()
            print('epoch [%d] finished, loss is %f' % (epoch, loss_sum))
        toc = time.time()
        print('training %d epochs cost %4.f s' % (EPOCH, (toc - tic)))
        torch.save(lstm_model.state_dict(), save_path)
    else:
        lstm_model.load_state_dict(torch.load(save_path))
        error_vector = []
        pred = []
        for i in range(test_data.shape[0]):
            var_x = torch.from_numpy(test_data[i]).float().unsqueeze(2).to(DEVICE)
            tmp_pred = lstm_model(var_x)
            pred.append(tmp_pred.cpu().data.numpy().flatten())
        pred = np.array(pred)
        error_vector = pred - test_target
        print(error_vector.shape)
        np.savez('vectors',
                 error_vector=error_vector,
                 pred_vector=pred,
                 target_vector=test_target,
                 label=test_label,
        )
