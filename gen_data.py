# -*- coding: utf-8 -*-
"""
使用滑动窗口获得时间序列片段
"""
from lstm_ad import *

if __name__ == '__main__':
    path = './input/A1Benchmark/real_%s.csv'
    save_path = './data/segments-%s'
    seg_T = 60
    data = []
    label = []
    info = []
    for i in range(1, 68):
        print('reading file:', path % i)
        df = read_data(path % i)
        seg_ts, seg_label = gen_segments(df['scaled_value'].values, df['is_anomaly'].values, seg_T)
        data.extend(seg_ts)
        label.extend(seg_label)
        info.append([df.shape[0], sum(df['is_anomaly'])])
    data, label = np.array(data), np.array(label)
    info = np.array(info)
    print(data.shape, label.shape)
    print('Total points, anomaly points:', sum(info[:, 0]), sum(info[:, 1]))
    # np.savez(save_path % seg_T, data=data, label=label)