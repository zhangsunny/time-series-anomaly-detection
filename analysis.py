# -*- coding: utf-8 -*-
"""
对lstm_ed.py模型重新表示的特征和误差进行分析
比较直接使用重构的特征或使用重构误差哪个较好
"""
import numpy as np
from zsutils import load_train_test, DEVICE, gen_batch, classify_analysis
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    path = './data/train-test-0.4.npz'
    train_data, test_data, train_label, test_label = load_train_test(path)
    vectors = np.load('lstm_ed.npz')
    error_vector = vectors['error_vector']
    pred_vector = vectors['pred_vector']

    model = KNeighborsClassifier()
    model.fit(error_vector, train_label)
    pred = model.predict(error_vector)

    model2 = KNeighborsClassifier()
    model2.fit(pred_vector, train_label)
    pred2 = model2.predict(pred_vector)

    classify_analysis(train_label, pred)
    classify_analysis(train_label, pred2)

    # plt.figure(figsize=(8, 3))
    # plt.subplot(2, 1, 1)
    # plt.plot(train_data[train_label == 0][0], color='black', label='ground truth')
    # plt.plot(pred_vector[train_label == 0][0], color='red', label='prediction')
    # plt.plot(error_vector[train_label == 0][0], color='blue', label='error')
    # plt.legend()
    # plt.title('Normal TS')
    # plt.tight_layout(True)
    # plt.subplot(2, 1, 2)
    # plt.plot(train_data[train_label == 1][0], color='black', label='ground truth')
    # plt.plot(pred_vector[train_label == 1][0], color='red', label='prediction')
    # plt.plot(error_vector[train_label == 1][0], color='blue', label='error')
    # plt.legend()
    # plt.title('Anomaly TS')
    # plt.tight_layout(True)
    # plt.show()




