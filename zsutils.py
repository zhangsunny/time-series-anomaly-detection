# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score, accuracy_score, precision_score, recall_score
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 使用joblib保存模型
def save_pkl(path, model):
    joblib.dump(model, path)


# 使用joblib加载模型
def load_pkl(path):
    model = joblib.load(path)
    return model


# 产生batch data
def gen_batch(x, y, batch_size, device='cuda'):
    i = 0
    while i < x.shape[0]:
        batch_end = i + batch_size
        if batch_end >= x.shape[0]:
            batch_end = x.shape[0]
        var_x = x[i:batch_end].to(device)
        var_y = y[i:batch_end].to(device)
        i = batch_end
        yield var_x, var_y


# 对分类结果进行分析
def classify_analysis(y, pred, labels=[0, 1], beta=1):
    conf_mat = confusion_matrix(y, pred, labels=labels)
    print('='*10+'Confusion Matirx'+'='*10)
    print(conf_mat)
    tn, fp, fn, tp = conf_mat.ravel()
    print('=' * 10 + 'Classify Score' + '=' * 10)
    print('tn, fp, fn, tp:', tn, fp, fn, tp)
    print('Accuracy:', accuracy_score(y, pred))
    print('Precision:', precision_score(y, pred))
    print('Recall:', recall_score(y, pred))
    score = fbeta_score(y, pred, beta=beta, labels=labels, pos_label=labels[-1])
    print('F-measure score(beta=%s): %s' % (beta, score))


# 读取保存的训练和测试数据
def load_train_test(path):
    train_test = np.load(path)
    train_data = train_test['train_data']
    test_data = train_test['test_data']
    train_label = train_test['train_label']
    test_label = train_test['test_label']
    return train_data, test_data, train_label, test_label


# 保持类别比例划分正常和异常数据
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

