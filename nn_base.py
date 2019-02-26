# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from zsutils import load_train_test, classify_analysis, gen_batch, split_train_test, DEVICE
import os


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(60, 16)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.tanh(self.fc1(x)))


if __name__ == '__main__':
    path = './data/segments-60.npz'
    segments = np.load(path)
    data = segments['data']
    label = segments['label']
    print('shape of data, label:', data.shape, label.shape)
    anomaly_size = sum(label)
    anomaly_ratio = anomaly_size / data.shape[0]
    print('anomaly size, ratio:', anomaly_size, anomaly_ratio)

    split_ratio = 0.4
    save_path = './data/train-test-%s' % split_ratio
    if not os.path.exists(save_path + '.npz'):
        print('split train test...')
        train_data, test_data, train_label, test_label = split_train_test(data, label, split_ratio)
        np.savez(save_path, train_data=train_data, test_data=test_data,
                 train_label=train_label, test_label=test_label)
    else:
        print('read train test...')
        train_data, test_data, train_label, test_label = load_train_test(save_path + '.npz')
    print(train_data.shape, test_data.shape, sum(train_label), sum(test_label))

    batch_size = 512
    EPOCH = 500
    lr = 0.001
    model_save_path = './models/nn.model'
    TEST = True
    x = torch.from_numpy(train_data).float()
    y = torch.from_numpy(train_label).long()
    model = NN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    if not TEST:
        for epoch in range(EPOCH):
            batches = gen_batch(x, y, batch_size)
            loss_sum = 0
            for var_x, var_y in batches:
                pred = model(var_x)
                loss = loss_func(pred, var_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            print('epoch %d loss: %f' % (epoch, loss_sum))
        torch.save(model.state_dict(), model_save_path)
    else:
        model.load_state_dict(torch.load(model_save_path))

    var_test_x = torch.from_numpy(test_data).float().to(DEVICE)
    var_test_y = torch.from_numpy(test_label).long().to(DEVICE)
    test_pred = model(var_test_x)
    test_pred = torch.max(test_pred, dim=1)[1].cpu().data.numpy()
    classify_analysis(test_label, test_pred)

