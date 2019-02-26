# -*- coding: utf-8 -*-
import numpy as np
from zsutils import load_train_test, DEVICE, gen_batch, classify_analysis
import torch
from torch import nn


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, n_filters, hidden_size, time_step):
        super(CNN_LSTM, self).__init__()
        self.input_size = input_size
        self.n_filters = n_filters
        self.conv1 = nn.Sequential(
            # batch_size * 1 * input_size
            nn.Conv1d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            # batch_size * out_channels * input_size
            nn.ReLU(),
            # batch_size * out_channels * input_size/2
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters,
                out_channels=n_filters//2,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.lstm = nn.LSTM(
            input_size=int(np.floor(hidden_size/2)*np.floor(input_size/4)),
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.input_size)
        # print(x.shape)
        x = self.conv1(x.unsqueeze(1))
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(batch_size, self.time_step, -1)
        # print(x.shape)
        out, (h, c) = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred


# 用滑动窗口处理获得多维特征
def slide_window(x, T):
    ts = []
    for i in range(x.shape[0] - T + 1):
        last = i + T
        ts.append(x[i:last])
    return np.array(ts)


def gen_multi_feature(x, T):
    ret = []
    for i in range(x.shape[0]):
        ts = slide_window(x[i], T)
        ret.append(ts)
    return np.array(ret)


if __name__ == '__main__':
    path = './data/train-test-0.4.npz'
    train_data, test_data, train_label, test_label = load_train_test(path)

    T = 10
    train_data = gen_multi_feature(train_data, T)
    test_data = gen_multi_feature(test_data, T)

    TEST = False
    model_save_path = './models/cnn_lstm.model'
    tmp_save_path = './models/cnn_lstm_%s.model'
    batch_size = 512
    time_step = train_data.shape[1]
    input_size = train_data.shape[2]
    n_filters = 64
    hidden_size = 64
    EPOCH = 100
    lr = 1e-2

    model = CNN_LSTM(input_size, n_filters, hidden_size, time_step).to(DEVICE)

    if not TEST:
        print('training model...')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()
        x = torch.from_numpy(train_data).float()
        y = torch.from_numpy(train_label).long()
        for epoch in range(EPOCH):
            batches = gen_batch(x, y, batch_size)
            loss_sum = 0
            for var_x, var_y in batches:
                pred = model(var_x)
                loss = loss_func(pred, var_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * pred.shape[0]
            print('epoch %d loss: %f' % (epoch, loss_sum/x.shape[0]))
            if (epoch + 1) % 50 == 0:
                torch.save(model.state_dict(), tmp_save_path % (epoch+1))
        torch.save(model.state_dict(), model_save_path)
    else:
        print('loading model...')
        model.load_state_dict(torch.load(model_save_path))

    var_test_x = torch.from_numpy(test_data).float().to(DEVICE)
    var_test_y = torch.from_numpy(test_label).long().to(DEVICE)
    # 测试序列似乎还是太大，只能分批测试
    test_pred = []
    batches = gen_batch(var_test_x, var_test_y, batch_size=batch_size)
    for var_x, var_y in batches:
        pred = model(var_x)
        test_pred.extend(list(pred.cpu().data.numpy()))
    test_pred = np.array(test_pred)
    test_pred = np.argmax(test_pred, axis=1)
    classify_analysis(test_label, test_pred)
