# -*- coding: utf-8 -*-
import numpy as np
from zsutils import load_train_test, DEVICE, gen_batch, classify_analysis
import torch
from torch import nn


# 使用LSTM进行分类
class LSTM_BASE(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, time_step):
        super(LSTM_BASE, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=16)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        pred = self.fc2(self.tanh(self.fc1(out[:, -1, :])))
        return pred


if __name__ == '__main__':
    path = './data/train-test-0.4.npz'
    train_data, test_data, train_label, test_label = load_train_test(path)

    TEST = True
    model_save_path = './models/lstm_base.model'
    batch_size = 512
    time_step = train_data.shape[1]
    hidden_size = 64
    input_size = 1
    EPOCH = 500
    lr = 0.01
    model = LSTM_BASE(input_size, batch_size, hidden_size, time_step).to(DEVICE)

    if not TEST:
        print('training model...')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()
        x = torch.from_numpy(train_data).float().unsqueeze(2)
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
                loss_sum += loss.item()
            print('epoch %d loss: %f' % (epoch, loss_sum))
        torch.save(model.state_dict(), model_save_path)
    else:
        print('loading model...')
        model.load_state_dict(torch.load(model_save_path))

    var_test_x = torch.from_numpy(test_data).float().unsqueeze(2).to(DEVICE)
    var_test_y = torch.from_numpy(test_label).long().to(DEVICE)
    # 测试序列似乎还是太大，只能分批测试
    test_pred = []
    batches = gen_batch(var_test_x, var_test_y, batch_size=batch_size)
    for var_x, var_y in batches:
        pred = model(var_x)
        test_pred.extend(list(pred.cpu().data.numpy()))
    test_pred = np.array(test_pred)
    test_pred = np.argmax(test_pred, axis=1)
    # test_pred = torch.max(test_pred, dim=1)[1].cpu().data.numpy()
    classify_analysis(test_label, test_pred)
