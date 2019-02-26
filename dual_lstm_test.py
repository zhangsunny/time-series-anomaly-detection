# -*- coding: utf-8 -*-
import numpy as np
from zsutils import load_train_test, DEVICE, gen_batch, classify_analysis, gen_multi_feature
import torch
from torch import nn
from dual_lstm import AttnDecoder, AttnEncoder


class DualLSTM(nn.Module):
    def __init__(self, input_size, n_filters, hidden_size, time_step):
        super(DualLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_step = time_step
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
        self.encoder = AttnEncoder(int(np.floor(hidden_size/2)*np.floor(input_size/4)),
                                   hidden_size, time_step)
        self.decoder = AttnDecoder(input_size, hidden_size, hidden_size, time_step)

    def forward(self, x):
        origin_x = x
        batch_size = x.shape[0]
        x = x.view(-1, self.input_size)
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = x.view(batch_size, self.time_step, -1)
        x_encoded = self.encoder(x)
        pred = self.decoder(x_encoded, origin_x)
        return pred


if __name__ == '__main__':
    path = './data/train-test-0.4.npz'
    train_data, test_data, train_label, test_label = load_train_test(path)

    T = 10
    train_data = gen_multi_feature(train_data, T)
    test_data = gen_multi_feature(test_data, T)

    TEST = False
    model_save_path = './models/dual_lstm.model'
    tmp_save_path = './models/dual_lstm_%s.model'
    batch_size = 50
    time_step = train_data.shape[1]
    input_size = train_data.shape[2]
    n_filters = 64
    hidden_size = 64
    EPOCH = 500
    lr = 0.01
    model = DualLSTM(input_size, n_filters, hidden_size, time_step).to(DEVICE)

    if not TEST:
        print('training model...')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()
        x = torch.from_numpy(train_data).float()
        y = torch.from_numpy(train_label).long()
        for epoch in range(EPOCH):
            batches = gen_batch(x, y, batch_size)
            loss_sum = 0
            counter = 0
            for var_x, var_y in batches:
                pred = model(var_x)
                loss = loss_func(pred, var_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * pred.shape[0]
                counter += 1
                print('finish epoch %s - batch %s' % (epoch + 1, counter))
            print('epoch %d loss: %f' % (epoch, loss_sum/x.shape[0]))
            if (epoch + 1) % 50 == 0:
                torch.save(model.state_dict(), tmp_save_path % (epoch + 1))
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
    # test_pred = torch.max(test_pred, dim=1)[1].cpu().data.numpy()
    classify_analysis(test_label, test_pred)
