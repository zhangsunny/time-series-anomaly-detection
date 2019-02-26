# -*- coding: utf-8 -*-
"""
使用LSTM构建编码器和解码器
使用重构后的数据进行分类或者使用重构误差进行分类
尝试实现"LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"
"""
import numpy as np
from zsutils import load_train_test, DEVICE, gen_batch, classify_analysis
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable


# 使用LSTM进行编码和解码
class LSTM_ED(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, time_step):
        super(LSTM_ED, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        self.tanh = nn.Tanh()
        self.decoder = nn.LSTMCell(
            input_size=self.input_size + self.hidden_size,
            hidden_size=self.hidden_size,
        )

    def forward(self, x):
        out_e, (h_e, c_e) = self.encoder(x)
        hn_e = h_e[-1]
        pred = self.init_variable(x.shape[0], x.shape[1], x.shape[2])
        x_0 = self.tanh(self.fc(hn_e))
        pred[:, 0, :] = x_0.view(-1, self.input_size)
        h_d = torch.cat((hn_e, x_0), dim=1)
        for t in range(1, self.time_step):
            hd_t, cd_t = self.decoder(h_d)
            x_t = self.tanh(self.fc(hd_t))
            h_d = torch.cat((hd_t, x_t), dim=1)
            pred[:, t, :] = x_t.view(-1, self.input_size)
        return pred[:, range(pred.shape[1]-1, -1, -1)]
        # return pred

    @staticmethod
    def init_variable(*args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)
# class LSTM_ED(nn.Module):
#     def __init__(self, input_size, batch_size, hidden_size, time_step):
#         super(LSTM_ED, self).__init__()
#         self.input_size = input_size
#         self.batch_size = batch_size
#         self.hidden_size = hidden_size
#         self.time_step = time_step
#         self.encoder = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             batch_first=True,
#             num_layers=1,
#         )
#         self.decoder = nn.LSTM(
#             input_size=self.encoder.hidden_size,
#             hidden_size=self.encoder.hidden_size//2,
#             batch_first=True,
#             num_layers=1,
#         )
#         self.fc = nn.Linear(self.decoder.hidden_size, self.input_size)
#
#     def forward(self, x):
#         out_e, (h_e, c_e) = self.encoder(x)
#         out_d, (h_d, c_d) = self.decoder(out_e)
#         pred = self.fc(out_d)
#         return pred


if __name__ == '__main__':
    path = './data/train-test-0.4.npz'
    train_data, test_data, train_label, test_label = load_train_test(path)

    TEST = False
    model_save_path = './models/lstm_ed.model'
    batch_size = 512
    time_step = train_data.shape[1]
    hidden_size = 64
    input_size = 1
    EPOCH = 100
    lr = 1e-3
    model = LSTM_ED(input_size, batch_size, hidden_size, time_step).to(DEVICE)

    if not TEST:
        print('training model...')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # loss_func = nn.CrossEntropyLoss()
        loss_func = nn.MSELoss()
        # 只使用正常数据训练编码器
        x = torch.from_numpy(train_data[train_label == 0]).float().unsqueeze(2)
        y = torch.from_numpy(train_label[train_label == 0]).long()
        for epoch in range(EPOCH):
            batches = gen_batch(x, y, batch_size)
            loss_sum = 0
            for var_x, var_y in batches:
                pred = model(var_x)
                loss = loss_func(pred, var_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            print('epoch %d loss: %f' % (epoch, loss_sum))
        torch.save(model.state_dict(), model_save_path)
        # 使用训练好的模型表示正常和异常数据
        var_train_x = torch.from_numpy(train_data).float().unsqueeze(2).to(DEVICE)
        var_train_y = torch.from_numpy(train_label).long().to(DEVICE)
        train_pred = []
        batches = gen_batch(var_train_x, var_train_y, batch_size=batch_size)
        for var_x, var_y in batches:
            pred = model(var_x)
            train_pred.extend(list(pred.cpu().data.numpy()))
        pred_vector = np.array(train_pred)
        pred_vector = pred_vector.reshape(pred_vector.shape[0], pred_vector.shape[1])
        error_vector = train_data - pred_vector
        print(pred_vector.shape, error_vector.shape)
        np.savez('lstm_ed', pred_vector=pred_vector, error_vector=error_vector)

    else:
        print('loading model...')
        model.load_state_dict(torch.load(model_save_path))

        # 使用测试集测试
        var_test_x = torch.from_numpy(test_data).float().unsqueeze(2).to(DEVICE)
        var_test_y = torch.from_numpy(test_label).long().to(DEVICE)
        # 测试序列似乎还是太大，只能分批测试
        test_pred = []
        batches = gen_batch(var_test_x, var_test_y, batch_size=batch_size)
        for var_x, var_y in batches:
            pred = model(var_x)
            test_pred.extend(list(pred.cpu().data.numpy()))
        test_pred = np.array(test_pred)
        # test_pred = np.argmax(test_pred, axis=1)
        # test_pred = torch.max(test_pred, dim=1)[1].cpu().data.numpy()
        # classify_analysis(test_label, test_pred)
        plt.figure(figsize=(15, 5))
        plt.plot(test_pred[-1], label='prediction', color='red')
        plt.plot(test_data[-1], label='ground truth', color='black')
        plt.show()
