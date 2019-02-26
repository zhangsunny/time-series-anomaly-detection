# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTM_TS(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, time_step):
        super(LSTM_TS, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=16)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        pred = self.fc2(self.tanh(self.fc1(h.squeeze(0))))
        return pred
        # h = self.init_variable(1, x.shape[0], self.hidden_size)
        # c = self.init_variable(1, x.shape[0], self.hidden_size)
        # preds = self.init_variable(x.shape[0], self.time_step)
        # for t in range(self.time_step):
        #     out, (h_new, c_new) = self.lstm(x[:, t, :].unsqueeze(1), (h, c))
        #     h, c = h_new, c_new
        #     pred = self.fc2(self.tanh(self.fc1(h.squeeze(0))))
        #     preds[:, t] = pred.flatten()
        # return preds

    @staticmethod
    def init_variable(*args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)
