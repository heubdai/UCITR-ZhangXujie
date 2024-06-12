# credit
# https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/nlp-lstm-byhand.ipynb
# https://github.com/duskybomb/tlstm
import numpy as np
import torch
import torch.nn as nn
import math


class TSLTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,device):
        super(TSLTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.W_d = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim).to(torch.float32))
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4).to(torch.float32))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4).to(torch.float32))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4).to(torch.float32))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():  # same init method as PyTorch LSTM implementation
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, time_deltas, init_states=None):
        """
        Implementation of T-LSTM node

        Args:
            input: array of dim (batch_size, seq_length, x_dim) where same seq_length in each batch
            time_deltas: tensor of time_deltas with shape (batch_size, Delta_t)
            init_states: If None then initialize as zeros, if not None ensure correct dimensions

        """
        batch_size, sequence_length, x_dim = inputs.shape

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_dim, requires_grad=False).to(torch.float32).to(self.device),
                        torch.zeros(batch_size, self.hidden_dim, requires_grad=False).to(torch.float32).to(self.device))
        else:
            h_t, c_t = init_states

        # For brevity
        exp_1 = torch.exp(torch.tensor(np.ones((batch_size, x_dim)),dtype=torch.float)).to(self.device)
        HS = self.hidden_dim

        hidden_seq = []
        for t in range(sequence_length):
            c_s = torch.tanh(c_t @ self.W_d)
            c_hat_s = c_s * (1 / torch.log(exp_1 + time_deltas[:, t,:]))  # expand as ensures the g(Delta_t) are replicated once for each dimension of c_st
            c_l = c_t - c_s
            c_adj = c_l + c_hat_s
            x_t = inputs[:, t, :]  # for all batches take the t'th period
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, cand_mem, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_adj + i_t * cand_mem
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)

class MTSLTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,device,isrelu=False):
        super(MTSLTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.isrelu = isrelu
        self.W_d = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim).to(torch.float32))
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4).to(torch.float32))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4).to(torch.float32))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4).to(torch.float32))

        self.init_weights()

    def extra_repr(self) -> str:
        return 'input_dim={}, hidden_dim={}'.format(
            self.input_dim, self.hidden_dim
        )


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():  # same init method as PyTorch LSTM implementation
            weight.data.uniform_(-stdv, stdv)

    def node_forward(self, h_t, c_t, input, time_deltas):

        HS = self.hidden_dim

        exp_1 = torch.exp(torch.tensor(1.0))
        c_s = torch.tanh(c_t @ self.W_d)
        c_hat_s = c_s * (1 / torch.log(exp_1 + time_deltas).expand_as(c_s))  # expand as ensures the g(Delta_t) are replicated once for each dimension of c_st
        c_l = c_t - c_s
        c_adj = c_l + c_hat_s
        x_t = input  # for all batches take the t'th period
        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, cand_mem, o_t = (
            torch.sigmoid(gates[:, :HS]),  # input
            torch.sigmoid(gates[:, HS:HS * 2]),  # forget
            torch.tanh(gates[:, HS * 2:HS * 3]),
            torch.sigmoid(gates[:, HS * 3:]),  # output
        )
        c_t = f_t * c_adj + i_t * cand_mem
        h_t = o_t * torch.tanh(c_t)
        if self.isrelu:
            return torch.relu(h_t),torch.relu(c_t)
        else:
            return torch.tanh(h_t),torch.tanh(c_t)

    def forward(self, inputs, time_deltas, init_states=None):
        """
        Implementation of T-LSTM node

        Args:
            input: array of dim (batch_size, seq_length, x_dim) where same seq_length in each batch
            time_deltas: tensor of time_deltas with shape (batch_size, Delta_t)
            init_states: If None then initialize as zeros, if not None ensure correct dimensions

        """
        batch_size, sequence_length, x_dim = inputs.shape

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_dim, requires_grad=False).to(torch.float32).to(self.device),
                        torch.zeros(batch_size, self.hidden_dim, requires_grad=False).to(torch.float32).to(self.device))
        else:
            h_t, c_t = init_states

        # For brevity
        exp_1 = torch.exp(torch.tensor(1.0))
        HS = self.hidden_dim

        hidden_seq = []
        for t in range(sequence_length):
            h_t,c_t = self.node_forward(h_t,c_t,inputs[:,t,:],time_deltas[:,t:t+1])
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)