import torch
import torch.nn as nn
from .tlstm_node import TSLTM,MTSLTM

class Irregular_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super(Irregular_encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # print(self.input_dim,self.hidden_dim,output_dim)
        # The Stacked T-LSTM node takes input and maps it to hidden states
        cell_list = []
        for layer in range(num_layers):
            if layer == 0:
                cell_list.append(MTSLTM(input_dim, hidden_dim[0],device,True))
            else:
                cell_list.append(MTSLTM(hidden_dim[layer-1],hidden_dim[layer],device))
                # cell_list.append(TSLTM(hidden_dim[layer - 1], hidden_dim[layer],device))

        self.cell_list = nn.ModuleList(cell_list)

        # The linear layer which maps from the hidden state to the predicted y
        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_dim[-1], hidden_dim[-1] * 2),
                                          torch.nn.Sigmoid(),
                                          torch.nn.Linear(hidden_dim[-1] * 2, hidden_dim[-1]),
                                          torch.nn.ReLU())

    def forward(self, inputs, time_deltas):
        # print("log input shape ", inputs.shape)
        bs, seq_sz, _ = inputs.shape
        hidden_sequences_list = []
        for layer in range(self.num_layers):
            hidden_sequences_list.append(torch.empty([bs, seq_sz, self.hidden_dim[layer]])) # (N_layers, n_batches, sequence_length, N_features)

        for layer in range(self.num_layers):
            if layer == 0:
                hidden_sequences_list[layer], (h_T, c_T) = self.cell_list[layer](inputs, time_deltas)
            else:
                hidden_sequences_list[layer], (h_T, c_T) = self.cell_list[layer](hidden_sequences_list[layer - 1].clone(),time_deltas)

        # yhat = self.W_y(hidden_sequences_list[layer])  # inputs last hidden_sequence to dense output layers
        # yhat = torch.transpose(yhat,1,2)
        # yhat = self.de_to_one(yhat)
        h_T,c_T = self.linear(h_T),self.linear(c_T)
        return hidden_sequences_list, (h_T, c_T)

class Interval_decoder(torch.nn.Module):
    def __init__(self,batch_size,input_dim, hidden_dim, device):
        super(Interval_decoder, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.cell = MTSLTM(input_dim,hidden_dim,True)

    def forward(self, H, C, time_deltas, input):
        hidden_seq,(h_t,c_t) = self.cell(inputs=input, time_deltas=time_deltas, init_states=(H,C))
        return hidden_seq, (h_t, c_t)

class Interval_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Interval_encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = MTSLTM(input_dim, hidden_dim,device,True)


    def forward(self, inputs, time_deltas):
        # print("log input shape ", inputs.shape)
        bs, seq_sz, _ = inputs.shape
        hidden_sequences_list, (h_T, c_T) = self.cell(inputs, time_deltas)
        return hidden_sequences_list, (h_T, c_T)


class Mask_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Mask_encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = TSLTM(input_dim, hidden_dim,device)


    def forward(self, inputs, time_deltas):
        # print("log input shape ", inputs.shape)
        bs, seq_sz, _ = inputs.shape
        hidden_sequences_list, (h_T, c_T) = self.cell(inputs, time_deltas)
        return hidden_sequences_list, (h_T, c_T)
