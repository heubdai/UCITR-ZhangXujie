import torch
import torch.nn as nn
from .tlstm_node import TSLTM,MTSLTM

class Irregular_decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super(Irregular_decoder, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # print(self.input_dim,self.hidden_dim,output_dim)
        # The Stacked T-LSTM node takes input and maps it to hidden states
        cell_list = []
        h_linear_list = []
        for layer in range(num_layers):
            if layer == 0:
                cell_list.append(MTSLTM(input_dim, hidden_dim[0],device,True))
            else:
                cell_list.append(MTSLTM(hidden_dim[layer-1],hidden_dim[layer],device))
                # cell_list.append(TSLTM(hidden_dim[layer - 1], hidden_dim[layer],device))

        self.cell_list = nn.ModuleList(cell_list)
        self.h_linear_list = torch.nn.Linear(hidden_dim[0], input_dim)
        # The linear layer which maps from the hidden state to the predicted y
        self.W_y = torch.nn.Linear(hidden_dim[self.num_layers - 1], output_dim)
        # self.de_to_one = torch.nn.Linear(405,1)

    def forward(self, init_h, init_c, last_input, time_deltas):
        # print("log input shape ", inputs.shape)
        bs, seq_sz = time_deltas.shape
        h_t, c_t = init_h, init_c
        hidden_sequences_list = []
        for layer in range(self.num_layers):
            hidden_sequences_list.append(torch.empty([bs, seq_sz, self.hidden_dim[layer]])) # (N_layers, n_batches, sequence_length, N_features)

        for layer in range(self.num_layers):
            if layer == 0:
                hidden_seq = torch.empty([bs, seq_sz, self.hidden_dim[layer]])
                for t in range(seq_sz):
                    if t == 0:
                        imput_h = self.h_linear_list(last_input)
                        h_t, c_t = self.cell_list[layer].node_forward(h_t, c_t, imput_h, time_deltas[:, t:t + 1])
                    else:
                        imput_h = self.h_linear_list(h_t)
                        # imput_h = c_t
                        h_t, c_t = self.cell_list[layer].node_forward(h_t, c_t, imput_h, time_deltas[:, t:t + 1])
                    hidden_seq[:,t,:] = h_t.unsqueeze(0)
                hidden_sequences_list[layer] = hidden_seq
            else:
                hidden_sequences_list[layer], (h_t, c_t) = self.cell_list[layer].forward(hidden_sequences_list[layer-1].clone().to(self.device),time_deltas)

        # yhat = self.W_y(hidden_sequences_list[layer])  # inputs last hidden_sequence to dense output layers
        # yhat = torch.transpose(yhat,1,2)
        # yhat = self.de_to_one(yhat)
        return hidden_sequences_list[-1].to(self.device), (h_t, c_t)