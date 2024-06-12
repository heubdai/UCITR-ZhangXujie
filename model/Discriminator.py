import torch
import torch.nn as nn
from .tlstm_node import TSLTM,MTSLTM

class Mask_Discriminator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, device):
        super(Mask_Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # print(self.input_dim,self.hidden_dim,output_dim)
        # The Stacked T-LSTM node takes input and maps it to hidden states
        self.LSTM_node = MTSLTM(input_dim, hidden_dim, device)
        # The linear layer which maps from the hidden state to the predicted y
        self.Linear = torch.nn.Linear(hidden_dim, output_dim)
        # self.de_to_one = torch.nn.Linear(405,1)

    def forward(self, inputs, time_deltas):
        # print("log input shape ", inputs.shape)
        bs, seq_sz, _ = inputs.shape
        hidden_state, (h_T,c_T) = self.LSTM_node(inputs, time_deltas)

        dis_out = self.Linear(hidden_state)
        dis_out = torch.nn.Sigmoid(dis_out)

        return dis_out

class Predictor(torch.nn.Module):
    def __init__(self, code_dim,hidden_dim,num_layer):
        super(Predictor, self).__init__()
        self.num_layer = num_layer+1
        cell_list = []
        for layer in range(num_layer):
            if layer == 0:
                cell_list.append(torch.nn.Linear(code_dim, hidden_dim[0]))
            else:
                cell_list.append(torch.nn.Linear(hidden_dim[layer - 1], hidden_dim[layer]))
        cell_list.append(torch.nn.Linear(hidden_dim[-1],code_dim))
                # cell_list.append(TSLTM(hidden_dim[layer - 1], hidden_dim[layer],device))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input):
        for layer in range(self.num_layer):
            if layer == 0:
                output = self.cell_list[0](input)
            else:
                output = self.cell_list[layer](output)

        return output