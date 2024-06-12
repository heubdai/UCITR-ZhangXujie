from .Discriminator import Predictor,Mask_Discriminator
from .Irregular_decoder import Irregular_decoder
from .Irregular_encoder import Irregular_encoder, Interval_encoder, Mask_encoder, Interval_decoder
from .tlstm_model import TLSTMModel
import torch.nn as nn
import torch
import torch.nn.modules.linear
# 此模型的total_hidden是由去掉预测时刻的剩余数据得到的
class UCITR(torch.nn.Module):
    def __init__(self, length, hidden_dim, input_dim, device,args):
        super(UCITR,self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.args = args
        self.predict_timestep = args.predict_timestep
        self.batchsize = args.batch_size
        self.length = length
        self.Wk = nn.ModuleList(
            [nn.Linear(hidden_dim, 256) for i in range(self.predict_timestep)])
        self.encoder_interval = Interval_encoder(input_dim, input_dim, device)
        self.encoder_mask = Mask_encoder(input_dim, input_dim, device)
        self.decoder = Interval_decoder(batch_size=self.batchsize, hidden_dim=hidden_dim, input_dim=input_dim, device=device)
        self.linear_encoder = nn.Linear(2*input_dim,256)
        self.embedding_linear = nn.Linear(804,hidden_dim)
        self.conv_encode1 = torch.nn.ConvTranspose1d(self.length-self.predict_timestep , 1, kernel_size=5)
        self.conv_encode2 = torch.nn.ConvTranspose1d(self.length-self.predict_timestep , 1, kernel_size=11)
        self.conv_encode3 = torch.nn.ConvTranspose1d(self.length-self.predict_timestep ,1,kernel_size=23)
        self.linear_c = torch.nn.Linear(2*input_dim,hidden_dim)
        self.linear_h = torch.nn.Linear(2*input_dim,hidden_dim)
        linear_decode_list = []
        for i in range(self.length):
            linear_decode_list.append(nn.Linear(hidden_dim,input_dim*2))
        self.linear_decode_list = nn.ModuleList(linear_decode_list)
        self.linear_decoder = nn.Linear(hidden_dim,input_dim)

    def forward(self, inputs, time_delat, missing_mask):
        hidden, total_hidden, (h_t,c_t) = self.inner_encode(inputs, time_delat, missing_mask)
        output = self.decode(h_t, c_t, inputs=inputs, time_delat=time_delat)
        total_hidden = torch.squeeze(total_hidden,1)
        # hidden_front = hidden[:,:-self.args.predict_timestep,:]
        hidden_back = hidden[:,-self.predict_timestep:,:]
        pred = torch.empty((self.batchsize, self.predict_timestep, 256)).float().to(self.device)
        for i in range(self.predict_timestep):
            linear = self.Wk[i]
            pred[:,i] = linear(total_hidden)
        return total_hidden,output,pred,hidden_back

    def inner_encode(self, inputs, time_delat, missing_mask):
        hidden, (ih_T, ic_T) = self.encoder_interval(inputs, time_delat)
        hidden_msak ,(mh_T,mc_T) = self.encoder_mask(inputs, missing_mask)
        h_T = torch.cat((ih_T,mh_T),dim=1)
        c_T = torch.cat((ic_T,mc_T),dim=1)
        linear_h = torch.tanh(self.linear_h(h_T))
        linear_c = torch.tanh(self.linear_c(c_T))
        hidden = torch.cat((hidden, hidden_msak), dim=2)
        hidden = self.linear_encoder(hidden)
        hidden_front = hidden[:,:-self.predict_timestep,:]
        hidden1 = self.conv_encode1(hidden_front)
        hidden2 = self.conv_encode2(hidden_front)
        hidden3 = self.conv_encode3(hidden_front)
        cat_hidden = torch.cat((hidden1,hidden2,hidden3),dim=2)
        total_hidden = self.embedding_linear(cat_hidden)
        return hidden, total_hidden, (linear_h, linear_c)

    def encode(self, inputs, time_delat, missing_mask):
        hidden,total_hidden, (h_T, c_T) = self.inner_encode(inputs, time_delat, missing_mask)
        return hidden, torch.squeeze(total_hidden,1)

    def decode(self, h_t, c_t, inputs, time_delat):

        input = torch.flip(inputs[:,1:],dims=[1])
        zero = torch.zeros(self.batchsize,1,input.shape[2]).to(torch.float32).to(self.device)
        input = torch.cat((zero,input),dim=1)
        output,(h_t,c_t) =self.decoder(time_deltas=time_delat, input=input,H=h_t,C=c_t)
        output = torch.flip(output,dims=[1])

        output = self.linear_decoder(output)
        return output

