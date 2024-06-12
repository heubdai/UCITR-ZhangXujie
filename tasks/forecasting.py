import numpy as np
import time
from . import _eval_protocols as eval_protocols
import torch
import torch.optim as optim

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }

class pre_model(torch.nn.Module):
    def __init__(self,predict_step,hidden,inputdim,device):
        super(pre_model, self).__init__()
        self.predict_step = predict_step
        self.hidden = hidden
        self.inputdim = inputdim
        self.device = device
        self.Wk = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, inputdim) for i in range(self.predict_step)])

    def forward(self,batchsize, input_rep):
        pre_res = torch.empty((batchsize, self.predict_step, self.inputdim)).float().to(self.device)

        for i in range(self.predict_step):
            linear = self.Wk[i]
            pre_res[:,i] = linear(input_rep)
        return pre_res


def eval_forecasting(model, train_data, test_data, train_delat, test_delat, train_mask, test_mask, args,train_real,test_real):

    device = args.device
    hidden = args.repr_dims
    predict_step = train_real.shape[1]
    inputdim = train_data.shape[2]
    batchsize = train_data.shape[0]
    _, train_rept = model.encode(torch.tensor(train_data).to(torch.float32).to(device),
                                 torch.tensor(train_delat).to(torch.float32).to(device),
                                 torch.tensor(train_mask).to(torch.float32).to(device))
    _, test_rept = model.encode(torch.tensor(test_data).to(torch.float32).to(device),
                                         torch.tensor(test_delat).to(torch.float32).to(device),
                                         torch.tensor(test_mask).to(torch.float32).to(device))
    train_rept = train_rept.detach()
    test_rept = test_rept.detach()
    my_loss = torch.nn.MSELoss()
    premodel = pre_model(predict_step,hidden,inputdim,device).to(device)
    print(premodel)
    optimizer = optim.Adam(premodel.parameters(), lr=0.003)
    for i in range(200):
        # 30个epoch
        real = torch.tensor(train_real).to(torch.float32).to(device)
        pre_res = premodel(batchsize,train_rept)
        loss = my_loss(pre_res, real)
        print("loss ",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pre_res = premodel(test_data.shape[0],test_rept)
    # pre_res = torch.mul(pre_res,torch.tensor(dim_record).to(device))
    test_real = torch.tensor(test_real).to(torch.float32).to(device)
    # test_real = torch.mul(test_real,torch.tensor(dim_record).to(device))
    loss = my_loss(pre_res,test_real)
    print(loss)
    return loss.item()
# 114.1018, 2474.9781,514.1903