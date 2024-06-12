import random
import argparse
from model import UCITR
from model import UCITR_Linear
import torch
import numpy as np
from torch import optim
import time
from model import TLSTMModel
import torch.nn.functional as F
from utils import load_UEA
import time
import matplotlib.pyplot as plt
from tasks.classification import eval_classification

def data_dim_norm(origin_data):
    for i in range(origin_data.shape[-1]):
        dmin = origin_data[:,:,i].min()
        dmax = origin_data[:,:,i].max()
        origin_data[:,:,i] = (origin_data[:,:,i]-dmin) / (dmax-dmin)
    return origin_data

def train(args):
    PATH = args.dataset + str(args.epoch) + ".pt"
    print("save name as {}".format(PATH))
    train_X, train_y, test_X, test_y = load_UEA(args.dataset)
    # train_X = train_X[:,:,0:5]
    train_X = data_dim_norm(train_X)
    test_X = data_dim_norm(test_X)
    device = args.device
    epoch = args.epoch
    # epoch = 1
    print("total epoch ", epoch)
    batchsize = args.batch_size
    time_delat = np.ones(train_X.shape[0:2])
    test_time_delat = np.ones(test_X.shape[0:2])
    print("time_delat shape ", time_delat.shape)
    print("train x shape ", train_X.shape)
    print("test x shape ", test_X.shape)
    # model = UCITR(input_dim=train_X.shape[2], hidden_dim=[128], num_layers=1, output_dim=128, device=device,predict_rate=0.2).to(device)
    model = UCITR_Linear(input_dim=train_X.shape[2], hidden_dim=[128], num_layers=1, output_dim=128, device=device,
                         predict_rate=0.2, args=args).to(device)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    print(model)
    if args.weights != 'none':
        print("load weights in {}".format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    my_loss = torch.nn.MSELoss(reduction="sum")
    batch_num = int(train_X.shape[0] / batchsize)
    for i in range(epoch):
        starttime = time.perf_counter()
        idx = np.arange(train_X.shape[0])
        random.shuffle(idx)
        train_X = train_X[idx]
        train_y = train_y[idx]
        reconstruct_loss = torch.tensor(0.0)
        predict_loss = torch.tensor(0.0)
        for j in range(batch_num):
            data = torch.tensor(train_X[j * batchsize:j * batchsize + batchsize]).to(torch.float32).to(device)
            delat = torch.tensor(time_delat[j * batchsize:j * batchsize + batchsize]).to(torch.float32).to(device)
            # h, c, reconstrction,(prec_h,observed_h) = model(data, delat, time_delat[j*batchsize:j*batchsize+batchsize])
            hidden, reconstrction, pred, hiddenback = model(data, delat, delat)
            # loss1 = my_loss(torch.transpose(data,1,2),torch.transpose(reconstrction,1,2))
            loss1 = my_loss(data, reconstrction)
            reconstruct_loss += loss1.item()
            loss2 = my_loss(pred, hiddenback)
            predict_loss += loss2.item()
            optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            # if i > 50:
            loss2.backward()
            optimizer.step()
            # optimizer.zero_grad()
            # optimizer.step()
        endtime = time.perf_counter()
        # for j in range(3):
        #     print(reconstrction[j])
        #     print(data[j])
        #     print("第",j,"组重构与真实结果对比")
        # if i % 10 == 0:
        # print("end of epoch {} last loss1 {} loss2 {} learning time {}".format(i, loss1, loss2, endtime-starttime))
        print("end of epoch {} 重构损失 {} 预测损失 {} 训练时间 {}".format(i, reconstruct_loss, predict_loss, endtime - starttime))

    if epoch != 0:
        torch.save(model.state_dict(), PATH)