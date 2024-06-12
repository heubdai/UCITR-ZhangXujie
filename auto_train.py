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
from utils import load_UEA,load_UCR,load_electricity,load_forecast_csv
import time
import matplotlib.pyplot as plt
from tasks.classification import eval_classification
from tasks.forecasting import eval_forecasting

def plot_hidden(hidden,batch_num):
    hidden = np.array(hidden.detach().cpu())
    for i in range(5):
        plt.plot(np.arange(hidden[i].shape[0]),hidden[i])
    plt.savefig(str(batch_num) + "_hidden.jpg")
    plt.show()

def data_dim_norm(origin_data):
    for i in range(origin_data.shape[-1]):
        dmin = origin_data[:,:,i].min()
        dmax = origin_data[:,:,i].max()
        origin_data[:,:,i] = (origin_data[:,:,i]-dmin) / (dmax-dmin)
    return origin_data

def mask_aug(dataset,ratio):
    mask = np.ones(dataset.shape)
    aug_data = dataset
    randmask = np.random.rand(dataset.shape[0],dataset.shape[1],dataset.shape[2])
    mask[np.where(randmask<ratio)] = 0
    aug_data[np.where(randmask<ratio)] = random.random()
    return aug_data, mask

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--repr_dims', type=int, default=800)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--predict_timestep', type=int, default=6)
    parser.add_argument('--dataset', type=str, default='NATOPS')
    parser.add_argument('--weights', type=str, default='none')
    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--test_missing_rate', type=float, default=0.0)
    parser.add_argument('--train_missing_rate', type=float, default=0.2)
    parser.add_argument('--test_plot', action="store_true")
    parser.add_argument('--test_eval', action="store_true")
    parser.add_argument('--loader', type=str, required=True,
                        help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')

    args = parser.parse_args()
    return args

def train(args):
    if args.loader == 'UCR':
        task_type = 'classification'
        train_X, train_y, test_X, test_y = load_UCR(args.dataset)
        train_X = data_dim_norm(train_X)
        test_X = data_dim_norm(test_X)

    elif args.loader == 'UEA':
        task_type = 'classification'
        train_X, train_y, test_X, test_y = load_UEA(args.dataset)
        train_X = data_dim_norm(train_X)
        test_X = data_dim_norm(test_X)

    elif args.loader == 'electricity':
        task_type = 'forecasting'
        # elec_data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(args.dataset)
        # train_X = elec_data[:, train_slice]
        # args.dataset = "LD2011_2014.txt"
        data = load_electricity(args.dataset, 24 * 7)
        seg = int(data.shape[0] * 0.8)
        train_X = data[0:seg]
        test_X = data[seg:]
        # train_y,test_y = 0,0

    device = args.device
    epoch = args.epoch
    print("total epoch ", epoch)
    batchsize = args.batch_size
    time_delat = np.ones(train_X.shape[0:2])
    test_time_delat = np.ones(test_X.shape[0:2])
    # print("time_delat shape ", time_delat.shape)
    # print("train x shape ", train_X.shape)
    # model = UCITR_Linear(input_dim=train_X.shape[2], hidden_dim=800, length=train_X.shape[1], device=device, args=args).to(device)
    model = UCITR(input_dim=train_X.shape[2], hidden_dim=args.repr_dims, length=train_X.shape[1], device=device,
                  args=args).to(device)
    # print(model)
    if args.weights != 'none':
        print("load weights in {}".format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    my_loss = torch.nn.MSELoss()
    batch_num = int(train_X.shape[0] / batchsize)
    alpha = 1
    for i in range(epoch):
        starttime = time.perf_counter()
        idx = np.arange(train_X.shape[0])
        # random.shuffle(idx)
        t_train_X = train_X[idx]
        # t_train_y = train_y[idx]
        reconstruct_loss = torch.tensor(0.0)
        predict_loss = torch.tensor(0.0)
        # if i > epoch*0.7:
        #     alpha=1
        for j in range(batch_num):
            data = torch.tensor(t_train_X[j * batchsize:j * batchsize + batchsize]).to(torch.float32).to(device)
            aug_data, mask = mask_aug(t_train_X[j * batchsize:j * batchsize + batchsize], args.train_missing_rate)
            aug_data, mask = torch.tensor(aug_data).to(torch.float32).to(device), torch.tensor(mask).to(
                torch.float32).to(device)
            delat = torch.tensor(time_delat[j * batchsize:j * batchsize + batchsize]).to(torch.float32).to(device)
            # h, c, reconstrction,(prec_h,observed_h) = model(data, delat, time_delat[j*batchsize:j*batchsize+batchsize])
            hidden, reconstrction, pred, hiddenback = model(aug_data, delat, mask)

            # plot_hidden(hidden,j)
            # loss1 = my_loss(torch.transpose(data,1,2),torch.transpose(reconstrction,1,2))
            loss1 = my_loss(data, reconstrction)
            reconstruct_loss += loss1.item()
            loss2 = my_loss(pred, hiddenback)
            loss = alpha * loss2 + loss1
            predict_loss += loss2.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        endtime = time.perf_counter()
        # for j in range(3):
        #     print(reconstrction[j])
        #     print(data[j])
        #     print("第",j,"组重构与真实结果对比")
        if i % 10 == 0:
        # print("end of epoch {} last loss1 {} loss2 {} learning time {}".format(i, loss1, loss2, endtime-starttime))
            print("end of epoch {} 重构损失 {} 预测损失 {} 训练时间 {}".format(i, reconstruct_loss, predict_loss, endtime - starttime))

    return model,(train_X,train_y,time_delat,test_X,test_y,test_time_delat)

def test_plot(train_X,args,model,time_delat) :
    device = args.device
    batchsize = args.batch_size
    batch_num = int(train_X.shape[0] / batchsize)
    if args.test_plot:
        for j in range(min(int(batch_num/2),5)):
            data, mask = mask_aug(train_X[j * batchsize:j * batchsize + batchsize], 0.000002)
            data, mask = torch.tensor(data).to(torch.float32).to(device), torch.tensor(mask).to(torch.float32).to(
                device)
            delat = torch.tensor(time_delat[j * batchsize:j * batchsize + batchsize]).to(torch.float32).to(device)
            # h, c, reconstrction,(prec_h,observed_h) = model(data, delat, time_delat[j*batchsize:j*batchsize+batchsize])
            hidden, reconstrction, pred, hiddenback = model(data, delat, mask)
            data = np.array(data.detach().cpu())
            reconstrction = np.array(reconstrction.detach().cpu())
            for i in range(int(batchsize/1)):
                for dim in range(min(4,data.shape[2])):
                    plt.subplot(2,2,dim+1)
                    plt.plot(np.arange(data[i,:,dim].shape[0]), data[i,:,dim],color='b')
                    plt.plot(np.arange(data[i,:,dim].shape[0]), reconstrction[i,:,dim],color='r')
                    plt.ylim(0,1)
                    # print(data[i,:,dim])
                    # print(reconstrction[i,:,dim])
                plt.show()
                plt.savefig("pic/"+str(j)+"_"+str(i)+"_contrs.jpg")
                plt.close()
            print("---------------------------------------------------------------------------------------------")

def test_eval(args,model,data):
    device = args.device
    train_X,train_y,time_delat,test_X,test_y,test_time_delat = data
    if args.test_eval:
        print("Evaluation begin\n")
        _, train_mask = mask_aug(train_X, args.test_missing_rate)
        _, test_mask = mask_aug(test_X, args.test_missing_rate)
        out, eval_res = eval_classification(model,train_data=train_X,train_labels=train_y,test_data=test_X,test_labels=test_y, train_delat = time_delat, test_delat = test_time_delat, device = device, train_mask = train_mask, test_mask = test_mask)
        print('Evaluation result:', eval_res)
        return eval_res


if __name__ == "__main__":
    args=get_parser()
    acc = 0
    count=0
    epoch=30
    print("begin train dataset {}".format(args.dataset))
    while True:
        args.epoch = epoch
        epoch += 10
        for i in range(20):
            model,data = train(args)
            result = test_eval(args=args,model=model,data=data)
            if result['acc'] > acc:
                acc = result['acc']
                PATH = "tempweights/" + args.dataset + str(args.epoch) + ".pt"
                print("save name as {}".format(PATH))
                torch.save(model.state_dict(), PATH)
                count=0
                with open(args.logfile, "a") as f:
                    f.write("dataset {} batchsize {} epoch {} predict_timestep {}\n".format(args.dataset, args.batch_size,
                                                                                            args.epoch,
                                                                                            args.predict_timestep))
                    f.write("Evaluation result: {}\n_________\n".format(result))
                    f.close()
            else:
                if count>20:
                    exit(0)
                else:
                    count+=1


