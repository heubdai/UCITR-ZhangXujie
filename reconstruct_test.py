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
import pandas as pd

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

def save_data(data,enddata):
    # data = np.array(data.detach().cpu().numpy())
    enddata = np.array(enddata.detach().cpu().numpy())
    print("begin data saving")
    writer1 = pd.ExcelWriter('start_data.xlsx')  # 创建excel文件
    writer2 = pd.ExcelWriter('end_data.xlsx')  # 创建excel文件
    for i in range(data.shape[0]):
        startdata = pd.DataFrame(data[i])  # 转换为DataFrame对象
        startdata.to_excel(writer1, 'sheet_1'+str(i), float_format='%.2f')  # 写入数据，保留两位小数
        end_data = pd.DataFrame(enddata[i])
        end_data.to_excel(writer2, 'sheet_1'+str(i), float_format='%.2f')  # 写入数据，保留两位小数
    writer1.save()  # 保存文件
    writer2.save()  # 保存文件
    writer1.close()  # 关闭文件
    writer2.close()  # 关闭文件

if __name__ == "__main__":
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
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')

    args = parser.parse_args()

    PATH = "tempweights/" + args.dataset+str(args.epoch)+".pt"
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
        train_X, eval_train, eval_train_label, eval_test, eval_test_label = load_electricity(args.dataset,96,24)

    elif args.loader == 'ETT':
        task_type = 'forecasting'
        # elec_data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(args.dataset)
        # train_X = elec_data[:, train_slice]
        # args.dataset = "LD2011_2014.txt"
        train_X, eval_train, eval_train_label, eval_test, eval_test_label = load_electricity(args.dataset,96,128)

    device = args.device
    epoch = args.epoch
    print("total epoch ",epoch)
    batchsize = args.batch_size
    time_delat = np.ones(train_X.shape[0:2])
    print("time_delat shape ", time_delat.shape)
    print("train x shape ", train_X.shape)
    # model = UCITR_Linear(input_dim=train_X.shape[2], hidden_dim=800, length=train_X.shape[1], device=device, args=args).to(device)
    model = UCITR(input_dim=train_X.shape[2], hidden_dim=args.repr_dims, length=train_X.shape[1], device=device, args=args).to(device)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
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
            data = torch.tensor(t_train_X[j*batchsize:j*batchsize+batchsize]).to(torch.float32).to(device)
            aug_data, mask = mask_aug(t_train_X[j*batchsize:j*batchsize+batchsize],args.train_missing_rate)
            aug_data, mask = torch.tensor(aug_data).to(torch.float32).to(device), torch.tensor(mask).to(torch.float32).to(device)
            delat = torch.tensor(time_delat[j*batchsize:j*batchsize+batchsize]).to(torch.float32).to(device)
            # h, c, reconstrction,(prec_h,observed_h) = model(data, delat, time_delat[j*batchsize:j*batchsize+batchsize])
            hidden, reconstrction,pred,hiddenback = model(aug_data, delat, mask)

            if i == epoch-1 and j == 0:
                save_data(t_train_X[j*batchsize:j*batchsize+batchsize],reconstrction)
            # plot_hidden(hidden,j)
            # loss1 = my_loss(torch.transpose(data,1,2),torch.transpose(reconstrction,1,2))
            loss1 = my_loss(data,reconstrction)
            reconstruct_loss += loss1.item()
            loss2 = my_loss(pred, hiddenback)
            loss = alpha*loss2+loss1
            predict_loss += loss2.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        endtime = time.perf_counter()
        # for j in range(3):
        #     print(reconstrction[j])
        #     print(data[j])
        #     print("第",j,"组重构与真实结果对比")
        # if i % 10 == 0:
        # print("end of epoch {} last loss1 {} loss2 {} learning time {}".format(i, loss1, loss2, endtime-starttime))
        print("end of epoch {} 重构损失 {} 预测损失 {} 训练时间 {}".format(i, reconstruct_loss, predict_loss, endtime-starttime))

    if epoch != 0:
        print("save name as {}".format(PATH))
        torch.save(model.state_dict(), PATH)
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
                    plt.ylim(-3,3)
                    # print(data[i,:,dim])
                    # print(reconstrction[i,:,dim])
                plt.show()
                plt.savefig("pic/"+str(j)+"_"+str(i)+"_contrs.jpg")
                plt.close()
            print("---------------------------------------------------------------------------------------------")

    if args.test_eval:
        if task_type == 'classification':
            print("Classification Evaluation begin\n")
            _, train_mask = mask_aug(train_X, args.test_missing_rate)
            _, test_mask = mask_aug(test_X, args.test_missing_rate)
            test_time_delat = np.ones(test_X.shape[0:2])
            out, eval_res = eval_classification(model,train_data=train_X,train_labels=train_y,test_data=test_X,test_labels=test_y, train_delat = time_delat, test_delat = test_time_delat, device = device, train_mask = train_mask, test_mask = test_mask)
            with open(args.logfile,"a") as f:
                f.write("dataset {} batchsize {} epoch {} predict_timestep {}\n".format(args.dataset,args.batch_size,args.epoch,args.predict_timestep))
                f.write("Evaluation result: {}\n_________\n".format(eval_res))
                f.close()
            print('Evaluation result:', eval_res)
        elif task_type == 'forecasting':
            print("Forecasting Evaluation begin\n")
            _, train_mask = mask_aug(eval_train, args.test_missing_rate)
            _, test_mask = mask_aug(eval_test, args.test_missing_rate)
            time_delat = np.ones(eval_train.shape[0:2])
            test_time_delat = np.ones(eval_test.shape[0:2])
            loss = eval_forecasting(model, train_data=eval_train, test_data=eval_test, train_delat = time_delat, test_delat = test_time_delat, train_mask = train_mask, test_mask = test_mask, args=args,train_real=eval_train_label,test_real=eval_test_label)
