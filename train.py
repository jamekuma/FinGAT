import copy
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from torch import optim
import logging
import os.path as osp
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.graph_pool import CategoricalGraph, CategoricalGraphAtt, CategoricalGraphPool
from parse_arg import parse_basic_args

args = parse_basic_args()

# log 
args.log = (args.log if not args.log.endswith("/") else args.log[:-1]) + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = osp.join('logs', args.log)
os.makedirs(exp_dir, exist_ok=True)
logging.basicConfig(level=logging.DEBUG,
    format='%(levelname)s:%(asctime)s  %(message)s', datefmt='%Y-%d-%m-%H:%M:%S',
    handlers=[logging.FileHandler(osp.join(exp_dir, 'output.log')), logging.StreamHandler()])
logging.info(args)
writer = SummaryWriter(osp.join(exp_dir, "tb_log"))

# load data 
data_path = args.data
with open(data_path,"rb") as f:
    data = pickle.load(f)
# print(data["train"]["x1"].shape)
# exit(0)


# 全连接图
# inner_edge = np.array(np.load("./Taiwan_inner_edge.npy"))
# inner10_edge = np.array(np.load("./edge_10.npy"))
# inner20_edge = np.array(np.load("./Taiwan_inner_edge20.npy"))
# outer_edge = np.array(np.load("./Taiwan_outer_edge.npy"))



# data["train"]["x1"]: [num_weeks, ticker_num, time_step, input_dim]     (183, 50, 7, 28)
# exit(0)
time_step = data["train"]["x1"].shape[-2]
input_dim = data["train"]["x1"].shape[-1]
num_weeks = data["train"]["x1"].shape[0]
# train_size = int(num_weeks*0.2)     # useless
device = args.device
agg_week_num = args.week_num        # 3
n_category = 17
# print(data["train"]["x1"][0, 0, 0, -n_category:])
# print(data["train"]["x1"][0, 1, 0, -n_category:])
# print(data["train"]["x1"][0, 0, 1, -n_category:])
# print(data["train"]["x1"][0, 0:7, 0, -n_category:])

# 
# exit(0)

# convert data into torch dtype
train_w1 = torch.Tensor(data["train"]["x1"].astype(float)).float().to(device)
train_w2 = torch.Tensor(data["train"]["x2"].astype(float)).float().to(device)
train_w3 = torch.Tensor(data["train"]["x3"].astype(float)).float().to(device)
train_w4 = torch.Tensor(data["train"]["x4"].astype(float)).float().to(device)
# inner_edge = torch.tensor(inner_edge.T,dtype=torch.int64).to(device)
inner_edge = None
# inner10_edge = torch.tensor(inner10_edge.T,dtype=torch.int64).to(device)
# inner20_edge = torch.tensor(inner20_edge.T,dtype=torch.int64).to(device)
# outer_edge = torch.tensor(outer_edge.T,dtype=torch.int64).to(device)
outer_edge = None
# test data 
test_w1 = torch.Tensor(data["test"]["x1"].astype(float)).float().to(device)
test_w2 = torch.Tensor(data["test"]["x2"].astype(float)).float().to(device)
test_w3 = torch.Tensor(data["test"]["x3"].astype(float)).float().to(device)
test_w4 = torch.Tensor(data["test"]["x4"].astype(float)).float().to(device)
test_data = [test_w1,test_w2,test_w3,test_w4]#[-agg_week_num:]

# label data
train_reg = torch.Tensor(data["train"]["y_return ratio"]).float()
train_cls = torch.Tensor(data["train"]["y_up_or_down"]).float()
test_y = data["test"]["y_return ratio"] 
test_cls = data["test"]["y_up_or_down"] 
test_shape = test_y.shape[0]
loop_number = 100 if args.model == "CAT" else 10
ks_list = [5,10,20]
# use torch loader
# train_dataset = Data.TensorDataset(train_w1,train_w2,train_w3,train_w4,train_reg,train_cls)
# train_loader = Data.DataLoader(
#     dataset=train_dataset,     
#     batch_size=128,      
#     shuffle=True,               
# )

# check data shape
# print("Training shape:",train_x.shape,train_y.shape)
# print("Testing shape:",test_x.shape,test_y.shape)

def eval(model, dataset, gt_y, gt_cls, epoch, phase):
     # evaluate 
    model.eval()
    logging.info(f"[{phase}] Evaluate at epoch %s"%(epoch+1))
    y_pred, y_pred_cls = model.predict_toprank(dataset,device,top_k=5)

    # calculate metric 
    y_pred = np.array(y_pred).ravel()
    gt_y = np.array(gt_y).ravel()
    mae = mean_absolute_error(gt_y, y_pred)
    acc_score = Acc(gt_cls,y_pred_cls)

    results = []
    for k in ks_list:
        IRRs , MRRs ,Prs =[],[],[]
        for i in range(test_shape):
            M = MRR(np.array(gt_y[loop_number*i:loop_number*(i+1)]),np.array(y_pred[loop_number*i:loop_number*(i+1)]),k=k)
            MRRs.append(M)
            P = Precision(np.array(gt_y[loop_number*i:loop_number*(i+1)]),np.array(y_pred[loop_number*i:loop_number*(i+1)]),k=k)
            Prs.append(P)
        over_all = [mae,acc_score,np.mean(MRRs),np.mean(Prs)]
        results.append(over_all)
        
        writer.add_scalar(f"{phase}/K_{k}/mae", mae, epoch)
        writer.add_scalar(f"{phase}/K_{k}/acc_score", acc_score, epoch)
        writer.add_scalar(f"{phase}/K_{k}/MRRs", np.mean(MRRs), epoch)
        writer.add_scalar(f"{phase}/K_{k}/Prs", np.mean(Prs), epoch)
    
    print_res = [[round(j, 4) for j in i] for i in results]
    logging.info(f"[{phase}] " + str(print_res))

    # print('MAE:',round(mae,4),' IRR:',round(np.mean(IRRs),4),' MRR:',round(np.mean(MRRs),4)," Precision:",round(np.mean(Prs),4))
    performance = [mae,acc_score,np.mean(MRRs),np.mean(Prs)]        # last
    return performance, results

def train(args):
    global test_y
    model_name = args.model
    l2 = args.l2
    lr = args.lr
    beta = args.beta
    gamma = args.gamma 
    alpha = args.alpha
    device = args.device
    epochs = args.epochs
    hidden_dim = args.dim 
    use_gru = args.use_gru
    
    if model_name == "CG":
        assert False
        model = CategoricalGraph(input_dim,time_step,hidden_dim,inner10_edge,outer_edge,agg_week_num,device).to(device)
    elif model_name == "CAT":
        model = CategoricalGraphAtt(input_dim,time_step,hidden_dim,inner_edge,outer_edge,agg_week_num,use_gru,n_category,device).to(device)
    elif model_name == "CPool":
        assert False
        model = CategoricalGraphPool(input_dim,time_step,hidden_dim,inner_edge,inner20_edge,outer_edge,agg_week_num,use_gru,device).to(device)

    # initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Number of parameters:%s" % pytorch_total_params)

    # optimizer & loss 
    optimizer = optim.Adam(model.parameters(), weight_decay=l2,lr=lr)
    reg_loss_func = nn.L1Loss(reduction='none')
    cls_loss_func = nn.BCELoss(reduction='none')

    # save best model
    best_metric_IRR = None
    best_metric_MRR = None
    best_results_IRR = None
    best_results_MRR = None
    global_best_IRR = 999
    global_best_MRR = 0

    r_loss = torch.tensor([]).float().to(device)
    c_loss = torch.tensor([]).float().to(device)
    ra_loss = torch.tensor([]).float().to(device)
    for epoch in range(epochs):
        for week in range(num_weeks):
            model.train() # prep to train model
            batch_x1,batch_x2,batch_x3,batch_x4 = train_w1[week].to(device), \
                                                train_w2[week].to(device),\
                                                train_w3[week].to(device),\
                                                train_w4[week].to(device)
            # batch_x1: [ticker_num, time_step, dim]
            batch_weekly = [batch_x1,batch_x2,batch_x3,batch_x4][-agg_week_num:]
            # print(batch_weekly)
            # exit(0)
            batch_reg_y = train_reg[week].view(-1,1).to(device)
            batch_cls_y = train_cls[week].view(-1,1).to(device)
            reg_out, cls_out = model(batch_weekly)
            reg_out, cls_out = reg_out.view(-1,1), cls_out.view(-1,1)

            # calculate loss
            reg_loss = reg_loss_func(reg_out,batch_reg_y) # (target_size, 1) 
            cls_loss = cls_loss_func(cls_out,batch_cls_y)
            # print(reg_out)
            # print(batch_reg_y)
            print(reg_out.shape)
            print(batch_reg_y.shape)
            rank_loss = torch.relu(-(reg_out.view(-1,1)*reg_out.view(1,-1)) * (batch_reg_y.view(-1,1)*batch_reg_y.view(1,-1)))
            c_loss = torch.cat((c_loss,cls_loss.view(-1,1)))
            r_loss = torch.cat((r_loss,reg_loss.view(-1,1)))
            ra_loss = torch.cat((ra_loss,rank_loss.view(-1,1)))

            cls_loss = beta*torch.mean(c_loss)
            reg_loss = alpha*torch.mean(r_loss)
            rank_loss = gamma*torch.sum(ra_loss)
            loss = reg_loss + rank_loss + cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r_loss = torch.tensor([]).float().to(device)
            c_loss = torch.tensor([]).float().to(device)
            ra_loss = torch.tensor([]).float().to(device)
            if (week+1) % 144 ==0:  
                logging.info("REG Loss:%.4f CLS Loss:%.4f RANK Loss:%.4f  Loss:%.4f"% (reg_loss.item(),cls_loss.item(),rank_loss.item(),loss.item()))
            writer.add_scalar("train/reg_loss", reg_loss.item(), epoch * num_weeks + week)
            writer.add_scalar("train/cls_loss", cls_loss.item(), epoch * num_weeks + week)
            writer.add_scalar("train/rank_loss", rank_loss.item(), epoch * num_weeks + week)
            writer.add_scalar("train/loss", loss.item(), epoch * num_weeks + week)
        # evaluate 
        performance, results = eval(model, [test_w1,test_w2,test_w3,test_w4], test_y, test_cls, epoch, "val")
        eval(model, [train_w1,train_w2,train_w3,train_w4], train_reg, train_cls, epoch, "train")
        
        # print(performance)

        # save best 
        if performance[2] > global_best_MRR:
            global_best_MRR = performance[2]
            best_metric_MRR = performance
            best_results_MRR =  results
    
    return best_metric_IRR, best_metric_MRR, best_results_IRR, best_results_MRR


def MRR(test_y,pred_y,k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y
    
    predict = predict.sort_values("pred_y",ascending = False ).reset_index(drop=True)
    predict["pred_y_rank_index"] = (predict.index)+1
    predict = predict.sort_values("y",ascending = False )

    return sum(1/predict["pred_y_rank_index"][:k])


def Precision(test_y,pred_y,k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y
    
    predict1 = predict.sort_values("pred_y",ascending = False )
    predict2 = predict.sort_values("y",ascending = False )
    correct = len(list(set(predict1["y"][:k].index) & set(predict2["y"][:k].index)))
    return correct/k


def IRR(test_y,pred_y,k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y
    
    predict1 = predict.sort_values("pred_y",ascending = False )
    predict2 = predict.sort_values("y",ascending = False )
    return sum(predict2["y"][:k]) - sum(predict1["y"][:k])

def Acc(test_y,pred_y):
    test_y = np.ravel(test_y)
    pred_y = np.ravel(pred_y)
    pred_y = (pred_y>0)*1
    acc_score = sum(test_y==pred_y) / len(pred_y)

    return acc_score


if __name__ == "__main__":
    best_metric_IRR, best_metric_MRR, best_results_IRR, best_results_MRR = train(args)
    logging.info("-------Final result-------")
    logging.info("[BEST MRR] MAE:%.4f ACC:%.4f MRR:%.4f Precision:%.4f" % tuple(best_metric_MRR))
    for idx, k in enumerate(ks_list):
        logging.info("[BEST RESULT MRR with k=%s] MAE:%.4f ACC:%.4f MRR:%.4f Precision:%.4f" % tuple(tuple([k])+tuple(best_results_MRR[idx])))
