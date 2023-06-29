# coding=UTF-8
import socket
from tkinter import *
from GUI import TrainWindow


import torch
from utils import *
from connFun import *
from initDate import *

from torch.utils.data import DataLoader
from argu import Arguments


#实例化参数类
args = Arguments()


# 固定化随机数种子，使得每次训练的随机数都是固定的
torch.manual_seed(args.seed)

# #创建训练集
train_datasets = MyDataSet('data/NSL-KDD/training_attack_types.txt',"data/NSL-KDD/KDDTrain+_11_2.txt")


# #创建测试集
test_datasets = MyDataSet('data/NSL-KDD/training_attack_types.txt',"data/NSL-KDD/KDDTest+.txt",
                                    maxTemp=train_datasets.maxTemp,minTemp=train_datasets.minTemp)


all = sum([x[1] for x in train_datasets.counter.most_common()])
TP = torch.tensor([all/train_datasets.counter["begin"],all/train_datasets.counter["dos"],
                    all/train_datasets.counter["probe"],all/train_datasets.counter["u2r"],
                    all/train_datasets.counter["r2l"]]).log()#TP权重矩阵

if __name__ == '__main__':

    Socket_tcp=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    model = Net()
    
    try:
        root=Tk()
        root.geometry("1000x770+500+300")
        root.title("基于联邦学习的恶意流量检测系统客户端")
        app = TrainWindow(master=root,socket = Socket_tcp,model=model,
                train_datasets=train_datasets,test_datasets=test_datasets,args=args,TP=TP,name=2)
        root.mainloop()
    finally:
        Socket_tcp.close()