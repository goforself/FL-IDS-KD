import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        #输入维度为1，输出维度为20，卷积核大小为：5*5，步幅为1
        self.conv1 = nn.Conv2d(1,10,3,1)  
        self.conv2 = nn.Conv2d(10,20,3,1) 
        self.conv3 = nn.Conv2d(20,20,3,1) 
        self.fc1 = nn.Linear(1800,300)
        #最后映射到5维上
        self.drop = torch.nn.Dropout(p=0.5)
        self.drop2 = torch.nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(300,5)

    def forward(self,x):
        #print(x.shape)
        x = F.tanh(self.conv1(x))#41*256*1 -> 39*254*10
        x = F.max_pool2d(x,2,2)#39*254*10 -> 19*127*10
        #print(x.shape)
        x = self.drop2(x)
        x = F.tanh(self.conv2(x))#19*127*10 -> 17*125*20
        x = F.max_pool2d(x,2,2)#17*125*20 -> 8*62*20
        #print(x.shape)
        x = self.drop2(x)
        x = F.tanh(self.conv3(x))#8*62*20 -> 6*60*20
        x = F.max_pool2d(x,2,2)#6*60*20 -> 3*30*20
        #print(x.shape)
        x = x.view(-1,1800)
        x= self.drop(x)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
