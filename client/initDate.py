from collections import Counter
import imp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyDataSet(Dataset):

    def __init__(self, typesPth,dataPth,minTemp=None,maxTemp=None,SMO=False):
        #---------------建立攻击映射字典---------------
        attackDir = {}
        attackDir['normal'] = 'begin'
        with open(typesPth, 'r') as f:
            for line in f.readlines():
                attack, spice = line.strip().split(' ')
                attackDir[attack] = spice
        f.close()
        self.attackDir = attackDir
        #-------------建立字符串到数值的映射-------------
        f_protocol={'tcp':0,'udp':1, 'icmp':2}
        f_service = {}
        #服务层的值
        f_temp=['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 
                'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 
                'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 
                'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
        for i in f_temp:
            f_service[i] = f_temp.index(i)
        f_flag = { 'OTH':0, 'REJ':1, 'RSTO':2, 'RSTOS0':3, 'RSTR':4, 'S0':5, 'S1':6, 'S2':7, 'S3':8, 'SF':9, 'SH':10 }
        f_attack_type = {"begin":0,"dos":1,"probe":2,"u2r":3,"r2l":4}

        #--------读入文件并进行训练集与测试集的划分----------
        header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
                        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
                        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']
        attack_info = pd.read_csv(dataPth,names=header_names)
        self.attack_info = attack_info.to_numpy()
        #对攻击类型进行映射
        attack_info["label"] = attack_info["attack_type"].map(lambda x:attackDir[x])

        self.counter = Counter(attack_info['label'])
        #去掉最后两列
        attack_info.drop("attack_type",axis=1,inplace=True)
        attack_info.drop("success_pred",axis=1,inplace=True)
        #对protocol进行映射
        attack_info["protocol_type"] = attack_info["protocol_type"].map(lambda x:f_protocol[x])
        #对service进行映射
        attack_info["service"] = attack_info["service"].map(lambda x:f_service[x])
        #对flag进行映射
        attack_info["flag"] = attack_info["flag"].map(lambda x:f_flag[x])
        self.data = attack_info.to_numpy()
        #得到不含标签的数据
        self.x = self.data[:,:-1]
        self.y = self.data[:,-1]
        #对攻击类型进行独热编码映射
        self.y = np.array([f_attack_type[i] for i in self.y])
        self.normalization(minTemp=minTemp,maxTemp=maxTemp)
        self.len = len(self.y)
    
    def __getitem__(self, index: int):
        bin_data=[]
        for w in self.x[index]:
            for p in range(256):
                if(p<=int(w*255)):
                    bin_data.append(1)
                else:
                    bin_data.append(0)
        temp = np.array(bin_data).reshape((1,41,256)).tolist()
        temp = torch.tensor(temp).float()
        temp.requires_grad_(True)
        return temp,self.y[index]

    def getDateByIndex(self,index):
        #依次返回x和y
        return self.attack_info[:,:-2][index],self.attack_info[:,-2][index]
    
    def __len__(self):
        return self.len

    def normalization(self,minTemp=None,maxTemp=None):
        #对self.x进行数据归一化
        if maxTemp == None:
            minTemp = [self.x[:,i].min() for i in range(len(self.x[0]))]
            maxTemp = [self.x[:,i].max() for i in range(len(self.x[0]))]
        for i in range(len(self.x)):
            for j in range(len(self.x[0])):
                if maxTemp[j] == minTemp[j] :
                    #'num_outbound_cmds', 'is_host_login'全为0，进行跳过
                    continue
                self.x[i,j] = (self.x[i,j]-minTemp[j])/(maxTemp[j]-minTemp[j])
        self.minTemp=minTemp
        self.maxTemp = maxTemp

    def getMinTemp(self):
        return self.minTemp

    def getMaxTemp(self):
        return self.maxTemp

    def getSetInfo(self):
        #返回数据集信息
        return Counter(self.counter)


if __name__=='__main__':
    dataset = MyDataSet('../data/NSL-KDD/training_attack_types.txt',"../data/NSL-KDD/KDDTrain+.txt")
    x=3350
    img_numpy = dataset.__getitem__(x)[0].detach().reshape(41,256,1).numpy()
    print(dataset.__getitem__(x)[1])
    plt.imshow(img_numpy,cmap=plt.get_cmap('gray'))
    plt.show()
    #pic.save('random.jpg')
