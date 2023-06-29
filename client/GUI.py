from distutils.log import error
import pickle
import random
import socket
from copy import deepcopy
from cProfile import label
from hmac import trans_36
from os import access
from telnetlib import GA
from tkinter import *
from tkinter import messagebox
from turtle import title
from unicodedata import name

import matplotlib.pyplot as plot
import numpy as np

from argu import Arguments
from connFun import *
from initDate import *
from model import *
from utils import *

# 输入提示字典
inputTip = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
            "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
            "num_shells","num_access_files","num_outbound_cmds","is_hot_login","is_guest_login","count","srv_count",
            "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
            "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","1"]

# 映射协议类型
f_protocol={'tcp':0,'udp':1, 'icmp':2,"1":4}

#目标主机的网络服务类型
f_service = {}
f_temp=['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 
                'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 
                'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 
                'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50',"1"]
for i in f_temp:
    f_service[i] = f_temp.index(i)

#连接正常或错误的状态
f_flag = { 'OTH':0, 'REJ':1, 'RSTO':2, 'RSTOS0':3, 'RSTR':4, 'S0':5, 'S1':6, 'S2':7, 'S3':8, 'SF':9, 'SH':10 ,"1":5}

#映射输出类型
f_attack_type = ["begin","dos","probe","u2r","r2l"]

class TrainWindow(Frame):
    """一个经典的GUI程序的类的写法"""
    def __init__(self, master=None, socket:socket=None, model:Net=None, train_datasets:MyDataSet=None, 
                        test_datasets:MyDataSet=None,args:Arguments=None, TP:torch.tensor=None, name=None):
        super().__init__(master)      #super()代表的是父类的定义，而不是父类对象

        if train_datasets!=None:
            train_loader = DataLoader(train_datasets,batch_size=args.batch_size,shuffle=True)
            #训练集信息
            self.trainInfo = train_datasets.getSetInfo()
            #训练集
            self.train_loader = train_loader
            #训练数据集
            self.trainDataSet = train_datasets
        if test_datasets!=None:
            test_loader = DataLoader(test_datasets,batch_size=args.test_batch_size,shuffle=True)
            #测试集信息
            self.testInfo = test_datasets.getSetInfo()
            #测试集
            self.test_loader = test_loader
            #训练数据集
            self.testDataSet = test_datasets

        #顶层窗口
        self.master=master
        #TCP连接
        self.socket = socket
        #训练模型
        self.model = model
        #本地对比模型
        self.testModel = deepcopy(model)
        #训练后保存图片的名字
        self.name = name


        #训练参数
        self.args = args
        #数据集不平衡度
        self.TP = TP
        
        self.place(x=0,y=0)

        self.createWidget()
    def createWidget(self):
        """创建组件"""
        # 创建IP提示
        self.IPlabel = Label(self,text="IP address：",font=("黑体",18))
        self.IPlabel.grid(row=0,column=0,columnspan=1,sticky='e',padx=(30,8),pady=40)

        # 创建IP输入
        self.IPinput = Entry(self,width=15,font=("黑体",18))
        self.IPinput.grid(row=0,column=1,columnspan=1,padx=15,pady=40)
        self.IPinput.insert(0,"127.0.0.1")

        # 创建Port提示
        self.portLabel = Label(self,text="Port number：",font=("黑体",18))
        self.portLabel.grid(row=0,column=2,columnspan=1,sticky='e',padx=(30,8),pady=40)

        # 创建Port输入
        self.portInput = Entry(self,width=10,font=("黑体",18))
        self.portInput.grid(row=0,column=3,columnspan=1,padx=15,pady=40)
        self.portInput.insert(0,'8887')

        #创建网络测试按钮
        self.btnConnectTest = Button(self,text="connect test",font=("黑体",18))
        self.btnConnectTest.grid(row=0,column=5,columnspan=1)
        self.btnConnectTest["command"] =self.connectTest

        # 创建口令提示
        self.labelToken = Label(self,text="input token：",font=("黑体",18))
        self.labelToken.grid(row=1,column=0,columnspan=1,sticky='e',padx=(30,8),pady=40)

        # 创建口令输入
        self.entryToken = Entry(self,width=48,font=("黑体",18))
        self.entryToken.grid(row=1,column=1,columnspan=4,padx=15,pady=40)

        #创建提交口令按钮
        self.btnToken = Button(self,text="submit token",font=("黑体",18))
        self.btnToken.grid(row=1,column=5,columnspan=1)
        self.btnToken["command"] =self.sendTokenClick

        # 创建一个提示文本框
        self.textTrain = Text(self,width=95,height=25,font=("Helvetica",15))
        self.textTrain.grid(row=2,column=0,columnspan=10,rowspan=5,padx=20)

        # 创建测试按钮
        self.testBtn = Button(self,text="test model",font=("黑体",18))
        self.testBtn.grid(row=7,column=3,columnspan=1,padx=15,pady=40)
        self.testBtn["command"] =self.create

        # 创建查看数据集按钮
        self.testBtn = Button(self,text="view test set",font=("黑体",18))
        self.testBtn.grid(row=7,column=1,columnspan=1,padx=15,pady=40)
        self.testBtn["command"] =self.viewSet

    def create(self):
        #创建测试模型的窗口
        top = Toplevel()

        self.top = top

        top.title('test the model')
        top.geometry("1500x850+250+150")
        top.Entry=[]

        myfont = 17
        mypadx = 10
        mypady = 10

        myrow = 0
        mycolumn = 0

        for i in range(0,41):
            strEntry = StringVar(value="")

            #提示标签
            top.Label = Label(top,width=30,text=inputTip[i]+":",font=("黑体",myfont))
            top.Label.grid(row=myrow,column=mycolumn,padx = mypadx,pady = mypady)

            mycolumn += 1
            if mycolumn == 6:
                #切换到下一行
                myrow += 1
                mycolumn = 0

            #文本输入
            top.Entry.append(Entry(top,width=10,font=("黑体",myfont),textvariable=strEntry))

            top.Entry[i].grid(row=myrow,column=mycolumn,padx = mypadx,pady = mypady)

            mycolumn += 1
            if mycolumn == 6:
                #切换到下一行
                myrow += 1
                mycolumn = 0

        # 创建IP提示
        top.labelMsg = Label(top,text="Label：",font=("黑体",25))
        top.labelMsg.grid(row=15,column=0,columnspan=1,padx = mypadx,pady = mypady)

        # 创建IP提示
        top.labelValue = Label(top,text="",font=("黑体",25))
        top.labelValue.grid(row=15,column=1,columnspan=2,padx = mypadx,pady = mypady)

        # 创建IP提示
        top.predMsg = Label(top,text="Predict:",font=("黑体",25))
        top.predMsg.grid(row=15,column=3,columnspan=1,padx = mypadx,pady = mypady)

        # 创建IP提示
        top.predValue = Label(top,text="",font=("黑体",25))
        top.predValue.grid(row=15,column=4,columnspan=2,padx = mypadx,pady = mypady)

        # 创建填充内容按钮
        top.initBtn = Button(top,text="Get Random Data",font=("黑体",20))
        top.initBtn.grid(row=18,column=0,columnspan=3,padx=15,pady=40)
        top.initBtn["command"] = self.initData

        # 创建测试按钮
        top.predBtn = Button(top,text="Predict The Attack Type",font=("黑体",20))
        top.predBtn.grid(row=18,column=3,columnspan=3,padx=15,pady=40)
        top.predBtn["command"] = self.pred

    def initData(self):
        #在数据集中随机抽取一条数据，并将其设置为标签的值
        index = random.randint(0,1000)
        x,y = self.testDataSet.getDateByIndex(index)
        y = self.testDataSet.attackDir[y]
        for i in range(0,41):
            #删除原始数据
            self.top.Entry[i].delete(0,END)
            #插入数据
            self.top.Entry[i].insert(0,x[i])
        self.top.labelValue.configure(text=y)
        self.top.predValue.configure(text="")

    def connectTest(self):
        #获取IP的值
        IP = self.IPinput.get()
        self.IPinput.delete(0,'end')

        #获取端口号
        Port = eval(self.portInput.get())
        self.portInput.delete(0,'end')

        try:
            self.socket.connect((IP,Port))
            self.textTrain.insert("end","connect successfully!\n")
        except Exception:
            self.textTrain.insert("end","error message,connection failed!\n")
        
    def sendTokenClick(self):
        """
            发送entryToken的内容到服务器，并提示发送成功
            同时开始本地训练
        """
        token = self.entryToken.get()
        self.entryToken.delete(0,'end')

        messagebox.showinfo("信息","token:"+token+" send successfully!")

        #向服务器发送token
        if sendToken(self.socket,token)==True:
            self.textTrain.insert("end","sever checked token： "+token+" right!\n")
            self.textTrain.update()
            self.train()
        else:
            self.textTrain.insert("end","sever checked token： "+token+" wrong，please input again!\n")
            return 

    def train(self):
        globalAcc = []
        testAcc = []
        while True:
            #获取当前迭代次数
            recv_data = recv_all(self.socket)
            epoch = pickle.loads(recv_data)
            send_all(self.socket,"0".encode("utf-8"))

            #接收蒸馏标签
            recv_data = recv_all(self.socket)
            # print("after :"+str(len(recv_data)))
            g_pred=pickle.loads(recv_data)
            self.textTrain.insert("end","client received the soft-target!\n")
            self.textTrain.update()
            send_all(self.socket,"0".encode("utf-8"))

            #接收校验码
            checkCode = recv_all(self.socket).decode("utf-8")
            if checkCode=="1":
                self.textTrain.insert("end","train has completed!\n")
                self.textTrain.update()
                self.socket.close()
                break
            
            self.textTrain.insert("end","client begin to train model!\n")
            self.textTrain.update()
            #开始训练
            
            #个性化训练
            soft_target,globalAcc_temp = local_train(self.model,self.train_loader,self.test_loader,self.args,self.TP,epoch,g_pred)
            _,testAcc_temp = local_train(self.model,self.train_loader,self.test_loader,self.args,self.TP,epoch,g_pred=None)

            globalAcc.append(globalAcc_temp)
            testAcc.append(testAcc_temp)

            #上传本地模型
            model_serial=pickle.dumps(soft_target)
            send_all(self.socket,model_serial)
            self.textTrain.insert("end","client has send the local model to sever\n")
            self.textTrain.update()

        #可视化
        self.viewResult(np.array(globalAcc),np.array(testAcc),self.name)

        torch.save(self.model,"local_model.weight")
        torch.save(self.testModel,"local_testModel.weight")
    
    def viewSet(self):
        #得到训练集和测试集的柱状图

        #攻击类型
        xAxis = ["begin","dos","probe","r2l","u2r"]

        testInfo = self.testInfo
        yAxisTest = [testInfo[x] for x in xAxis]
        trainInfo = self.trainInfo
        yAxisTrain = [trainInfo[x] for x in xAxis]

        #单个柱状图的宽度
        weith = 0.4

        #确定y轴
        plt.bar([i for i in range(len(xAxis))],yAxisTrain,label='train',width=weith)
        plt.bar([i+weith for i in range(len(xAxis))],yAxisTest,label='test',width=weith)

        #确定x轴
        plt.xticks([i+weith/2 for i in range(len(xAxis))],xAxis)

        #训练样本数量
        for x,y in zip([i for i in range(len(xAxis))],yAxisTrain):
            plt.text(x,y,'%d'%y,ha="center")

        #测试样本数量

        for x,y in zip([i for i in range(len(xAxis))],yAxisTest):
            plt.text(x+weith,y,'%d'%y,ha="center")

        plt.title("train set information")
        plt.xlabel("attack type")
        plt.ylabel("sample number")
        plt.legend()
        plt.show()

    def viewResult(self,globalAcc,testAcc,name=None):
        """查看训练结果并存储"""
        #攻击类型
        title = ["begin","dos","probe","u2r","r2l","total"]

        fig,axs = plt.subplots(nrows = 2, ncols = 3,figsize = (20,6),dpi=100)

        for i in range(6):
            gAcc = globalAcc[:,i]
            tAcc = testAcc[:,i]

            axs[i//3][i%3].plot([i for i in range(len(gAcc))],gAcc,label="federated model")
            axs[i//3][i%3].plot([i for i in range(len(tAcc))],tAcc,label="local model")
            axs[i//3][i%3].set_xlabel("Epoch")
            axs[i//3][i%3].set_ylabel("Acc")
            axs[i//3][i%3].set_title(title[i])
            axs[i//3][i%3].legend(loc='upper left')
        # fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig("resultCompare{}.png".format(name))
        plt.show()


    def pred(self):
        #对读取的内容使用模型测试

        inputTest = []
        for x in self.top.Entry:
            if x.get() == "":
                messagebox.showinfo("预测结果","input arguments lacked")
                return 
            inputTest.append(x.get())


        #------------字典映射，将字符串映射为数字------------
        #映射协议类型
        inputTest[1] = str(f_protocol[inputTest[1]])

        #目标主机的网络服务类型
        inputTest[2] = str(f_service[inputTest[2]])

        # 连接正常或错误的状态
        inputTest[3] = str(f_flag[inputTest[3]])
        #------------字典映射，将字符串映射为数字------------

        # print(inputTest)

        # 归一化
        minTemp = self.trainDataSet.getMinTemp()
        maxTemp = self.trainDataSet.getMaxTemp()
        for i in range(0,len(inputTest)):
            if maxTemp[i] == minTemp[i] :
                #'num_outbound_cmds', 'is_host_login'全为0，进行跳过
                continue
            temp = eval(inputTest[i])
            inputTest[i] = (temp-minTemp[i])/(maxTemp[i]-minTemp[i])

        bin_data=[]
        for w in inputTest:
            for p in range(256):
                if(p<=int(w*255)):
                    bin_data.append(1)
                else:
                    bin_data.append(0)
        
        temp = np.array(bin_data).reshape((1,1,41,256)).tolist()
        temp = torch.tensor(temp).float()

        # 加载训练完成的模型
        local_model = Net()
        local_model.load_state_dict(torch.load("Net.weight"))
        pred = local_model(temp)
        pred = f_attack_type[pred.argmax()]
        
        self.top.predValue.configure(text=pred)

if __name__ == "__main__":
    root=Tk()
    root.geometry("1000x770+500+300")
    root.title("基于联邦学习的恶意流量检测系统客户端")
    app = TrainWindow(master=root)
    root.mainloop()
