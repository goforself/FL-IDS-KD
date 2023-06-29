#用于构建优化器
import torch
import torch.optim as optim
from torch import contiguous_format, nn
from torch.utils.data import DataLoader
import gc

from model import Net


def softLoss(student_pred:torch.tensor,teacher_pred:torch.tensor,T:int):
    """
        在client中计算两个softTarget的损失
        student_pred：学生的软标签
        teacher_pred：教师的软标签
        T：蒸馏温度
    """
    #在Client中计算预测值与真实值的损失
    s_pred = nn.Softmax(dim=1)(student_pred/T)
    t_pred = nn.Softmax(dim=1)(teacher_pred/T)
    return nn.KLDivLoss(reduction="batchmean")(s_pred.log(),t_pred)

def local_train(model:Net,train_loader:DataLoader,test_loader:DataLoader,args,TP,g_epoch,g_pred):
    """
        model:待训练的模型
        train_loader：训练集
        test_loader：测试集
        args：训练基本参数
        C：初始化代价敏感指标
        TP：数据集不平衡度
        g_epoch:全局迭代次数
        g_pred:全局软标签，若为nil，则说明是本地测试组
    """

    #初始化贝叶斯代价敏感指标
    C = torch.full((1,5),0.8)[0]#代价敏感贝叶斯矩阵：每一类错误分类的概率，初始全为0.8

    #定义Bob的优化器
    opt = optim.Adam(model.parameters(), lr = args.lr)

    model.train()

    FF = 0.8#先验概率分类错误情况下，模型分类错误的概率,FF
    FT = 0.2#先验概率分类错误情况下，模型分类正确的概率,FT
    TF = 0.2#先验概率分类正确情况下，模型错误分类的概率,TF
    TT = 0.8#先验概率分类正确情况下，模型正确分类的概率,TT

    for epoch in range(1, args.epochs +1):
        #print(Bob_model.conv1.weight)
        soft_target = torch.full((5,5),0.0).detach()
        #防止除0错误
        num = torch.tensor([0.0000001]*5)
        #传递模型
        for epoch_ind, (data, target) in enumerate(train_loader):
            opt.zero_grad()
            CEloss = nn.CrossEntropyLoss(reduction='none')
            client_pred = model(data)
            temp = torch.full((1,len(target)),0.0)[0]
            #计算代价敏感指标
            for y_h,y_t, in zip(client_pred,target):
                if y_h.argmax() == y_t.item():
                    #预测正确
                    P_B1 = C[y_t.item()]
                    P_B2 = 1-P_B1
                    C[y_t.item()] = (P_B1*TF)/(P_B1*TF+P_B2*TT)#贝叶斯公式
                    if C[y_t.item()]<0.05:
                        C[y_t.item()] = 0.05
                else:
                    #预测错误
                    P_B1 = C[y_t.item()]
                    P_B2 = 1-P_B1
                    C[y_t.item()] = (P_B1*FF)/(P_B1*FF+P_B2*FT)#贝叶斯公式
                    if C[y_t.item()]>0.95:
                        C[y_t.item()] = 0.95
            for i,y_t in zip(range(len(target)),target):
                temp[i] = C[y_t]*TP[y_t]

            client_loss = CEloss(client_pred,target)
            #GL为列表，这里只是暂存
            GL = []
            #计算全局损失函数
            if g_epoch != 0:
                #第二次以上训练
                if g_pred != None:
                    teacher_loss = softLoss(client_pred,g_pred[target],T=5)
                    GL = 0.7*client_loss+0.3*teacher_loss
                else:
                    GL = client_loss+0.0
            else:
                #第一次训练
                GL = client_loss+0.0
            
            #损失函数为y和y_hat的交叉熵，然后乘以代价敏感指标
            loss = torch.mean(temp*GL)
            loss.backward()
            opt.step()

            if g_pred!=None:
                #生成soft_target
                soft_target[target] += client_pred.detach()
                num[target]+=1

            if epoch_ind%100==0 :
                print("There is epoch:{} epoch_ind:{} loss:{:.6f}".format(epoch,epoch_ind,loss.data.item()))
                gc.collect()
                break
        result = 0
        if epoch % args.log_interval==0:
            #获得loss
            result = test(model,test_loader,"test:")
            
    return soft_target/num,result

#定义测试函数
def test(model, test_loader,modelName=None):
    """进行测试，并返回测试结果"""
    model.eval()
    TP = torch.tensor([0.,0.,0.,0.,0.])
    FP = torch.tensor([0.,0.,0.,0.,0.])
    FN = torch.tensor([0.,0.,0.,0.,0.])
    TN = torch.tensor([0.,0.,0.,0.,0.])
    CEloss = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    label = ["begin","dos","probe","u2r","r2l"]
    correctNum = {"begin":0,"dos":0,"probe":0,"u2r":0,"r2l":0}
    totalNum = {"begin":0,"dos":0,"probe":0,"u2r":0,"r2l":0}
    tartex = torch.tensor([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    loss = 0
    with torch.no_grad():
        for ind,(data, target) in enumerate(test_loader):
            output=model(data)
            loss += CEloss(output,target).item()
            if output.argmax() == target.item():
                #分类正确
                correct+=1
                TP[target.item()]+=1.
                correctNum[label[output.argmax()]]+=1
                #加其余四个的FN
                for i in range(5):
                    if target.item()!=i:
                        TN[i]+=1
            else:
                FP[output.argmax()]+=1.
                FN[target.item()]+=1.
                #加其余三个的FN
                for i in range(5):
                    if target.item()!=i and output.argmax()!=i:
                        TN[i]+=1

            totalNum[label[target.item()]]+=1
            tartex[target.item()][output.argmax()]+=1
            break
        test_loss /= len(test_loader.dataset)
        print(modelName)
        print('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.1f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100.* correct / len(test_loader.dataset)))
        print('{}/{}   {}/{}   {}/{}    {}/{}    {}/{}\n'.format(
            correctNum['begin'],totalNum["begin"],
            correctNum['dos'],totalNum["dos"],
            correctNum['probe'],totalNum["probe"],
            correctNum['u2r'],totalNum["u2r"],
            correctNum['r2l'],totalNum["r2l"]))
        print("召回率：",TP/(TP+FN))
        print("查准率：",TP/(TP+FP))
        print("准确率：",(TP+TN)/(TP+TN+FP+FN))
        print("F1:",2/(1/(TP/(TP+FP))+1/(TP/(TP+FN))))
        print(tartex)
        with open("./data3.log","a+") as f:
            f.write('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.1f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100.* correct / len(test_loader.dataset)))
            f.write('{}/{}   {}/{}   {}/{}    {}/{}    {}/{}\n'.format(
            correctNum['begin'],totalNum["begin"],
            correctNum['dos'],totalNum["dos"],
            correctNum['probe'],totalNum["probe"],
            correctNum['u2r'],totalNum["u2r"],
            correctNum['r2l'],totalNum["r2l"]))
            f.write("召回率：{}\n".format(TP/(TP+FN)))
            f.write("查准率：{}\n".format(TP/(TP+FP)))
            f.write("准确率：{}\n".format((TP+TN)/(TP+TN+FP+FN)))
            f.write("F1：{}\n".format(2/(1/(TP/(TP+FP))+1/(TP/(TP+FN)))))
            f.write(str(tartex))
            f.close()
    #防止除0错误，加上bias防止分母为0
    bias = 0.0001
    return [100.*correctNum['begin']/(totalNum["begin"]+bias),
            100.*correctNum['dos']/(totalNum["dos"]+bias),
            100.*correctNum['probe']/(totalNum["probe"]+bias),
            100.*correctNum['u2r']/(totalNum["u2r"]+bias),
            100.*correctNum['r2l']/(totalNum["r2l"]+bias),
            100.* correct / len(test_loader.dataset)]

def federateModel(models):
    """
        对所有软标签相加求和
    """
    m = torch.full((5,5),0.0)
    for model in models:
        m+=model
    m/=len(model)
    return m