# coding=UTF-8
import pickle

from threading import Thread
import threading   
from connFun import recv_all, send_all,checkToken
from utils import federateModel
from model import Net
import socket

#端口
address = 8887

#验证令牌
TOKEN = "1"

#参与方数量
Client_num = 2
 
#聚合次数
EPOCH = 5

#全局模型
fedModel = None

#每轮聚合中有效线程的数量,用于标志清空models
num = 0

#存放每个客户端的模型
models = []

lock = threading.Lock()

def run(client_socket:socket.socket,addr):
    global models,num

    #当前聚合次数
    epoch = 0

    #创建原子操作
    lock.acquire()
    try:
        while True:
            #验证token
            client_token = recv_all(client_socket).decode("utf-8")
            print("%s : 服务器传输握手成功。。。"%str(addr))
            if checkToken(client_socket,client_token,TOKEN)==True:
                break
            

        send_all(client_socket,pickle.dumps(epoch))
        print("%s : 迭代次数已下发，等待客户端接收"%str(addr))
        recv_all(client_socket)
        
        #下发初始模型
        model = pickle.dumps(Net())
        send_all(client_socket,model)
        #传输成功握手
        recv_all(client_socket)

        #校验码，识别当前训练是否完成
        send_all(client_socket,"0".encode("utf-8"))
    finally:
        lock.release()

    print("%s : 初始模型已下发，正式开始训练。。。"%str(addr))

    while True:
        #创建原子操作
        lock.acquire()
        try:
            #接收本地模型
            client_model = recv_all(client_socket)
            client_model = pickle.loads(client_model)
            print('{} : 第{}轮迭代，服务器已接收到本地模型'.format(str(addr),str(epoch)))
            models.append(client_model)
        finally:
            lock.release()

        #等待其他客户端训练完成
        print("%s ： 等待其他客户端训练完成"%str(addr))
        while True:
            if len(models)==Client_num:
                break

        #开始聚合
        fedModel = federateModel(models)
        #创建原子操作
        lock.acquire()
        try:
            num+=1
            if num==Client_num:
                #所有线程都完成聚合
                models.clear()
                num=0
        finally:
            lock.release() 
        print("%s : 服务器已将全局模型聚合完成"%str(addr))
        epoch += 1
                        
        #下发全局模型
        fedModel = pickle.dumps(fedModel)

        while True:
            if len(models)==0:
                break

        lock.acquire()
        try:
            send_all(client_socket,pickle.dumps(epoch))
            print("%s : 迭代次数已下发，等待客户端接收"%str(addr))
            recv_all(client_socket)

            send_all(client_socket,fedModel)
            print("%s : 全局模型已下发，等待客户端接收"%str(addr))
            recv_all(client_socket)

            if epoch == EPOCH:
                #迭代次数完成
                print("%s : 迭代次数完成"%str(addr))
                send_all(client_socket,"1".encode("utf-8"))
                client_socket.close()
                break
            send_all(client_socket,"0".encode("utf-8"))
        finally:
            lock.release()

if __name__ == '__main__':

    #创建套接字
    Socket_tcp=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    #绑定并监听端口
    Socket_tcp.bind(("",address))
    Socket_tcp.listen(128)

    while True:
        client_socket,addr=Socket_tcp.accept()
        #监听到连接后，创建一个线程
        print("%s : 服务测试连接成功！\n"%str(addr))
        client_th = Thread(target=run,args=(client_socket,addr))
        client_th.start()        

    Socket_tcp.close()
