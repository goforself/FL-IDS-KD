# coding=UTF-8
import socket

"""
    超大消息发送的函数封装
    不允许出现recv_all或send_all连续的情况，否则可能出现覆盖接收
"""

def recv_all(socket:socket.socket)->bytes:
    """
        从socket中获取大容量信息，返回bytes类型数据
    """
    res = b''
    #获取发送信息的长度
    res_size=int(socket.recv(1024).decode("utf-8"))
    #发送校验
    socket.send("0".encode("utf-8"))

    size=0
    while size<res_size:
        #拼接发送信息
        temp=socket.recv(1024)
        res+=temp
        size+=len(temp)
    return res

def send_all(socket:socket.socket,data:bytes):
    """
        往指定socket中发送data数据
    """
    socket.send(("%s"%len(data)).encode("utf-8"))
    socket.recv(1024)
    socket.send(data)

def sendToken(socket:socket.socket,token):
    #向套接字发送token
    send_all(socket,token.encode("utf-8"))
    #接收token校验值
    tokenCheck = recv_all(socket).decode("utf-8")
    send_all(socket,"0".encode("utf-8"))
    
    if tokenCheck=="1":
        #token验证失败
        print("token验证失败")
        return False
    else:
        print("token正确 连接成功")
        return True

def checkToken(socket:socket.socket,client_token,token):
    #验证token
    if client_token != token :
        send_all(socket,"1".encode("utf-8"))
        recv_all(socket)
        return False
    else:
        send_all(socket,"0".encode("utf-8"))
        recv_all(socket)
        return True