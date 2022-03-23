#!/usr/bin/env python
# encoding: utf-8
import socket
import json
import pickle
import time
import urllib
import urllib.request
import cv2
from run import *

"""
protobuf序列化
netty编解码有很多种 objectencoder stringencoder 一般string比较好 对于服务端与客户端语言用的不一致
最终的传输都是编码bytes
如果要用objectencode 可以用thrift定义一个类
LengthFieldBasedFrameDecoder
"""

class SocketService:
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.data_string = None

        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # 客户端心跳维护
            self.s.connect((host, port))
            print("连接成功")
        except:
            print("无法连接服务器")

    def login(self, type_id, ID, UserName, Password):
        login_msg = {'type': type_id, 'clientId': ID, 'userName': UserName, 'password': Password}
        login_msg_json = json.dumps(login_msg)
        self.s.send(login_msg_json.encode('utf-8'))

    # {"clientId": "001", "params": null, "type": "ASK",
    #  "commonResult": {"code": 200, "data": null, "message": "处理成功", "command": "vd_request"}}
    def ask(self, type_id, ID):  # netty也是通过socket发送到网络中，netty发送的数据是ByteBuf结构的
        msg = {'clientId': ID, 'params': None, 'type': type_id, 'auth': 'c99d0b086eb17bb543867c1f95a0a088', 'commonResult': {'code': 200, 'data': None, 'message': '处理成功', 'command': 'vd_request'}}
        msg_json = json.dumps(msg)
        self.s.sendall(msg_json.encode('utf-8'))

    def send_data(self, type_id, ID, data):  # netty也是通过socket发送到网络中，netty发送的数据是ByteBuf结构的
        msg = {'clientId': ID, 'params': None, 'type': type_id, 'auth': 'c99d0b086eb17bb543867c1f95a0a088', 'commonResult': {'code': 200, 'data': data, 'message': '处理成功', 'command': 'rt_return'}}
        msg_json = json.dumps(msg)
        self.s.sendall(msg_json.encode('utf-8'))

    def rec(self, buff=4096*4):
        return self.s.recv(buff)

    def __del__(self):
        self.s.close()

if __name__ == "__main__":
    id = '001'
    # s.login(type_id="LOGIN", ID="001", UserName="hibox", Password="hibox_1028")  # 用string的话可以直接用encoder和decoder进行编解码
    import torch
    for i in range(10000):
        outpath = 'test0/0-0.mp4'
        state, result, result = run(outpath, verbose=False, logging=True)
        print(result)
