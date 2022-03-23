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
    print('init model')
    CD = CabinetDetector()
    host = 'devapi.noguard.shop'
    port = 9090
    s = SocketService(host, port)
    id = '001'
    socket.setdefaulttimeout(60)
    # s.login(type_id="LOGIN", ID="001", UserName="hibox", Password="hibox_1028")  # 用string的话可以直接用encoder和decoder进行编解码
    for i in range(10000):
        # s.ask(type_id="ASK", ID="{:0<3}".format(i))
        s.ask(type_id="ASK", ID=id)
        Event_Url = json.loads(s.rec().decode())
        process_list = Event_Url['body']['serverInfo']['data']
        count = 0
        print('cnt {}'.format(i))
        if process_list:
            for j, single_msg in enumerate(process_list):
                event_id = single_msg['dooreventid']
                url = single_msg['ossurl']
                # download video
                outpath = 'video/{}-{}.mp4'.format(i, j)
                logpath = 'logs/{}.txt'.format(os.path.basename(url))
                print(outpath)
                try:
                    urllib.request.urlretrieve(url, outpath)
                # print(url)
                except socket.timeout:
                    print('error {}'.format(url))
                    data = {'errorCode': 0, 'errorMessage': 'FAILED', 'eventId': event_id, 'data': [{'goodId': 30, 'goodNum': 0}]}
                    s.send_data(type_id="ASK", ID=id, data=data)
                    print(data)
                    print(s.rec().decode())
                    continue
                result_ret = list()
                state, result, result_dict = CD.run(outpath, logpath, verbose=False, logging=True)
                for key, value in result_dict.items():
                    result_ret.append({'goodId': key, 'goodNum': value})
                if state:
                    data = {'errorCode': 0, 'errorMessage': 'SUCCESS', 'eventId': event_id,
                            'data': result_ret}
                else:
                    data = {'errorCode': 0, 'errorMessage': 'FAILED', 'eventId': event_id,
                            'data': result_ret}
                    print('a long video')
                s.send_data(type_id="ASK", ID=id, data=data)
                print(data)
                print(s.rec().decode())
                count += 1
                print(count)
        else:
            data = {'errorCode': 0, 'errorMessage': 'FAILED', 'eventId': 0, 'data': [{'goodId': 30, 'goodNum': 0}]}
            s.send_data(type_id="ASK", ID=id, data=data)
            print(data)
            print(s.rec().decode())
    del s
