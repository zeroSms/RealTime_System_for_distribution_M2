#
# 表情の決定と通信処理
#
import sys
import time
import socket
import pickle

# 自作ライブラリ
from . import add_data, get_feature, setup_variable, stop
from paz.backend import camera as CML

# ================================= パスの取得 ================================ #
path = setup_variable.path
server_address = setup_variable.server_address


# ================================= 表情の決定・通信 ================================ #
def client_face(to_server=True, port_select='1', audience_num=0):
    if to_server:
        host = server_address  # サーバーのホスト名
        client_address = socket.gethostname()  # クライアント側のホスト名
        port = setup_variable.port_num[port_select]['audience']

        client = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
        client.connect((host, port))  # これでサーバーに接続します

        response = {'presenter': False,
                    'ID': audience_num,
                    'timeStamp': time.time(),
                    'class': 'Face',
                    }

    while stop.stop_flg:
        time.sleep(1)
        timeStamp = time.time()

        # 判定された表情の出力
        pred_face = CML.process_window()
        # print(timeStamp, pred_face)

        # サーバーへの送信
        if to_server:
            response['timeStamp'] = round(timeStamp, 2)
            # response['action'] = setup_variable.face_symbol(pred_face[0])
            response['action'] = pred_face

            try:
                message = pickle.dumps(response)
                client.sendall(message)  # データを送信
            except Exception as e:
                print(e)
                continue

            # try:
            #     client.settimeout(2)
            #     recv_msg = client.recv(4096)  # レシーブは適当な2の累乗にします（大きすぎるとダメ）
            #     print(recv_msg.decode().replace('b', ''))
            # except Exception as e:
            #     print(e)
            #     continue
