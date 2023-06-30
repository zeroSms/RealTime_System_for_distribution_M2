#
# データ処理スレッド
#
import time
import numpy as np
import collections
import pandas as pd
import socket
import pickle
import pyautogui

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 自作ライブラリ
from . import add_data, get_feature, setup_variable, stop
from paz.backend import camera as CML

analysis_csv = [setup_variable.analysis_columns]  # windowデータの追加
answer_list = []  # 正解データリスト（windowごと）
feature_list = []  # 特徴量抽出データ
window_num = 0  # window番号
realtime_pred = []

# ================================= パスの取得 ================================ #
path = setup_variable.path
server_address = '192.168.2.111'


# ============================ ラベル整形スレッド ============================== #
# ウィンドウラベルの付与，正解ラベルデータの作成
def label_shape(window):
    window_T = np.array(window).T  # 転置　（labelごとの個数を計算するため）
    label_num = collections.Counter(window_T[0])  # labelごとの個数を計算
    threshold = setup_variable.threshold

    # 正解データファイルの出力
    if label_num['nod'] > int(len(window) * threshold):
        answer_num = 1
    elif label_num['shake'] > int(len(window) * threshold):
        answer_num = 2
    else:
        answer_num = 0
    window_T[0] = [window_num] * len(window)  # window_IDの追加

    return window_T.T, answer_num

# 表情の文字列を記号に変換
def face_symbol(pred_face):
    face_dict = {
        'neutral'   : 'a',
        'happy'     : 'b',
        'surprise'  : 'c',
        'sad'       : 'd',
        'angry'     : 'e',
        'fear'      : 'f',
        'disgust'   : 'g'
    }
    return face_dict[pred_face]


# ============================ データ処理スレッド ============================== #
# 測定したデータを処理する関数
def ProcessData():
    global window_num
    while stop.stop_flg:
        window = add_data.sensor.process_window()
        if window:
            window_num += 1
            analysis_csv.extend(label_shape(window))


# 測定したデータを処理する関数
push_server = []


def Realtime_analysis(to_server=False, get_face=False):
    global window_num, feature_list, push_server

    # 特徴量リスト
    clf = pickle.load(open(path + '\\data_set\\analysis_files\\102\\trained_model.pkl', 'rb'))
    sensor_name = 'gyro'
    get_feature.feature_name(sensor_name)
    filename = path + '\\data_set\\analysis_files\\102\\feature_list_selection.csv'
    selection_X = pd.read_csv(filename)

    if to_server:
        host = server_address  # サーバーのホスト名
        client_address = socket.gethostname()  # クライアント側のホスト名
        port = setup_variable.port  # 49152~65535

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成をします
        client.connect((host, port))  # これでサーバーに接続します

        response = {'timeStamp': time.time(),
                    'class': 'Head',
                    'User': socket.gethostbyname(client_address)
                    }

    while stop.stop_flg:
        window = []
        window = add_data.sensor.process_window()
        if window:
            window_num += 1

            # ウィンドウラベルの付与，正解ラベルデータの作成
            result_window, answer_num = label_shape(window)
            answer_list.append(answer_num)
            analysis_csv.extend(result_window)

            # リアルタイム行動分析-特徴量抽出
            feature_list.append(get_feature.get_feature(window, sensor_name))
            test_x = pd.DataFrame(feature_list, columns=get_feature.feature_columns)
            selection_X_columns = selection_X.columns[1:].tolist()  # 特徴量選択後のカラムを取得
            test_x = test_x.loc[:, selection_X_columns]
            test_x = test_x.values

            y_pred = clf.predict(test_x)    # 頭の動きを判定
            print(y_pred, answer_num)       # 判定された行動の出力
            realtime_pred.extend(y_pred)    # 予測結果を追加
            feature_list = []               # 該当ウィンドウの特徴量リスト初期化

            # 判定された表情の出力
            if get_face:
                pred_face = CML.process_window()
                print(pred_face)
            # else:
            #     # 表情の判定がない場合，頭の動きを可視化(main_Realtime.py)
            #     pyautogui.press(str(y_pred[0]))

            # サーバーへの送信
            if to_server:
                response['head_action'] = y_pred[0]
                if get_face:
                    response['face_action'] = face_symbol(pred_face)
                massage = pickle.dumps(response)
                client.send(massage)  # データを送信

    print(realtime_pred)
    print(answer_list)
    test_y = pd.Series(data=answer_list)
    y_pred = pd.Series(data=realtime_pred)
    print(classification_report(test_y, y_pred, target_names=['others', 'nod', 'shake']))
