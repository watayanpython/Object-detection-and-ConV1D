import os
import json
import cv2
import base64
import numpy as np
from flask import Flask, request, Response
import csv
import torch
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
import csv
import datetime

app = Flask(__name__)
count = 0
df = None

# 正規化用の変数
max_conf = 1.0
min_conf = 0.0
max_x = 640
min_x = 0
max_y = 480
min_y = 0

# モデルの読み込み
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
motor_prediction_model = tf.keras.models.load_model('C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_model5.h5')

# データ正規化のためのScalerを作成
scaler_X = MinMaxScaler()

# 画像を保存するフォルダの作成
image_dir = "./flask_test_images4"
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

def save_image(image):
    # データの変換処理
    image_dec = base64.b64decode(image)
    data_np = np.frombuffer(image_dec, dtype='uint8')
    decimg = cv2.imdecode(data_np, 1)

    # 画像ファイルを保存
    global count
    filename = ".//flask_test_images3/image{}.png".format(count)
    cv2.imwrite(filename, decimg)
    count += 1

    return filename

@app.route('/save', methods=['POST'])
def save():
    global df
    dt_now = datetime.datetime.now()
    print("start;", dt_now)
    # データの取得
    data = request.get_json()
    image = data['image']
    # csv_data = data['data']
    # CSVデータをセパレートしてリストに追加
    # csv_values = csv_data.split(',')
    # csv_values = [float(value) for value in csv_values]
    # print(csv_data)

    # 画像の保存
    filename = save_image(image)

    # CSVファイルのパス
    # csv_file_path = "./flask_test_images3/object_data/data1.csv"

    # YoloV5で物体検出を行う
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    results = object_detection_model(img)
    detections = []

    # CSVファイルが存在しない場合は作成する
    # if not os.path.exists(csv_file_path):
    #     # CSVファイルを書き込みモードでオープン
    #     with open(csv_file_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["confidence", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "left-motor", "right-motor"])  # ヘッダーを書き込む

    # bottleの中で最も確信度の高いオブジェクトを検出
    bottle_max_confidence = 0.0
    bottle_max_confidence_index = -1
    for i, detection in enumerate(results.xyxy[0]):
        if object_detection_model.names[int(detection[5])] == "bottle" and detection[4] > bottle_max_confidence:
            bottle_max_confidence = detection[4]
            bottle_max_confidence_index = i

    # personの中で最も確信度の高いオブジェクトを検出
    person_max_confidence = 0.0
    person_max_confidence_index = -1
    for i, detection in enumerate(results.xyxy[0]):
        if object_detection_model.names[int(detection[5])] == "person" and detection[4] > person_max_confidence:
            person_max_confidence = detection[4]
            person_max_confidence_index = i

    if bottle_max_confidence_index != -1:
        bottle_detection = results.xyxy[0][bottle_max_confidence_index]
        x1, y1, x2, y2, bottle_conf, bottle_cls = bottle_detection.tolist()

        # データを正規化
        scaled_conf = (bottle_conf - min_conf) / (max_conf - min_conf)
        scaled_x1 = (x1 - min_x) / (max_x - min_x)
        scaled_y1 = (y1 - min_y) / (max_y - min_y)
        scaled_x2 = (x2 - min_x) / (max_x - min_x)
        scaled_y2 = (y2 - min_y) / (max_y - min_y)
        normalized_data = [[scaled_conf, scaled_x1, scaled_y1, scaled_x2, scaled_y2]]

        # バウンディングボックスの座標と確信度を出力
        print(f'クラス: {object_detection_model.names[int(bottle_cls)]}, 確信度: {bottle_conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')
        # 予測モデルに入力して左モーターと右モーターの出力を予測
        # motor_output = motor_prediction_model.predict(normalized_data)
        # print(motor_output)

        # 矩形を描画するなどの処理を追加することもできます
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # クラス名と確信度を表示する
        label = f'{object_detection_model.names[int(bottle_cls)]} {bottle_conf:.2f}'
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        # bottleが検出されなかった場合は確信度と座標を0として出力
        bottle_conf, x1, y1, x2, y2 = 0, 0, 0, 0, 0
        print(f'クラス: None, 確信度: {bottle_conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')
        # データを正規化
        normalized_data = scaler_X.fit_transform([[bottle_conf, x1, y1, x2, y2]])
        # 予測モデルに入力して左モーターと右モーターの出力を予測
        # motor_output = motor_prediction_model.predict(normalized_data)
        # print(motor_output)

    if person_max_confidence_index != -1:
        person_detection = results.xyxy[0][person_max_confidence_index]
        person_x1, person_y1, person_x2, person_y2, person_conf, person_cls = person_detection.tolist()

        # データを正規化
        # scaled_conf = (person_conf - min_conf) / (max_conf - min_conf)
        person_scaled_x1 = (person_x1 - min_x) / (max_x - min_x)
        person_scaled_y1 = (person_y1 - min_y) / (max_y - min_y)
        person_scaled_x2 = (person_x2 - min_x) / (max_x - min_x)
        person_scaled_y2 = (person_y2 - min_y) / (max_y - min_y)
        person_normalized_data = [[person_scaled_x1, person_scaled_y1, person_scaled_x2, person_scaled_y2]]

        # バウンディングボックスの座標と確信度を出力
        print(f'クラス: {object_detection_model.names[int(person_cls)]}, 確信度: {person_conf:.2f}, 座標: ({int(person_x1)}, {int(person_y1)}), ({int(person_x2)}, {int(person_y2)})')
        # # 予測モデルに入力して左モーターと右モーターの出力を予測
        # motor_output = motor_prediction_model.predict(normalized_data)
        # print(motor_output)

        # 矩形を描画するなどの処理を追加することもできます
        cv2.rectangle(img, (int(person_x1), int(person_y1)), (int(person_x2), int(person_y2)), (0, 255, 0), 2)
        # クラス名と確信度を表示する
        label = f'{object_detection_model.names[int(person_cls)]} {person_conf:.2f}'
        cv2.putText(img, label, (int(person_x1), int(person_y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        # personが検出されなかった場合は確信度と座標を0として出力
        person_conf, person_x1, person_y1, person_x2, person_y2 = 0, 0, 0, 0, 0
        print(f'クラス: None, 確信度: {person_conf:.2f}, 座標: ({int(person_x1)}, {int(person_y1)}), ({int(person_x2)}, {int(person_y2)})')
        person_normalized_data = scaler_X.fit_transform([[person_x1, person_y1, person_x2, person_y2]])

    # # CSVデータを一時的に保持
    # if df is None:
    #     df = csv_data
    # else:
    #     df += csv_data

    # データをCSVファイルに追加
    # with open(csv_file_path, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([bottle_conf, x1, y1, x2, y2, person_x1, person_y1, person_x2, person_y2, csv_values[0], csv_values[1]])

    # 画像を表示する
    # cv2.imshow('YoloV5 Object Detection', img)
    # cv2.waitKey(1)
    # 入力データを統合
    input_data = [scaled_conf, scaled_x1, scaled_y1, scaled_x2, scaled_y2, person_scaled_x1, person_scaled_y1, person_scaled_x2, person_scaled_y2]
    print(input_data)
    motor_output = motor_prediction_model.predict(input_data)
    print(motor_output)
    # モーターの出力をJSON形式で返す
    motor_output_json = json.dumps({
            'left-motor': motor_output[0],
            'right-motor': motor_output[1]
        })

    dt_now1 = datetime.datetime.now()
    print("end:", dt_now1)
    # HTTPレスポンスを送信
    return Response(response=json.dumps({"motor_output": motor_output_json}), status=200)

    # dt_now1 = datetime.datetime.now()
    # print("end:", dt_now1)
    # # HTTPレスポンスを送信
    # return Response(response=json.dumps({"message": "Image and CSV data saved."}), status=200)

@app.teardown_appcontext
def teardown_appcontext(exception=None):
    global df

    # # レスポンスが終了したらCSVファイルに出力
    # if df is not None:
    #     csv_filename = "./flask_test_images3/data.csv"
    #     with open(csv_filename, 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerows(df)
    #     df = None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8070, threaded=True)
