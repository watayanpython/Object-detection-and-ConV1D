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
motor_prediction_model = tf.keras.models.load_model('C:./motor_model2.h5')

# データ正規化のためのScalerを作成
scaler_X = MinMaxScaler()

# 画像を保存するフォルダの作成
image_dir = "./detect_images5"
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

def save_image(image):
    # データの変換処理
    image_dec = base64.b64decode(image)
    data_np = np.frombuffer(image_dec, dtype='uint8')
    decimg = cv2.imdecode(data_np, 1)

    # 画像ファイルを保存
    global count
    filename = ".//detect_images5/image{}.png".format(count)
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
    # 画像を物体検出モデルに入力し、オブジェクトを検出する
    results = object_detection_model(img)
    detections = []

        # CSVファイルが存在しない場合は作成する
        # if not os.path.exists(csv_file_path):
        #     # CSVファイルを書き込みモードでオープン
        #     with open(csv_file_path, mode='w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(["confidence", "x1", "y1", "x2", "y2"])  # ヘッダーを書き込む
    motor_output_json = None
        # 検出されたオブジェクトの数だけ矩形を描画する
    if len(results.xyxy[0]) > 0:
        found_bottle = False
        for detection in results.xyxy[0]:
            if detection[4] > 0.5 and object_detection_model.names[int(detection[5])] == "bottle":  # 確信度が0.7以上でクラスが"person"の場合のみ処理を行う:  # 確信度が0.9以上の場合のみ処理を行う
                    x1, y1, x2, y2, conf, cls = detection.tolist()

                    # バウンディングボックスの座標を出力する
                    print(f'クラス: {object_detection_model.names[int(cls)]}, 確信度: {conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')
                    # 矩形を描画する
                    # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # CSVファイルに書き込む
                    # CSVファイルを書き込みモードでオープン
                    # with open(csv_file_path, mode='a', newline='') as file:
                    #     writer = csv.writer(file)
                    #     writer.writerow([conf, int(x1), int(y1), int(x2), int(y2)])

                    # データを正規化
                    scaled_conf = (conf - min_conf) / (max_conf - min_conf)
                    scaled_x1 = (x1 - min_x) / (max_x - min_x)
                    scaled_y1 = (y1 - min_y) / (max_y - min_y)
                    scaled_x2 = (x2 - min_x) / (max_x - min_x)
                    scaled_y2 = (y2 - min_y) / (max_y - min_y)
                    normalized_data = [[scaled_conf, scaled_x1, scaled_y1, scaled_x2, scaled_y2]]

                    # 予測モデルに入力して左モーターと右モーターの出力を予測
                    motor_output = motor_prediction_model.predict(normalized_data)
                    motor_output = np.squeeze(motor_output)
                    print(motor_output)

                    # モーターの出力をJSON形式で返す
                    motor_output_json = json.dumps({
                            'left-motor': str(motor_output[0]),
                            'right-motor': str(motor_output[1])
                        })
                    # クラス名と確信度を表示する
                    label = f'{object_detection_model.names[int(cls)]} {conf:.2f}'
                    # cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    found_bottle = True
                
            if not found_bottle:
                # bottleが検出されなかった場合はconfidenceと座標を0と出力
                conf, x1, y1, x2, y2 = 0, 0, 0, 0, 0
                print(f'クラス: None, 確信度: {conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')

                # CSVファイルに書き込む
                # with open(csv_file_path, mode='a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([conf, int(x1), int(y1), int(x2), int(y2)])

                # データを正規化
                normalized_data = scaler_X.fit_transform([[conf, x1, y1, x2, y2]])
                # 予測モデルに入力して左モーターと右モーターの出力を予測
                motor_output = motor_prediction_model.predict(normalized_data)
                motor_output = np.squeeze(motor_output)
                print(motor_output)

                # モーターの出力をJSON形式で返す
                motor_output_json = json.dumps({
                        'left-motor': str(motor_output[0]),
                        'right-motor': str(motor_output[1])
                    })

    dt_now1 = datetime.datetime.now()
    print("end:", dt_now1)
    # HTTPレスポンスを送信
    return Response(response=motor_output_json, status=200)

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
