import cv2
import json
import torch
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

max_conf = 1.0
min_conf = 0.0
max_x = 720
min_x = 0
max_y = 720
min_y = 0

# モデルの読み込み
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
motor_prediction_model = tf.keras.models.load_model('C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_model2.h5')

# カメラのキャプチャを開始する
cap = cv2.VideoCapture(0)

# データ正規化のためのScalerを作成
scaler_X = MinMaxScaler()

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()

    # フレームが正常に読み込めた場合だけ処理を実行する
    if ret:
        # 入力画像をyolov5用に変換する
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 画像を物体検出モデルに入力し、オブジェクトを検出する
        results = object_detection_model(img)
        detections = []
        # 検出されたオブジェクトの数だけ矩形を描画する
        if len(results.xyxy[0]) > 0:
            found_bottle = False
            for detection in results.xyxy[0]:
                if detection[4] > 0.5 and object_detection_model.names[int(detection[5])] == "bottle":  # 確信度が0.7以上でクラスが"person"の場合のみ処理を行う:  # 確信度が0.9以上の場合のみ処理を行う
                    x1, y1, x2, y2, conf, cls = detection.tolist()

                    # バウンディングボックスの座標を出力する
                    print(f'クラス: {object_detection_model.names[int(cls)]}, 確信度: {conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')
                    # 矩形を描画する
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # データを正規化
                    scaled_conf = (conf - min_conf) / (max_conf - min_conf)
                    scaled_x1 = (x1 - min_x) / (max_x - min_x)
                    scaled_y1 = (y1 - min_y) / (max_y - min_y)
                    scaled_x2 = (x2 - min_x) / (max_x - min_x)
                    scaled_y2 = (y2 - min_y) / (max_y - min_y)
                    normalized_data = [[scaled_conf, scaled_x1, scaled_y1, scaled_x2, scaled_y2]]
                    # 予測モデルに入力して左モーターと右モーターの出力を予測
                    motor_output = motor_prediction_model.predict(normalized_data)
                    print(motor_output)
                    # クラス名と確信度を表示する
                    label = f'{object_detection_model.names[int(cls)]} {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    found_bottle = True
                
            if not found_bottle:
                # bottleが検出されなかった場合はconfidenceと座標を0と出力
                conf, x1, y1, x2, y2 = 0, 0, 0, 0, 0
                print(f'クラス: None, 確信度: {conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')
                # データを正規化
                normalized_data = scaler_X.fit_transform([[conf, x1, y1, x2, y2]])
                # 予測モデルに入力して左モーターと右モーターの出力を予測
                motor_output = motor_prediction_model.predict(normalized_data)
                print(motor_output)
    else:
        break

            # 予測結果を辞書形式で作成
        result = {
                        # "detection": [
                        #     {
                        #         # "class": object_detection_model.names[int(cls)],
                        #         "confidence": conf,
                        #         "x1": int(x1),
                        #         "y1": int(y1),
                        #         "x2": int(x2),
                        #         "y2": int(y2)
                        #     }
                        # ],
                        "motor_output": [
                            {
                                "left_motor": float(motor_output[0][0]),
                                "right_motor": float(motor_output[0][1])
                            }
                        ]
                    }
        detections.append(result)

            # # JSONファイルとして出力
        with open("output_test.json", "w") as file:
                json.dump(detections, file, indent=4)

        # 画像を表示する
        cv2.imshow('YoloV5 Object Detection', frame)

        # 「q」キーを押すとループから抜け出す
        if cv2.waitKey(1) == ord('q'):
            break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
