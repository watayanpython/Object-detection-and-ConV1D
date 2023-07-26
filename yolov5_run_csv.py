import torch
import cv2
import json

# モデルを読み込む
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# カメラのキャプチャを開始する
cap = cv2.VideoCapture(0)

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()

    # フレームが正常に読み込めた場合だけ処理を実行する
    if ret:
        # 入力画像をyolov5用に変換する
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 画像をモデルに入力し、オブジェクトを検出する
        results = model(img)

        # 検出されたオブジェクトの数だけ矩形を描画する
        for detection in results.xyxy[0]:
            if detection[4] > 0.7 and model.names[int(detection[5])] == "person":  # 確信度が0.7以上でクラスが"person"の場合のみ処理を行う: 
                x1, y1, x2, y2, conf, cls = detection.tolist()

                # バウンディングボックスの座標を出力する
                print(f'クラス: {model.names[int(cls)]}, 確信度: {conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')

                # 矩形を描画する
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # クラス名と確信度を表示する
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 画像を表示する
        cv2.imshow('YoloV5 Object Detection', frame)

        # 検出されたオブジェクトの情報をJSON形式で出力する
        # objects = []
        # for detection in results.xyxy[0]:
        #     if detection[4] > 0.7 and model.names[int(detection[5])] == "person":  # 確信度が0.7以上でクラスが"person"の場合のみ処理を行う:  # 確信度の設定
        #         x1, y1, x2, y2, conf, cls = detection.tolist()
        #         obj = {
        #                 'detection':[
        #                     {
        #                         'class': model.names[int(cls)],
        #                         'confidence': conf,
        #                         'x1': int(x1),
        #                         'y1': int(y1),
        #                         'x2': int(x2), 
        #                         'y2': int(y2)
        #                     }
        #                 ]
        #         }
        #         objects.append(obj)

        # # JSONファイルとして出力
        # with open('./output.json', 'w') as file:
        #     json.dump(objects, file, indent=4)

        # print(json.dumps(objects, indent=4))

        # 「q」キーを押すとループから抜け出す
        if cv2.waitKey(1) == ord('q'):
            break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()


