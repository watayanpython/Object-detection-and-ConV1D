import cv2
import json
import torch

# モデルを読み込む
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# 動画ファイルのパス
video_path = "C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_data2/20230708_065300000_iOS.MOV"

# 動画の読み込み
cap = cv2.VideoCapture(video_path)

# 動画ファイルの情報を取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'mp4v')  # 出力動画のコーデックを指定

# 出力動画ファイルの設定
output_path = "C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_data2/movie_plot/detectionl6.MOV"
out = cv2.VideoWriter(output_path, codec, fps, (width, height))

while True:
    # 動画からフレームを読み込む
    ret, frame = cap.read()

    # フレームが正常に読み込めた場合だけ処理を実行する
    if ret:
        # 入力画像をyolov5用に変換する
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 画像をモデルに入力し、オブジェクトを検出する
        results = model(img)

        # 検出されたオブジェクトの数だけ矩形を描画する
        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                if detection[4] > 0.7 :  # 確信度が0.7以上でクラスが"person"の場合のみ処理を行う:  # 確信度が0.9以上の場合のみ処理を行う
                    x1, y1, x2, y2, conf, cls = detection.tolist()

                    # バウンディングボックスの座標を出力する
                    print(f'クラス: {model.names[int(cls)]}, 確信度: {conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')

                    # 矩形を描画する
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # クラス名と確信度を表示する
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            # 検出できなかった場合はconfidenceと座標を0と出力
            conf, x1, y1, x2, y2 = 0, 0, 0, 0, 0
            print(f'クラス: None, 確信度: {conf:.2f}, 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})')

        # 画像を表示する
        cv2.imshow('YoloV5 Object Detection', frame)

        # 出力動画にフレームを書き込む
        out.write(frame)

        # 「q」キーを押すとループから抜け出す
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
out.release()
