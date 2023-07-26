import os
import torch
import cv2
import json
import requests
import base64
import numpy as np
import datetime
from flask import Flask, request, Response
app = Flask(__name__)
count = 0

# # モデルを読み込む
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # 画像を保存するフォルダのパス
# image_dir = "./flask_test_images"
# if not os.path.isdir(image_dir):
#     os.mkdir(image_dir)

# def process_image_from_url(image_url):
#     # 画像のダウンロード
#     response = requests.get(image_url)
#     image_data = response.content

#     # 画像データをNumPy配列に変換
#     nparr = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # 画像を一時的に保存
#     temp_image_path = os.path.join(image_dir, "temp_image.png")
#     cv2.imwrite(temp_image_path, img)

#     # 画像の処理
#     process_image(temp_image_path)

#     # 一時的な画像ファイルを削除
#     os.remove(temp_image_path)

# def process_image(image_path):
#     # 画像の読み込み
#     img = cv2.imread(image_path)

#     # 入力画像をyolov5用に変換する
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 画像をモデルに入力し、オブジェクトを検出する
#     results = model(img)

#     # 検出されたオブジェクトの数だけ矩形を描画する
#     for detection in results.xyxy[0]:
#         if detection[4] > 0.7:  # 確信度が0.7以上の場合のみ処理を行う
#             x1, y1, x2, y2, conf, cls = detection.tolist()

#             # 矩形を描画する
#             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

#             # クラス名と確信度を表示する
#             label = f'{model.names[int(cls)]} {conf:.2f}'
#             cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     # 画像を保存
#     output_path = os.path.join(image_dir, f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
#     cv2.imwrite(output_path, img)

#     # 検出されたオブジェクトの情報をJSON形式で出力する
#     objects = []
#     for detection in results.xyxy[0]:
#         if detection[4] > 0.7:  # 確信度の設定
#             x1, y1, x2, y2, conf, cls = detection.tolist()
#             obj = {
#                 'detection': [
#                     {
#                         'class': model.names[int(cls)],
#                         'confidence': conf,
#                         'x1': int(x1),
#                         'y1': int(y1),
#                         'x2': int(x2),
#                         'y2': int(y2)
#                     }
#                 ]
#             }
#             objects.append(obj)

#     # JSONファイルとして出力
#     with open('./output.json', 'w') as file:
#         json.dump(objects, file, indent=4)

#     print(json.dumps(objects, indent=4))

# HTTPリクエストを受け付けるエンドポイント
@app.route('/save', methods=['POST'])
def process_image_endpoint():
    # リクエストのデータを取得
    data = request.get_json()
    image_url = data['image_url']

    # 画像の処理
    # process_image_from_url(image_url)

    # HTTPレスポンスを返す
    return Response(response=json.dumps({"message": "Image processed successfully."}), status=200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
