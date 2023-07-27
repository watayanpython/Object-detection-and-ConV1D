# README

## **ラジコンカー物体検出データ収集アプリケーション**

### **セットアップと使い方**

### 必要なライブラリのインストール

アプリケーションを動作させるには、以下のライブラリが必要です。インストールされていない場合は、事前にインストールしてください。

- Flask
- OpenCV (cv2)
- base64
- numpy
- torch
- tensorflow
- scikit-learn
- keras

### モデルのダウンロード

YoloV5の物体検出モデルを使用しています。モデルをダウンロードするために、以下のコードを実行してください。

```python
pythonCopy code
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

```

### データ保存用のフォルダを作成

アプリケーションは画像を保存するためのフォルダを使用します。適切なフォルダを作成してください。

```python
pythonCopy code
image_dir = "./serial/serial_image"
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

```

### データの収集

アプリケーションは **`/save`** というAPIエンドポイントにPOSTリクエストを送信することでデータ収集を行います。以下はPythonのコード例です。

```python
pythonCopy code
import requests
import base64

# 画像を読み込んでbase64形式に変換
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 任意のCSVデータをカンマ区切りで文字列化
csv_data = "0.8,0.2,0.5,0.7,0.3"

# JSONデータを作成
data = {
    "image": image_data,
    "data": csv_data
}

# APIエンドポイントにPOSTリクエストを送信
response = requests.post("http://localhost:8070/save", json=data)

# レスポンスを表示
print(response.json())

```

リクエストには、base64エンコードされた画像データとCSVデータを含むJSON形式のデータが必要です。

### 処理結果の確認

アプリケーションは物体検出を行い、検出結果をコンソールに表示します。また、CSVファイルにもデータを追記します。物体検出結果に対しては、矩形を描画した画像も保存されます。

### **データの利用**

CSVファイルに記録されたデータは、物体検出の学習データとして利用することができます。

