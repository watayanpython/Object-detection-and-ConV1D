## YOLOv5 と EasyOCR を使った OCR

YOLOv5オブジェクト検出アルゴリズムと、EasyOCR光学文字認識ライブラリを使用して、画像内のナンバープレートなどの文字列を抽出する方法を共有します。

### 必要なツール

- Python 3.6以上
- OpenCV
- YOLOv5
- EasyOCR

### インストール

#### OpenCV 
OpenCVをインストールするには、以下のコマンドを実行します。

```bash
pip install opencv-python
```

#### YOLOv5
YOLOv5をインストールするには、以下の手順に従ってください。

1. リポジトリをクローンする

```bash
git clone git@github.com:throo-io/car-number-recognition.git
```

2. 必要なライブラリをインストールする

```bash
pip install -r requirements.txt
pip install bs4 python-dotenv
```

3. .envファイルを作成

```bash
cp .env.sample .env
# 必要に応じて編集
vi .env
```

#### EasyOCR
EasyOCRをインストールするには、以下のコマンドを実行します。

```bash
pip install easyocr
```

### 使い方

1. yolov5_all.pyで画像からナンバープレートを検出し、切り抜き、カレントディレクトリにoutput_car_cropフォルダを作り保存します。次のコードを使用してください。

```python
import torch
from PIL import Image, ImageDraw, ImageFont
import os

# モード変換
def convert_to_rgb(image):
    if image.mode == 'RGB':
        return image
    else:
        return image.convert('RGB')
# 学習済みモデルの重みのpath    
model_path = './yolov5_watatani/car.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 推論対象のディレクトリ
input_dir = './yolov5_watatani/input'
# 矩形された画像を出力するディレクトリ
output_dir = './yolov5_watatani/output_car'
# 矩形で囲まれたオブジェクトのみを切り出して保存するディレクトリ
output_dir2 = './yolov5_watatani/output_car_crop'
# output_dir3 = '/Users/watatani/Desktop/yolov5/data/images/output_car_crop_label4'

# 出力先ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)
# os.makedirs(output_dir3, exist_ok=True)

# ディレクトリ内の画像ファイルに対してオブジェクトの検出を行う
for filename in os.listdir(input_dir):
    print(f'Processing file: {filename}')
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') :  # 拡張子が.jpgまたは.pngのファイルのみ処理
        # 画像ファイルのパスを作成
        image_path = os.path.join(input_dir, filename)
        
        # 画像を読み込む
        img = Image.open(image_path)

        img = convert_to_rgb(img)
        
        # 画像をモデルに入力し、オブジェクトを検出する
        results = model(img)
        
        # 矩形描画用に画像をコピーする
        draw_img = img.copy()

        print(f'Processing file: {filename}')
        highest_conf = 0
        highest_detection = None
        # 検出されたオブジェクトの中で確信度が最大なオブジェクトだけ矩形を描画する
        detected = False  # オブジェクトが検出されたかどうかのフラグ
        for detection in results.xyxy[0]:
            if detection[4] > highest_conf:
                highest_conf = detection[4]
                highest_detection = detection
            if highest_detection is not None: 
                detected = True
                x1, y1, x2, y2, conf, cls = highest_detection.tolist()
                # 中心座標と幅を計算し、正規化する
                x_center = (x1 + x2) / 2 / img.width
                y_center = (y1 + y2) / 2 / img.height
                width_norm = (x2 - x1) / img.width
                height_norm = (y2 - y1) / img.height
                draw = ImageDraw.Draw(draw_img)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
                # クラス名と確信度を表示する
                font = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', size=16)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                text_w, text_h = draw.textsize(label, font)
                draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill='red')
                draw.text((x1, y1 - text_h), label, fill='white', font=font)
                
                 # オブジェクトを切り取る
                cropped_img = img.crop((x1, y1, x2, y2))

                # 切り取った画像を保存する
                name, ext = os.path.splitext(filename)
                output_path2 = os.path.join(output_dir2, f'{name}.jpg')
                cropped_img.save(output_path2)

                # 座標とクラスラベルをtxtとして保存する
                # output_txt = os.path.join(output_dir3, f'{name}.txt')
                # with open(output_txt, 'w') as f:
                #     f.write(f'{int(cls)} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}')
        # オブジェクトが検出された画像を保存する
        if detected:
            output_path = os.path.join(output_dir, filename)
            draw_img.save(output_path)
```

2. easyocr_test.pyを用いて、切り抜かれたナンバープレートからテキストを読み取ります。次のコードを使用してください。

```python
import os
import cv2
import easyocr

# OCRエンジンを初期化する
reader = easyocr.Reader(['ja', 'en'], gpu=False)

# 処理する画像があるフォルダーのパスを取得する
img_folder = './yolov5_watatani/output_car_crop'

# 出力結果を保存するファイル
result_file = open('./yolov5_watatani/output.txt', 'w')

# 指定されたディレクトリ内の全てのファイルを処理する
for filename in os.listdir(img_folder):
    # ファイルパスを取得する
    img_path = os.path.join(img_folder, filename)
    
    # 画像を読み込む
    img = cv2.imread(img_path)
    
    # グレースケールに変換する
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二値化処理する
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  
    # 画像から文字列を認識する
    result = reader.readtext(binary)

    # 認識結果をファイルに記録する
    for i, bbox in enumerate(result):
        print(f"Image {filename}, Text {i}: {bbox[1]}\n")
        result_file.write(f"Image {filename}, Text {i}: {bbox[1]}\n")

# ファイルを閉じる
result_file.close()
```

### 注意事項

