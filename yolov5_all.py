import torch
from PIL import Image, ImageDraw, ImageFont
import os

def convert_to_rgb(image):
    if image.mode == 'RGB':
        return image
    else:
        return image.convert('RGB')
    
model_path = './yolov5x.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 推論対象のディレクトリと出力先ディレクトリの指定
input_dir = './data/images/input_test'
output_dir = './data/images/output'
output_dir2 = './data/images/output_crop'
output_dir3 = './data/images/output_crop2'

# 出力先ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)
os.makedirs(output_dir3, exist_ok=True)

# ディレクトリ内の画像ファイルに対してオブジェクトの検出を行う
for filename in os.listdir(input_dir):
    print(f'Processing file: {filename}')
    if filename.endswith('.JPG') or filename.endswith('.JPEG'):  # 拡張子が.jpgまたは.pngのファイルのみ処理
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
        # 検出されたオブジェクトの数だけ矩形を描画する
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
                font = ImageFont.truetype('arial.ttf', size=16)
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
                output_txt = os.path.join(output_dir3, f'{name}.txt')
                with open(output_txt, 'w') as f:
                    f.write(f'{int(cls)} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}')
        # オブジェクトが検出された画像を保存する
        if detected:
            output_path = os.path.join(output_dir, filename)
            draw_img.save(output_path)
                
        
# print(f'Saved output image to {output_path}')
# print(f'Saved cropped image to {output_path2}')

