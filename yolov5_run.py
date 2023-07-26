import torch
from PIL import Image, ImageDraw, ImageFont
import os
model_path = 'C:/Users/watat/Documents/yolov5/runs/train/exp5/weights/yolov5x.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 推論用の画像を読み込む
img = Image.open('C:/Users/watat/Documents/yolov5/data/images/input_test/image3.jpg')

# 画像をモデルに入力し、オブジェクトを検出する
results = model(img)

# 検出されたオブジェクトとその確信度を表示する
print(results.pandas().xyxy[0])

# 矩形描画用に画像をコピーする
draw_img = img.copy()

# 検出されたオブジェクトの数だけ矩形を描画する
for detection in results.xyxy[0]:
    if detection[0] > 0.8:  # 確信度が0.9以上の場合のみ描画
        x1, y1, x2, y2, conf, cls = detection.tolist()
        draw = ImageDraw.Draw(draw_img)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    # クラス名と確信度を表示する
        font = ImageFont.truetype('arial.ttf', size=16)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        text_w, text_h = draw.textsize(label, font)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill='red')
        draw.text((x1, y1 - text_h), label, fill='white', font=font)
    output_dir = 'C:/Users/watat/Documents/yolov5/data/images/output3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 検出されたオブジェクトの数だけ処理を繰り返す
for i, detection in enumerate(results.xyxy[0]):
    if detection[0] > 0.8:  # 確信度が0.9以上の場合のみ処理
        x1, y1, x2, y2, conf, cls = detection.tolist()

        # オブジェクトを切り取る
        cropped_img = img.crop((x1, y1, x2, y2))

        # 切り取った画像を保存する
        output_path = os.path.join(output_dir, f'crop_{i}.jpg')
        cropped_img.save(output_path)

        print(f'Saved cropped image to {output_path}')
# 描画された画像を表示する
draw_img.show()

