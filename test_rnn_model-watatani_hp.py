import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json

# CSVファイルのパス
csv_file = "motor_data2/data_1.csv"

# CSVファイルを読み込む
df = pd.read_csv(csv_file)

# 必要な要素の抽出
X_test = df[['confidence', 'x1', 'y1', 'x2', 'y2']].values

# データの正規化
scaler_X = MinMaxScaler()
X_test_scaled = scaler_X.fit_transform(X_test)

# モデルの読み込み
model = tf.keras.models.load_model('C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_model2.h5')

# 予測結果を格納するリスト
outputs = []

# 1ステップごとにデータを入力して予測
for i in range(len(X_test_scaled)):
    input_data = np.expand_dims(X_test_scaled[i], axis=0)
    output = model.predict(input_data)
    
    # 予測結果を辞書形式で作成
    result = {
        "output-motor": [
            {
                "left-motor": float(output[0][0]),
                "right-motor": float(output[0][1])
            }
        ]
    }
    outputs.append(result)

# JSONファイルとして出力
with open("output-motor.json", "w") as file:
    json.dump(outputs, file, indent=4)
