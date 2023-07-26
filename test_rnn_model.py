import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# データの前処理
data = {
    'confidence': [0.2, 0.6, 0.9, 0.7],
    'x1': [3, 200, 150, 180],
    'y1': [1, 80, 60, 70],
    'x2': [150, 250, 280, 320],
    'y2': [180, 180, 210, 190]
}

df = pd.DataFrame(data)

# データの正規化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
df_scaled = scaler_X.fit_transform(df)

# モデルの読み込み
model = tf.keras.models.load_model('C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_model3.h5')

# 1秒ごとにデータを入力して予測
for i in range(len(df_scaled)):
    input_data = np.expand_dims(df_scaled[i], axis=0)
    output = model.predict(input_data)
    # output = scaled_y.inverse_transform(output)
    print("Input:", df.iloc[i])
    print("Output:", output[0])
    print()
