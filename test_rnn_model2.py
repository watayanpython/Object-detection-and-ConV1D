import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ディレクトリパス
directory = './motor_data2'

# モデルの読み込み
model = tf.keras.models.load_model('C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_model2.h5')

# 各CSVファイルの処理
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        
        # CSVファイルを読み込む
        df = pd.read_csv(file_path)
        
        # データの前処理
        scaler_X = MinMaxScaler()
        df_scaled = scaler_X.fit_transform(df)
        
        # 1行ごとにデータを入力して予測
        for i in range(len(df_scaled)):
            input_data = np.expand_dims(df_scaled[i], axis=0)
            output = model.predict(input_data)
            
            print("File:", file)
            print("Input:", df.iloc[i])
            print("Output:", output[0])
            print()
