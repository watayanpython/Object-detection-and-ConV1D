import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# ディレクトリパス
directory = './motor_data2'
output_dir = './motor_data2/motor_plot'

# モデルの読み込み
model = tf.keras.models.load_model('C:/Users/watat/OneDrive - Kyushu University/yolov5/motor_model2.h5')

# 正規化用のスケーラー
scaler_X = MinMaxScaler()

# フォントの設定
plt.rcParams['font.family'] = 'Times New Roman'

# 各CSVファイルの処理
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        
        # CSVファイルを読み込む
        df = pd.read_csv(file_path)
        
        # 必要な要素の抽出
        X_test = df[['confidence', 'x1', 'y1', 'x2', 'y2']].values
        # データの正規化
        X_test_scaled = scaler_X.fit_transform(X_test)

        y_test = df[['left-motor', 'right-motor']].values
        
        # 予測
        y_pred = model.predict(X_test_scaled)
        
        # 線のスタイルと色を設定
        actual_line_style = 'solid'
        actual_line_color = 'black'
        predicted_line_style = 'dashed'
        predicted_line_color = 'blue'

        # 左右のモーターの予測結果を比較してプロット
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(y_test[:, 0], label='Actual', linestyle=actual_line_style, color=actual_line_color)
        plt.plot(y_pred[:, 0], label='Predicted', linestyle=predicted_line_style, color=predicted_line_color)
        plt.title(f'Left Motor - Actual vs Predicted (MAE: {mean_absolute_error(y_test[:, 0], y_pred[:, 0]):.4f})')
        plt.xlabel('Time[sec]')
        plt.ylabel('left-Motor Value')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(y_test[:, 1], label='Actual', linestyle=actual_line_style, color=actual_line_color)
        plt.plot(y_pred[:, 1], label='Predicted', linestyle=predicted_line_style, color=predicted_line_color)
        plt.title(f'Right Motor - Actual vs Predicted (MAE: {mean_absolute_error(y_test[:, 1], y_pred[:, 1]):.4f})')
        plt.xlabel('Time[sec]')
        plt.ylabel('right-Motor Value')
        plt.legend()
        
        plt.tight_layout()

        # プロットを保存
        output_path = os.path.join(output_dir, f'{file[:-4]}.png')
        plt.savefig(output_path)

        # plt.show()
