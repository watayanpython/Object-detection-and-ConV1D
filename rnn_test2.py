import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# データの読み込みと前処理
data_dir = 'C:/Users/watat/OneDrive - Kyushu University/yolov5/flask_test_images3/object_data'
data_files = os.listdir(data_dir)

# 学習データの作成
X_train = []
y_train = []

for file in data_files[:-1]:  # 最後のファイルをテストデータとする
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    
    X_train.append(df[['confidence', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].values)
    y_train.append(df[['left-motor', 'right-motor']].values)
    
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# テストデータの作成
file_path = os.path.join(data_dir, data_files[-1])
df_test = pd.read_csv(file_path)
X_test = df_test[['confidence', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].values
y_test = df_test[['left-motor', 'right-motor']].values

# データの正規化
scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
# y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
# y_test_scaled = scaler_y.transform(y_test)

print(X_train.shape)

# モデル設計
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((9, 1), input_shape=(9,)))
model.add(tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'))
model.add(tf.keras.layers.Conv1D(512, 3, activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling1D(2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='linear'))
model.summary()

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルの学習
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=2, verbose=1, validation_split=0.1)

# 損失の推移を取得
loss = history.history['loss']
val_loss = history.history['val_loss']

# 損失の推移をグラフで表示
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# テストデータでの予測
# y_pred_scaled = model.predict(X_test_scaled)
y_pred = model.predict(X_test_scaled)

# 予測結果の逆正規化
# y_pred = scaler_y.inverse_transform(y_pred_scaled)
# y_test = scaler_y.inverse_transform(y_test_scaled)

print(y_test)
print(y_pred.shape)

# MAEを計算
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# 予測結果の図を描画
time = np.arange(1, 8)

plt.figure(figsize=(10, 6))
plt.plot(time, y_test[:, 0], label='Ground Truth')
plt.plot(time, y_pred[:, 0], label='Prediction')
plt.title('Comparison of Output 1')
plt.xlabel('Times[sec]')
plt.ylabel('left-motor Output')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, y_test[:, 1], label='Ground Truth')
plt.plot(time, y_pred[:, 1], label='Prediction')
plt.title('Comparison of Output 2')
plt.xlabel('Times[sec]')
plt.ylabel('right-motor Output')
plt.legend()
plt.show()

# モデルの保存
model.save('./motor_model5.h5')
print("Model saved successfully.")
