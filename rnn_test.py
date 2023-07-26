import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ランダムな学習データ生成
np.random.seed(1000)
n_samples_train = 100
n_samples_test = 100

X_train = np.random.rand(n_samples_train, 5)  # 5次元の学習データ
y_train = 100*np.random.rand(n_samples_train, 2)  # 2次元の学習データ

X_test = np.random.rand(n_samples_test, 5)  # 5次元のテストデータ
y_test = 100*np.random.rand(n_samples_test, 2)  # 2次元のテストデータ

print(X_train)

# データの正規化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# モデル設計
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Reshape((5, 1), input_shape=(5,)))
model.add(tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'))
model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True))
model.add(tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='linear'))  # 出力層のユニット数は2 (左右のモータの入力)

model.summary()

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルの学習
history = model.fit(X_train_scaled, y_train_scaled, epochs=300, batch_size=4, verbose=1)

# 損失の推移を取得
loss = history.history['loss']

# 損失の推移をグラフで表示
plt.plot(loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# テストデータでの予測
y_pred_scaled = model.predict(X_test_scaled)

# 予測結果の逆正規化
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled)

print(y_test)
print(y_pred)

# MAEを計算
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# 予測結果の図を描画
time = np.arange(n_samples_test)

plt.figure(figsize=(10, 6))
plt.plot(time, y_test[:, 0], label='Ground Truth')
plt.plot(time, y_pred[:, 0], label='Prediction')
plt.title('Comparison of Output 1')
plt.xlabel('Time')
plt.ylabel('Output 1')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, y_test[:, 1], label='Ground Truth')
plt.plot(time, y_pred[:, 1], label='Prediction')
plt.title('Comparison of Output 2')
plt.xlabel('Time')
plt.ylabel('Output 2')
plt.legend()
plt.show()
