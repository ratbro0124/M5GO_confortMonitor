import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# 学習用JSONを読み込み
df = pd.read_json("dataset/pseudo_trainSet.json", orient = "records", lines = True)

X = df[["temperature", "humidity", "pressure"]].values
y = df["label"].values

# 学習と検証に再分割
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size = 0.2, random_state = 42)

# モデル定義
model = tf.keras.Sequential([
  tf.keras.layers.Input(shape = (3, )),             # 入力層（温度、湿度、気圧を入力）
  tf.keras.layers.Dense(8, activation = 'relu'),    # 中間層
  tf.keras.layers.Dense(6, activation = 'relu'),    # 中間層
  tf.keras.layers.Dense(3, activation = 'softmax')  # 出力層（3クラス分類）
])

model.compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy']
)

# 学習
history = model.fit(
  X_train, y_train,
  validation_data = (X_eval, y_eval),
  epochs = 30,
  batch_size = 16
)

# テストデータ読み込み
test_df = pd.read_json("dataset/pseudo_testSet.json", orient="records", lines=True)
# 特徴量とラベルを抽出
X_test = test_df[["temperature", "humidity", "pressure"]].values
y_test = test_df["label"].values

# 精度評価
loss, acc = model.evaluate(X_test, y_test)
print(f"Evaluation accuracy: {acc:.3f}")

# 学習済みモデルを出力
if not os.path.isdir("models"):
  os.mkdir("models")
model.save("models/comfort_model.keras")