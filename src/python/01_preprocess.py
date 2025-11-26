import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# JSONファイル読み込み
data = pd.read_json("dataset/comfort_pseudoData.json", orient = "records", lines = True)

# ラベルの数値化
le = LabelEncoder()
data["label_id"] = le.fit_transform(data["label"])

# 属性とラベルを分離
X = data[["temperature", "humidity", "pressure"]].values
y = data["label_id"].values

# スケーリング
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 標準化パラメータを出力
if not os.path.isdir("models"):
  os.mkdir("models")
joblib.dump(scaler, "models/scaler.pkl")

# 学習セット／テストセットを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# 学習セット／テストセットをDataFrameにまとめる
train_df = pd.DataFrame(X_train, columns = ["temperature", "humidity", "pressure"])
train_df["label"] = y_train
test_df = pd.DataFrame(X_test, columns = ["temperature", "humidity", "pressure"])
test_df["label"] = y_test

# JSON形式で保存（1行1レコード）
train_df.to_json("dataset/pseudo_trainSet.json", orient = "records", lines = True)
test_df.to_json("dataset/pseudo_testSet.json", orient = "records", lines = True)