import numpy as np
import pandas as pd
import os

# ①　擬似データセット生成パラメータ
np.random.seed(42)
num_samples = 300

# confortable擬似データ生成
temp_comf = np.random.normal(24.5, 1.0, num_samples)
humid_comf = np.random.normal(45, 5, num_samples)
pres_comf = np.random.normal(1013, 1.0, num_samples)
label_comf = ["comfortable"] * num_samples

# hot擬似データ生成
temp_hot = np.random.normal(29, 1.0, num_samples)
humid_hot = np.random.normal(60, 5, num_samples)
pres_hot = np.random.normal(1010, 1.0, num_samples)
label_hot = ["hot"] * num_samples

# cold擬似データ生成
temp_cold = np.random.normal(20, 1.0, num_samples)
humid_cold = np.random.normal(50, 5, num_samples)
pres_cold = np.random.normal(1015, 1.0, num_samples)
label_cold = ["cold"] * num_samples

# データ統合
data = pd.DataFrame({
    "temperature": np.concatenate([temp_comf, temp_hot, temp_cold]),
    "humidity":    np.concatenate([humid_comf, humid_hot, humid_cold]),
    "pressure":    np.concatenate([pres_comf, pres_hot, pres_cold]),
    "label":       label_comf + label_hot + label_cold
})

# JSON形式で保存（行ごとに辞書形式）
if not os.path.isdir("dataset"):
  os.mkdir("dataset")
data.to_json("dataset/comfort_pseudoData.json", orient = "records", lines = True)