import joblib

# スケーラー読み込み
scaler = joblib.load("models/scaler.pkl")

# 属性ラベル
colnames = ["temperature", "humidity", "pressure"]

# 出力ファイルを開く
with open("../cpp/scaler.h", "w") as f:
  f.write("#ifndef SCALER_H_\n#define SCALER_N_\n\n")
  f.write("// 自動生成された標準化パラメータ\n\n")

  for i, name in enumerate(colnames):
    mean = scaler.mean_[i]
    std = scaler.scale_[i]
    macro_name = name.upper()
    f.write(f"#define MEAN_{macro_name} {mean:.6f}\n")
    f.write(f"#define STD_{macro_name} {std:.6f}\n\n")

  f.write("#endif  // SCALER_H_\n")