import tensorflow as tf
from keras.models import load_model

# モデルの読み込み
model = load_model("models/comfort_model.keras")

# TFLiteへ変換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ファイルとして保存
with open("models/comfort_model.tflite", "wb") as f:
  f.write(tflite_model)

  print("☑️　TFLite model saved as comfort_model.tflite")