# 【IoT】基礎設計課題：環境モニタリング

## 概要
執筆中...

<br>

## 1. 必要なハードウェア
- M5GO IoT スターターキット V2.7
- 環境センサユニット：SHT3X, QMP6988（M5GO IoTスターターキットに付属）
- SDカード (8GB以上, FAT32フォーマット)
- 高品質 USB-C to USB-C ケーブル（Intel Macでは必ず直挿し）

<br>

## 2. Arduino 環境構築
1. Arduino IDE 2.x 系をインストール
2. ボードのインストール:
  - ESP32 2.0.6
  > [!WARNING]
  > ボードのバージョンを最新にしないこと（最新のものはTensorFlowと非互換）
3. ライブラリのインストール:
  - M5Stack 0.4.6
  - TensorFlowLite_ESP32 1.0.0
4. 使用ボードをM5Stack-Core-ESP32に設定
5. 書き込み速度を115200に設定

<br>

## 3. Python 環境構築
1. pyenv で Python 3.11 系をインストール
2. 以下のライブラリをインストール
  - TensorFlow 2.15.0
  - Keras 2.15.0
  - scikit-learn 1.5.2
  - joblib 1.4.2
  - pandas 2.2.2
  - numpy 1.26.4
  > [!WARNING]
  > TensorFlowとKerasのバージョンに注意<br>
  > 一部PCでは最新のTensorFlow（2.17.0）に対応していないためKeras3以降と非互換です
  ---
  bash
  ```bash
  $ pip install tensorflow==2.15.0 Keras==2.15.0 scikit-learn==1.5.2 joblib==1.4.2 pandas==2.2.2 numpy==1.26.4
  ```
  ---

<br>

## 4. モデル学習・変換（作業ディレクトリはsrc/pythonを想定）
1.	擬似データ生成
2.	標準化（StandardScaler を使用）とデータセット分割
3.	快適度分類モデルの学習
4.	TensorFlow Lite (.tflite) 形式への変換
5.	xxd コマンドで model.tflite を C++ ヘッダ (model_data.cc) に変換
  ---
  bash
  ```bash
  $ xxd -i models/comfort_model.tflite > ../cpp/model_data.cc
  ```
  ---
6.	Arduino IDEでインポートするためのヘッダファイルを作成
  ---
  model_data.h
  ```h
  #ifndef MODEL_DATA_H_
  #define MODEL_DATA_H_

  extern const unsigned char comfort_model_tflite[];
  extern const unsigned int comfort_model_tflite_len;

  #endif  // MODEL_DATA_H_
  ```
  ---
7.  model_data.ccを編集（先頭に #include を追加、末尾の文頭に const を追加）
  ---
  model_data.cc
  ```cpp
  #include "model_data.h"

  // 中略

  const unsigned int  // 以下省略
  ```
  ---
8.  標準化用のスケーラをヘッダファイルに変換

<br>

## 5. M5GOへのデプロイ
1.	Arduino IDE で model_data.cc, model_data.h, scaler.h をプロジェクトに追加
2.	センサデータを取得し、標準化
3.	モデル推論を実行
4.	LCD にセンサ値と推論ラベル（comfortable/hot/cold）を表示

<br>

## 6. 動作確認
- M5GO の LCD 上で
- 温度、湿度、気圧が正しく表示されること
- 推論ラベル（comfortable/hot/cold）が表示されることを確認

<br>

## 7. トラブルシュート

書き込みエラーが出た場合
-	書き込み速度は必ず 115200 に設定
-	Intel Mac は macOS Ventura 以降を推奨
-	USB-C to USB-C直挿し（USB-A変換不可）
-	書き込み時 BOOTボタンは不要（自動書き込み可能）
-	他のプロセス（VSCode、シリアルモニタなど）がポートを使用していないことを確認
-	書き込み失敗時は USB ケーブル、ポートを変更して再試行
