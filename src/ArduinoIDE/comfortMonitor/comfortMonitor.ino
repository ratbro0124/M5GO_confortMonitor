#include <M5Unified.h>
#include <TensorFlowLite_ESP32.h>
#include "model_data.h"
#include "scaler.h"

// TinyMLライブラリ関連（必要なヘッダを個別に追加）
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// センサ用
#include <SHT3X.h>
#include <QMP6988.h>

SHT3X sht3x;
QMP6988 qmp;

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

constexpr int tensor_arena_size = 8 * 1024;
uint8_t tensor_arena[tensor_arena_size];

// クラスラベル（順序に注意）
const char* label_table[] = {"cold", "comfortable", "hot"};

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Wire.begin();
  sht3x.begin();
  qmp.begin(&Wire, 0x70);  // 必要に応じて 0x56 に変更

  M5.Lcd.setTextSize(2);
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.println("TinyML Comfort Detector");

  // TensorFlow Lite 初期化
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  // エラーレポーター
  static tflite::ErrorReporter* error_reporter = nullptr;
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // モデルの取得
  const tflite::Model* model = tflite::GetModel(comfort_model_tflite);

  // 推論エンジンの初期化
  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      tensor_arena_size,
      error_reporter
  );
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  sht3x.update();
  qmp.update();

  float temp = sht3x.cTemp;
  float hum  = sht3x.humidity;
  float pres = qmp.pressure / 100.0;

  // 標準化して入力へ
  input->data.f[0] = (temp - MEAN_TEMPERATURE) / STD_TEMPERATURE;
  input->data.f[1] = (hum  - MEAN_HUMIDITY)    / STD_HUMIDITY;
  input->data.f[2] = (pres - MEAN_PRESSURE)    / STD_PRESSURE;

  // 推論実行
  interpreter->Invoke();

  // 出力処理（最大値のインデックス）
  int max_index = 0;
  float max_val = output->data.f[0];
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > max_val) {
      max_val = output->data.f[i];
      max_index = i;
    }
  }

  // LCD表示
  M5.Lcd.fillRect(0, 40, 320, 200, BLACK);
  M5.Lcd.setCursor(0, 40);
  M5.Lcd.printf("Temp: %.1f C\nHumi: %.1f %%\nPres: %.1f hPa\n", temp, hum, pres);
  M5.Lcd.setTextColor(GREEN, BLACK);
  M5.Lcd.printf("Prediction: %s (%.2f)\n", label_table[max_index], max_val);

  delay(2000);
}