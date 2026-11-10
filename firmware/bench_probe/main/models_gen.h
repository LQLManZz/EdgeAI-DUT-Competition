// Sinh tự động bởi scripts/analysis/gen_bench_models.py – không sửa tay
#pragma once
#include <stddef.h>
#include <stdint.h>

extern const uint8_t _binary_edgedet_c_center_144x192_tflite_start[];
extern const uint8_t _binary_edgedet_c_center_144x192_tflite_end[];
extern const uint8_t _binary_edgedet_c_center_192x256_tflite_start[];
extern const uint8_t _binary_edgedet_c_center_192x256_tflite_end[];
extern const uint8_t _binary_edgedet_c_center_240x320_tflite_start[];
extern const uint8_t _binary_edgedet_c_center_240x320_tflite_end[];
extern const uint8_t _binary_edgedet_c_center_96x128_tflite_start[];
extern const uint8_t _binary_edgedet_c_center_96x128_tflite_end[];
extern const uint8_t _binary_edgedet_c_yolo_144x192_tflite_start[];
extern const uint8_t _binary_edgedet_c_yolo_144x192_tflite_end[];
extern const uint8_t _binary_edgedet_p_center_144x192_tflite_start[];
extern const uint8_t _binary_edgedet_p_center_144x192_tflite_end[];
extern const uint8_t _binary_edgedet_p_center_192x256_tflite_start[];
extern const uint8_t _binary_edgedet_p_center_192x256_tflite_end[];
extern const uint8_t _binary_edgedet_p_center_240x320_tflite_start[];
extern const uint8_t _binary_edgedet_p_center_240x320_tflite_end[];
extern const uint8_t _binary_edgedet_p_center_96x128_tflite_start[];
extern const uint8_t _binary_edgedet_p_center_96x128_tflite_end[];
extern const uint8_t _binary_edgedet_p_yolo_144x192_tflite_start[];
extern const uint8_t _binary_edgedet_p_yolo_144x192_tflite_end[];

struct BenchModel { const char* name; const uint8_t* data; size_t size; };
static const BenchModel kModels[] = {
  {"edgedet_c_center_144x192", _binary_edgedet_c_center_144x192_tflite_start, (size_t)(_binary_edgedet_c_center_144x192_tflite_end - _binary_edgedet_c_center_144x192_tflite_start)},
  {"edgedet_c_center_192x256", _binary_edgedet_c_center_192x256_tflite_start, (size_t)(_binary_edgedet_c_center_192x256_tflite_end - _binary_edgedet_c_center_192x256_tflite_start)},
  {"edgedet_c_center_240x320", _binary_edgedet_c_center_240x320_tflite_start, (size_t)(_binary_edgedet_c_center_240x320_tflite_end - _binary_edgedet_c_center_240x320_tflite_start)},
  {"edgedet_c_center_96x128", _binary_edgedet_c_center_96x128_tflite_start, (size_t)(_binary_edgedet_c_center_96x128_tflite_end - _binary_edgedet_c_center_96x128_tflite_start)},
  {"edgedet_c_yolo_144x192", _binary_edgedet_c_yolo_144x192_tflite_start, (size_t)(_binary_edgedet_c_yolo_144x192_tflite_end - _binary_edgedet_c_yolo_144x192_tflite_start)},
  {"edgedet_p_center_144x192", _binary_edgedet_p_center_144x192_tflite_start, (size_t)(_binary_edgedet_p_center_144x192_tflite_end - _binary_edgedet_p_center_144x192_tflite_start)},
  {"edgedet_p_center_192x256", _binary_edgedet_p_center_192x256_tflite_start, (size_t)(_binary_edgedet_p_center_192x256_tflite_end - _binary_edgedet_p_center_192x256_tflite_start)},
  {"edgedet_p_center_240x320", _binary_edgedet_p_center_240x320_tflite_start, (size_t)(_binary_edgedet_p_center_240x320_tflite_end - _binary_edgedet_p_center_240x320_tflite_start)},
  {"edgedet_p_center_96x128", _binary_edgedet_p_center_96x128_tflite_start, (size_t)(_binary_edgedet_p_center_96x128_tflite_end - _binary_edgedet_p_center_96x128_tflite_start)},
  {"edgedet_p_yolo_144x192", _binary_edgedet_p_yolo_144x192_tflite_start, (size_t)(_binary_edgedet_p_yolo_144x192_tflite_end - _binary_edgedet_p_yolo_144x192_tflite_start)},
};
static const int kNumModels = sizeof(kModels) / sizeof(kModels[0]);
