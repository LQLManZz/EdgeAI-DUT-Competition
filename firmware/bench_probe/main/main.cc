// Benchmark các model probe Giai đoạn 1 trên ESP32-S3 bằng TFLite Micro (+ ESP-NN).
// Với mỗi model:
//   1) AllocateTensors (arena trong PSRAM) -> arena_used_bytes -> đo Invoke (warm-up + kRuns lần)
//   2) Chạy lại bằng RecordingMicroInterpreter + MicroProfiler: in phân rã arena và thời gian theo loại op
//   3) Nếu arena vừa SRAM nội thì đo lại với arena trong SRAM nội
// Output CSV qua console (USB-Serial/JTAG):
//   BENCH,<model>,<PSRAM|SRAM>,<arena_used>,<avg_ms>,<min_ms>,<max_ms>
//   OPSBEGIN/OPSEND bao quanh bảng ticks theo op; ALLOCBEGIN/ALLOCEND bao quanh phân rã arena
//   FAIL,<model>,<stage>   SKIP,<model>,SRAM,need=..,largest=..
#include <cstdio>
#include <cstring>
#include <new>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_chip_info.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "models_gen.h"

constexpr size_t kPsramArena = 3 * 1024 * 1024;
constexpr int kRuns = 5;

using Resolver = tflite::MicroMutableOpResolver<20>;
static Resolver g_resolver;
static tflite::MicroProfiler* g_profiler;  // ~48 KB -> đặt trong PSRAM, không chiếm SRAM nội

static void AddOps(Resolver& r) {
  // Không đăng ký TILE: TFLM không có kernel TILE (đây là điều cần chứng minh).
  r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddAdd(); r.AddConcatenation();
  r.AddResizeNearestNeighbor(); r.AddMaxPool2D(); r.AddReshape(); r.AddTranspose();
  r.AddStridedSlice(); r.AddPack(); r.AddShape(); r.AddHardSwish(); r.AddMean();
  r.AddMul(); r.AddQuantize(); r.AddDequantize(); r.AddExpandDims();
}

static void FillInput(TfLiteTensor* t) {
  uint32_t s = 12345;
  for (size_t i = 0; i < t->bytes; i++) { s = s * 1103515245u + 12345u; t->data.int8[i] = (int8_t)(s >> 24); }
}

// Đo thời gian. Trả về arena_used (0 nếu lỗi).
static size_t Bench(const BenchModel& m, uint8_t* arena, size_t arena_size, const char* mem) {
  const tflite::Model* model = tflite::GetModel(m.data);
  if (model->version() != TFLITE_SCHEMA_VERSION) { printf("FAIL,%s,schema\n", m.name); return 0; }
  auto* interp = new tflite::MicroInterpreter(model, g_resolver, arena, arena_size);
  if (interp->AllocateTensors() != kTfLiteOk) {
    // Không delete: destructor sau AllocateTensors lỗi có thể truy cập vùng chưa khởi tạo (crash đã gặp).
    printf("FAIL,%s,AllocateTensors(%s)\n", m.name, mem);
    return 0;
  }
  size_t used = interp->arena_used_bytes();
  FillInput(interp->input(0));
  if (interp->Invoke() != kTfLiteOk) { printf("FAIL,%s,Invoke\n", m.name); return 0; }  // warm-up
  int64_t tot = 0, mn = INT64_MAX, mx = 0;
  for (int i = 0; i < kRuns; i++) {
    int64_t t0 = esp_timer_get_time();
    interp->Invoke();
    int64_t dt = esp_timer_get_time() - t0;
    tot += dt; mn = dt < mn ? dt : mn; mx = dt > mx ? dt : mx;
  }
  printf("BENCH,%s,%s,%u,%.2f,%.2f,%.2f\n", m.name, mem, (unsigned)used,
         tot / 1000.0 / kRuns, mn / 1000.0, mx / 1000.0);
  delete interp;
  return used;
}

// Phân rã arena + profile theo op (arena PSRAM).
static void Profile(const BenchModel& m, uint8_t* arena, size_t arena_size) {
  const tflite::Model* model = tflite::GetModel(m.data);
  g_profiler->ClearEvents();
  auto* interp = new tflite::RecordingMicroInterpreter(model, g_resolver, arena, arena_size, nullptr, g_profiler);
  if (interp->AllocateTensors() != kTfLiteOk) return;
  printf("ALLOCBEGIN,%s\n", m.name);
  interp->GetMicroAllocator().PrintAllocations();
  printf("ALLOCEND,%s\n", m.name);
  FillInput(interp->input(0));
  interp->Invoke();
  g_profiler->ClearEvents();
  interp->Invoke();
  printf("OPSBEGIN,%s,ticks_per_s=%ld\n", m.name, (long)tflite::ticks_per_second());
  g_profiler->LogTicksPerTagCsv();
  printf("OPSEND,%s\n", m.name);
  delete interp;
}

static void BenchTask(void*) {
  vTaskDelay(pdMS_TO_TICKS(1500));
  AddOps(g_resolver);
  g_profiler = new (heap_caps_malloc(sizeof(tflite::MicroProfiler), MALLOC_CAP_SPIRAM)) tflite::MicroProfiler();
  esp_chip_info_t ci; esp_chip_info(&ci);
  printf("\nINFO,chip_rev=%d,cores=%d,cpu_mhz=%d,dcache=%d,dcache_line=%d\n", ci.revision, ci.cores,
         CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, CONFIG_ESP32S3_DATA_CACHE_SIZE, CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE);
  printf("INFO,free_internal=%u,largest_internal=%u,free_psram=%u\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
         (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  uint8_t* psram = (uint8_t*)heap_caps_aligned_alloc(16, kPsramArena, MALLOC_CAP_SPIRAM);
  if (!psram) { printf("FAIL,all,psram_alloc\n"); vTaskDelete(nullptr); }

  for (int i = 0; i < kNumModels; i++) {
    BenchModel m = kModels[i];
    // EMBED_FILES không đảm bảo căn lề 16 byte; flatbuffer cần căn lề -> copy sang PSRAM nếu lệch.
    uint8_t* copy = nullptr;
    if ((uintptr_t)m.data % 16) {
      copy = (uint8_t*)heap_caps_aligned_alloc(16, m.size, MALLOC_CAP_SPIRAM);
      memcpy(copy, m.data, m.size);
      m.data = copy;
      printf("ALIGN,%s,copied_to_psram\n", m.name);
    }
    size_t used = Bench(m, psram, kPsramArena, "PSRAM");
    if (used) {
      Profile(m, psram, kPsramArena);
      size_t need = used + 1024;
      size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      if (need + 16 * 1024 < largest) {  // chừa 16 KB cho hệ thống
        uint8_t* sram = (uint8_t*)heap_caps_aligned_alloc(16, need, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (sram) { Bench(m, sram, need, "SRAM"); heap_caps_free(sram); }
      } else {
        printf("SKIP,%s,SRAM,need=%u,largest=%u\n", m.name, (unsigned)need, (unsigned)largest);
      }
    }
    if (copy) heap_caps_free(copy);
  }
  printf("DONE\n");
  vTaskDelete(nullptr);
}

extern "C" void app_main(void) {
  // TFLM cần stack lớn hơn main task mặc định -> task riêng
  xTaskCreatePinnedToCore(BenchTask, "bench", 32 * 1024, nullptr, 5, nullptr, 1);
}
