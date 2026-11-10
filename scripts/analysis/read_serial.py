"""Reset ESP32-S3 (watchdog) rồi đọc console tới khi gặp DONE, lưu ra file.

Dùng (trong môi trường ESP-IDF đã export):
    python scripts/analysis/read_serial.py COM8 experiments/phase1_analysis/esp32s3_bench_raw.txt [timeout_s]

Lưu ý Windows + cổng USB-Serial/JTAG: reset bằng RTS (mặc định của esptool/pyserial) đưa chip vào
DOWNLOAD mode. Dùng `esptool --after watchdog_reset` để khởi động lại vào app, rồi mở cổng với DTR/RTS = 0.
"""
import subprocess, sys, time
import serial

port, out = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 900
subprocess.run([sys.executable, "-m", "esptool", "--port", port, "--before", "default_reset",
                "--after", "watchdog_reset", "chip_id"], check=True, capture_output=True)
s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 1
s.dtr = False
s.rts = False
for _ in range(40):
    try:
        s.open()
        break
    except serial.SerialException:
        time.sleep(0.25)
t0 = time.time()
with open(out, "w", encoding="utf-8") as f:
    while time.time() - t0 < timeout:
        line = s.readline().decode("utf-8", "replace").rstrip()
        if not line:
            continue
        f.write(line + "\n")
        f.flush()
        if line.startswith(("BENCH", "FAIL", "INFO", "SKIP")) or "Guru" in line or "abort" in line.lower():
            print(line, flush=True)
        if line.strip() == "DONE":
            break
