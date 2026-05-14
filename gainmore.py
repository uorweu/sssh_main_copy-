import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# --- CẤU HÌNH ---
FS = 48000      # Tần số 48kHz cho Mic I2S ổn định
DURATION = 5    # Thu âm trong 5 giây
GAIN = 20.0     # Hệ số khuếch đại (tăng nếu vẫn thấy bé)
FILENAME = "test_thu_am.wav"

print(f"--- ĐANG CHUẨN BỊ THU ÂM TRONG {DURATION} GIÂY ---")
print("Hãy nói gì đó hoặc tạo tiếng động đi ông giáo!")

# 1. Thực hiện thu âm
# dtype='float32' giúp việc nhân Gain chính xác hơn
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')

# Đợi cho đến khi thu âm xong
sd.wait()

print("--- THU ÂM XONG! ĐANG XỬ LÝ VÀ LƯU FILE ---")

# 2. Áp dụng Digital Gain (Khuếch đại kỹ thuật số)
processed_audio = recording * GAIN

# 3. Giới hạn biên độ để không bị rè (Clipping)
processed_audio = np.clip(processed_audio, -1.0, 1.0)

# 4. Lưu file dưới dạng WAV 16-bit (chuẩn phổ biến)
# Chuyển đổi từ float32 sang int16 để các trình phát nhạc đều nghe được
audio_int16 = (processed_audio * 32767).astype(np.int16)
write(FILENAME, FS, audio_int16)

print(f"Hoàn tất! Ông kiểm tra file '{FILENAME}' trong thư mục nhé.")
