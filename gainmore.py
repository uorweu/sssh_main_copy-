import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

FS = 48000
DURATION = 5
FILENAME = "test_super_thinh.wav"

print(f"--- ĐANG THU ÂM 'SIÊU THÍNH' TRONG {DURATION} GIÂY ---")

# Thu âm thô (Raw)
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()

# --- BỘ LỌC NÂNG CẤP ---
# 1. Khử DC Offset (Giúp âm thanh không bị lệch trục khi kích gain cao)
recording = recording - np.mean(recording)

# 2. Peak Normalization (Tự động kích âm lên kịch trần)
max_peak = np.max(np.abs(recording))
if max_peak > 0:
    # Kích lên mức cao nhất có thể
    processed_audio = recording / max_peak 
else:
    processed_audio = recording

# 3. Soft Clipping (Nếu muốn âm thanh nghe 'dày' hơn nữa)
processed_audio = np.tanh(processed_audio) 

# --- LƯU FILE ---
audio_int16 = (processed_audio * 32767).astype(np.int16)
write(FILENAME, FS, audio_int16)

print(f"Hoàn tất! File '{FILENAME}' đã được tối ưu độ nhạy.")
