import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter

# Cấu hình
FS = 48000
DURATION = 10 
FILENAME = "max_tam_2m.wav"

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_filter(data, cutoff=200, fs=48000):
    b, a = butter_highpass(cutoff, fs, order=5)
    return lfilter(b, a, data)

print("--- ĐANG THU ÂM CHẾ ĐỘ 'VIỄN THÁM' 2M ---")
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()

# 1. Lọc bỏ nhiễu nền tần số thấp (Cực kỳ quan trọng để thu xa)
filtered_audio = apply_filter(recording.flatten(), cutoff=200, fs=FS)

# 2. Peak Normalization (Đẩy kịch trần)
max_val = np.max(np.abs(filtered_audio))
if max_val > 0.0001: # Tránh chia cho 0
    processed_audio = filtered_audio / max_val
else:
    processed_audio = filtered_audio

# 3. Soft Limiter (Dùng hàm tanh để làm dày âm thanh ở xa)
processed_audio = np.tanh(processed_audio * 2) 

# Lưu file
audio_int16 = (processed_audio * 32767).astype(np.int16)
write(FILENAME, FS, audio_int16)
print(f"Xong! Thử đứng xa 2m và nói thầm xem nhé.")
