import numpy as np
from scipy.io import wavfile

# 알람 소리 생성
sample_rate = 44100
duration = 1.0  # 1초
t = np.linspace(0, duration, int(sample_rate * duration), False)

# 440Hz와 880Hz의 사인파를 합성
note1 = np.sin(2 * np.pi * 440 * t)
note2 = np.sin(2 * np.pi * 880 * t)
alarm_sound = note1 + note2

# 볼륨 조정
alarm_sound = alarm_sound * 0.3

# WAV 파일로 저장
wavfile.write('alarm.wav', sample_rate, alarm_sound.astype(np.float32)) 