import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, filtfilt
from scipy.signal import spectrogram

# 读取音频文件
# audio_data, fs = librosa.load('D:/my/DARE/MOTOR/Data/700：1_assembled/Motor_Broken_3.wav', sr=None)  # 替换为你的音频文件路径
audio_data, fs = librosa.load('D:/my/DARE/MOTOR/Data/lc3xs-iu70g.wav', sr=None)  # 替换为你的音频文件路径
# 定义Butterworth低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist 频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# 低通滤波：保留低频
low_cutoff = 100  # 设置低通滤波的截止频率
low_filtered_data = butter_lowpass_filter(audio_data, low_cutoff, fs)

# 计算短时傅里叶变换（STFT）
def compute_stft(data, fs, n_fft=2048, hop_length=256):
    stft_result = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
    return stft_result, librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=fs, hop_length=hop_length)

# 绘制STFT
stft_result, time_axis = compute_stft(low_filtered_data, fs)
plt.figure(figsize=(10, 4), dpi=150)
librosa.display.specshow(librosa.amplitude_to_db(stft_result, ref=np.max), sr=fs, x_axis='time', y_axis='log')
plt.colorbar(label='Decibels (dB)')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (log scale)")
plt.ylim(0, 100)  # 限制纵轴范围为0-100
plt.title("STFT Representation of Filtered Audio")
plt.show()
